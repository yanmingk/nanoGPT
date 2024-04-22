import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class CausalSelfMLAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_pdrop = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        # # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # for MLA
        self.m = config.m
        self.scale_attn = config.scale_attn
        self.block_size = config.block_size
        self.attn = MLA_tvm(self.block_size, self.m, self.n_embd//self.n_head, config.p, self.attn_pdrop, self.scale_attn, config.downsampling)
        

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # make contiguous for MLA
        k = k.contiguous().view(B*self.n_head, T, C // self.n_head)
        q = q.contiguous().view(B*self.n_head, T, C // self.n_head)
        v = v.contiguous().view(B*self.n_head, T, C // self.n_head)
        # print(k.shape,q.shape, v.shape)

        q, cs = self.attn(q,k,v)
        q = q.view(B, self.n_head, T, C // self.n_head)
        q = q.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        return self.resid_dropout(self.c_proj(q)), None
        # return self.resid_dropout(self.c_proj(q)), cs

        # # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # att = self.attn_dropout(att)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # # output projection
        # y = self.resid_dropout(self.c_proj(y))
        # return y




class MLA_tvm(torch.nn.Module):
    def __init__(self, n:int, m:int, d:int, p:int, attn_pdrop, scale_attn, downsampling):
        '''
            n need to be fix for a module
            Q,K,V: (b,n,d)
        '''

        self.n, self.m, self.d, self.p, self.attn_pdrop, self.scale_attn, self.downsampling = n, m, d, p, attn_pdrop, scale_attn, downsampling
        super(MLA_tvm, self).__init__()
        assert(n % m == 0), "n must be divisible by m"
        assert(n//m & (n//m-1) == 0), "n/m must be power of 2"

        # number of coarse levels
        self.L = int(math.log2(int(n/m))) - 1
        assert(m*2**(self.L-1) == n//4), "coarsest grid size is n/4?"

        # sequence lengths for all levels
        # self.nls = [n] + [ n // (m*2**l) for l in range(self.L)]

        # weights for averaging/convolution. queries remain the fine resolution (n)
        # conv_k = [torch.nn.Parameter(torch.ones((d, 1, m))/(m))]
        # conv_v = [torch.nn.Parameter(torch.ones((d, 1, m))/(m))]
        # for l in range(self.L): # 0, ..., log(n/m)-2
        #     stride = m*2**(l+1)
        #     conv_k.append(torch.nn.Parameter( torch.ones((d, 1, stride))/(stride) ))
        #     conv_v.append(torch.nn.Parameter( torch.ones((d, 1, stride))/(stride) ))
        # self.conv_k = torch.nn.ParameterList(conv_k)
        # self.conv_v = torch.nn.ParameterList(conv_v)


        if self.downsampling == "groupedconv":
            print("MLA using grouped conv on k,v")
            conv_k = []
            conv_v = []
            for l in range(self.L): # 0, ..., log2(n/m)-2
                w = m*(2**l) # m, m*2, m*4, m*8, ..., n/4
                conv_k.append(nn.Conv1d(d, p*d, w, stride=w, groups=d))
                conv_v.append(nn.Conv1d(d, p*d, w, stride=w, groups=d))
            self.conv_k = nn.ModuleList(conv_k)
            self.conv_v = nn.ModuleList(conv_v)
        elif self.downsampling == "conv":
            print("MLA using conv on k,v")
            conv_k = []
            conv_v = []
            for l in range(self.L): # 0, ..., log2(n/m)-2
                w = m*(2**l) # m, m*2, m*4, m*8, ..., n/4
                conv_k.append(nn.Conv1d(d, p*d, w, stride=w))
                conv_v.append(nn.Conv1d(d, p*d, w, stride=w))
            self.conv_k = nn.ModuleList(conv_k)
            self.conv_v = nn.ModuleList(conv_v)
        elif self.downsampling == "avgpool":
            print("MLA using avgpool on k,v")
            ### pooling
            pool_k = []
            pool_v = []
            for l in range(self.L): # 0, ..., log2(n/m)-2
                w = m*(2**l) # m, m*2, m*4, m*8, ..., n/4
                pool_k.append(nn.AvgPool1d(w//p, stride=w//p))
                pool_v.append(nn.AvgPool1d(w//p, stride=w//p))
            self.pool_k = nn.ModuleList(pool_k)
            self.pool_v = nn.ModuleList(pool_v)
        else:
            raise NotImplementedError        


            # conv_k = [nn.Conv1d(d, p*d, m, m)]
            # conv_v = [nn.Conv1d(d, p*d, m, m)]
            # for l in range(self.L - 1): # 0, ..., log2(n/m)-2-1
            #     stride = m*2**(l+1) # m*2, m*4, m*8, ..., n/4
            #     conv_k.append(nn.Conv1d(d, p*d, stride, stride))
            #     conv_v.append(nn.Conv1d(d, p*d, stride, stride))
            #     # conv_k.append(nn.Conv1d(d, 2*d, stride, stride))
            #     # conv_v.append(nn.Conv1d(d, 2*d, stride, stride))
            # self.conv_k = nn.ModuleList(conv_k)
            # self.conv_v = nn.ModuleList(conv_v)



        return

    # @staticmethod
    def stage1(self, Q, Kls):
        '''
            Compute product matrices 
            Q*K_l^T/sqrt(d)
            then take softmax for each query
            across all levels

            seperate: high level weights are counted as multiple values
                      in softmax if True.
        '''

        # L, m, nls, attn_pdrop = self.L, self.m, self.nls, self.attn_pdrop
        L, m, attn_pdrop, scale_attn, p= self.L, self.m, self.attn_pdrop, self.scale_attn, self.p
        B, n, d = Q.shape
        device = Q.device

        As = []
        Cs = []
        exp_sums = torch.zeros((B, n, L+1)).to(device)
        max_attn_weights = torch.zeros((B, n, L+1)).to(device)
        min_attn_weights = torch.zeros((B, n, L+1)).to(device)

        for l in range(L+1):
            Kl = Kls[l]
            A = diagonaled_mm_qk_fine_ltr(Q, Kl, m) if l==0 else diagonaled_mm_qk_coarse_ltr(Q, Kl, self.p)
            As.append(A)
            # As[l] = As[l].reshape(As[l].shape[0], As[l].shape[1]*As[l].shape[2], As[l].shape[3])

        # max attention weight for each query in each sequence, in every level.
        for l in range(L+1):
            max_attn_weights[:, :, l] = torch.max(As[l], dim=2, keepdim=False)[0]
            min_attn_weights[:, :, l] = torch.min(As[l], dim=2, keepdim=False)[0]
        # max attention weight for each query in each sequence, across all levels. shape (B,n,1)
        max_attn_weights_all_levels = torch.max(max_attn_weights, dim=2, keepdim=True)[0]
        min_attn_weights_all_levels = torch.min(max_attn_weights, dim=2, keepdim=True)[0]

        Bs = []
        # subtract by max weight.
        for l in range(L+1):
            Bs.append( As[l] - min_attn_weights_all_levels)
            # As[l] = As[l] - max_attn_weights_all_levels

        # compute softmax along dim 2.
        for l in range(L+1):
            # window_size = int(n/nls[l])
            # if seperate:
            #     exp_sums[:, :, l] = torch.sum(torch.exp(As[l]), dim=2) * window_size
            # else:
            #     exp_sums[:, :, l] = torch.sum(torch.exp(As[l]), dim=2)
            exp_sums[:, :, l] = torch.sum(torch.exp(Bs[l]), dim=2)
            if scale_attn and l>=1:
                exp_sums[:, :, l] *= (m*(2**(l-1))) // p # multiply by the window size
        exp_sum = torch.sum(exp_sums, dim=2, keepdim=True)

        for l in range(L+1):
            scale =  (m*(2**(l-1))) // p if (l!=0 and scale_attn) else 1
            Cs.append(scale * torch.exp(Bs[l]) / exp_sum)
            # dropout
            Cs[l] = torch.nn.functional.dropout(Cs[l], p=attn_pdrop, training = self.training)


        # return Cs, As
        return Cs, None

    def convolute(self, K, V):
        '''
            Compute K_l and V_l for all levels.
        '''
        Kl, Vl = K, V
        Kls = [Kl]
        Vls = [Vl]
        b,n,d = K.shape
        p = self.p

        for l in range(self.L):
            w = self.m*(2**l)
            nl = p * (n//w)
            if self.downsampling == "groupedconv" or self.downsampling == "conv":
                Kl = self.conv_k[l](K.transpose(1,2))
                assert(Kl.shape == (b, d*p, n//w)), f"conv{l} out shape{Kl.shape} neq {(b, d*p, n//w)}."
                # Kl = Kl.view((b,d,(n*2)//w))
                Kl = Kl.view((b, d, p, n//w))
                Kl = Kl.transpose(1,3)
                Kl = Kl.reshape((b, nl, d)).contiguous()

                Vl = self.conv_v[l](V.transpose(1,2))
                assert(Vl.shape == (b, d*p, n//w)), f"conv{l} out shape{Vl.shape} neq {(b, d*p, n//w)}."
                Vl = Vl.view((b, d, p, n//w))
                Vl = Vl.transpose(1,3)
                Vl = Vl.reshape((b, nl, d)).contiguous()
                Kls.append(Kl)
                Vls.append(Vl)         
            elif self.downsampling == "avgpool":
                Kl = self.pool_k[l](K.transpose(1,2)) # out: (b,d,np/w)
                assert(Kl.shape == (b, d, (n//w)*p)), f"conv{l} out shape{Kl.shape} neq {(b, d*p, n//w)}."
                Kl = Kl.transpose(1, 2).contiguous()# out: (b, np/w, d)
                Vl = self.pool_v[l](V.transpose(1,2)) # out: (b,d,np/w)
                assert(Vl.shape == (b, d, (n//w)*p)), f"conv{l} out shape{Vl.shape} neq {(b, d*p, n//w)}."
                Vl = Vl.transpose(1, 2).contiguous()# out: (b, np/w, d)
                Kls.append(Kl)
                Vls.append(Vl)     

            # Vl = self.conv_v[l](V.transpose(1,2))
            # # Vl = Vl.view((b,d,(n*2)//w))
            # Vl = Vl.transpose(1,2).contiguous()
            # Vls.append(Vl)

            # Kl = torch.nn.functional.conv1d(
            #     K.transpose(1,2),
            #     self.conv_k[l],
            #     # self.conv_k[l].expand((self.d, 1, -1)),
            #     stride=stride, 
            #     padding=0, 
            #     dilation=1, 
            #     ).transpose(1,2)
            #     # groups=self.d).transpose(1,2)
            # Kls.append(Kl.contiguous())

            # Vl = torch.nn.functional.conv1d(
            #     V.transpose(1,2),
            #     self.conv_v[l],
            #     # self.conv_v[l].expand((self.d, 1, -1)),
            #     stride=stride,
            #     padding=0,
            #     dilation=1,
            #     groups=self.d).transpose(1,2)
            # Vls.append(Vl.contiguous())
        return Kls, Vls

    def stage2(self, Cs, Vls):
        '''
            calculate C@V
        '''
        m = self.m
        for l in range(self.L + 1):
            if l==0:
                result = diagonaled_mm_cv_fine_ltr(Cs[l], Vls[l], m)   
            else:
                result_l = diagonaled_mm_cv_coarse_ltr(Cs[l], Vls[l], self.p)
                # scaling already taken into account in attn matrix, don't scale again here
                # if scale_attn:
                #     result_l *= m*2**(l-1) # multiply by the grid size
                result += result_l
        return result

    def forward(self, Q, K, V):
        Q = Q / math.sqrt(self.d)
        Kls, Vls = self.convolute(K, V)
        Cs, As = self.stage1( Q, Kls)
        result = self.stage2(Cs, Vls)
        # return result, Cs
        # return result, As
        return result, None








