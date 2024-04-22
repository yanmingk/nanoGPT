import torch
from mm import qk_fine, qk_coarse, cv_fine, cv_coarse
import math
from torch import nn

class fma(torch.nn.Module):
    def __init__(self, n:int, m:int, d:int, p:int, attn_pdrop, scale_attn, downsampling):
        '''
            n need to be fix for a module
            q,k,v: (b,n,d)
        '''

        self.n, self.m, self.d, self.p, self.attn_pdrop, self.scale_attn, self.downsampling = n, m, d, p, attn_pdrop, scale_attn, downsampling
        super(fma, self).__init__()
        assert(n % m == 0), "n must be divisible by m"
        assert(n//m & (n//m-1) == 0), "n/m must be power of 2"

        self.levels = int(math.log2(int(n/m))) - 1 # number of coarse levels
        assert(m*2**(self.levels-1) == n//4), "coarsest grid size is n/4?"
        print(f"using {self.downsampling} for downsampling, levels={self.levels}")
        conv_k = []
        conv_v = []
        for l in range(self.levels): # 0, ..., log2(n/m)-2
            w = m*(2**l)    # m, m*2, m*4, m*8, ..., n/4
            if self.downsampling == "groupedconv":
                conv_k.append(nn.conv1d(d, p*d, w, stride=w, groups=d))
                conv_v.append(nn.conv1d(d, p*d, w, stride=w, groups=d))
            elif self.downsampling == "conv":
                conv_k.append(nn.conv1d(d, p*d, w, stride=w))
                conv_v.append(nn.conv1d(d, p*d, w, stride=w))
            elif self.downsampling == "avgpool":
                conv_k.append(nn.avgPool1d(w//p, stride=w//p))
                conv_v.append(nn.avgPool1d(w//p, stride=w//p))
            else:
                raise NotImplementedError
        self.conv_k = nn.ModuleList(conv_k)
        self.conv_v = nn.ModuleList(conv_v)



    # @staticmethod
    def stage1(self, q, kls):
        levels, m, attn_pdrop, scale_attn, p = self.levels, self.m, self.attn_pdrop, self.scale_attn, self.p
        b, n, d = q.shape
        device = q.device

        attns = []
        cs = []
        exp_sums = torch.zeros((b, n, levels+1)).to(device)
        max_attn_weights = torch.zeros((b, n, levels+1)).to(device)
        min_attn_weights = torch.zeros((b, n, levels+1)).to(device)

        for l in range(levels+1):
            kl = kls[l]
            num_blocks = n // (m*(2**l))
            if l==0:
                c = torch.zeros((b, num_blocks, m, 3*m)).to(device)
                qk_fine(q, kl, c, num_blocks, m) #(b, num_blocks, m, 3*m)
                c = c.view((b, n, 3*m)).contiguous()
            else:
                c = torch.zeros((b, num_blocks, m, 4*p)).to(device)
                qk_coarse(q, kl, c, num_blocks, p) #(b, num_blocks, m, 4*p)
                c = c.view((b, n, 4*p)).contiguous()
            attns.append(c)
        # TODO: check if this is correct



        # max attention weight for each query in each sequence, in every level.
        for l in range(levels+1):
            max_attn_weights[:, :, l] = torch.max(attns[l], dim=2, keepdim=False)[0]
            min_attn_weights[:, :, l] = torch.min(attns[l], dim=2, keepdim=False)[0]
        # max attention weight for each query in each sequence, across all levels. shape (b,n,1)
        max_attn_weights_all_levels = torch.max(max_attn_weights, dim=2, keepdim=True)[0]
        min_attn_weights_all_levels = torch.min(max_attn_weights, dim=2, keepdim=True)[0]

        Bs = []
        # subtract by max weight.
        for l in range(levels+1):
            Bs.append( attns[l] - min_attn_weights_all_levels)
            # attns[l] = attns[l] - max_attn_weights_all_levels

        # compute softmax along dim 2.
        for l in range(levels+1):
            # window_size = int(n/nls[l])
            # if seperate:
            #     exp_sums[:, :, l] = torch.sum(torch.exp(attns[l]), dim=2) * window_size
            # else:
            #     exp_sums[:, :, l] = torch.sum(torch.exp(attns[l]), dim=2)
            exp_sums[:, :, l] = torch.sum(torch.exp(Bs[l]), dim=2)
            if scale_attn and l>=1:
                exp_sums[:, :, l] *= (m*(2**(l-1))) // p # multiply by the window size
        exp_sum = torch.sum(exp_sums, dim=2, keepdim=True)

        for l in range(levels+1):
            scale =  (m*(2**(l-1))) // p if (l!=0 and scale_attn) else 1
            cs.append(scale * torch.exp(Bs[l]) / exp_sum)
            # dropout
            cs[l] = torch.nn.functional.dropout(cs[l], p=attn_pdrop, training = self.training)


        # return cs, attns
        return cs, None

    def convolute(self, k, v):
        kl, vl = k, v
        kls = [kl]
        vls = [vl]
        b,n,d = k.shape
        p = self.p

        for l in range(self.levels):
            w = self.m*(2**l)
            nl = p * (n//w)
            if self.downsampling == "groupedconv" or self.downsampling == "conv":
                kl = self.conv_k[l](k.transpose(1,2))
                assert(kl.shape == (b, d*p, n//w)), f"conv{l} out shape{kl.shape} neq {(b, d*p, n//w)}."
                # kl = kl.view((b,d,(n*2)//w))
                kl = kl.view((b, d, p, n//w))
                kl = kl.transpose(1,3)
                kl = kl.reshape((b, nl, d)).contiguous()

                vl = self.conv_v[l](v.transpose(1,2))
                assert(vl.shape == (b, d*p, n//w)), f"conv{l} out shape{vl.shape} neq {(b, d*p, n//w)}."
                vl = vl.view((b, d, p, n//w))
                vl = vl.transpose(1,3)
                vl = vl.reshape((b, nl, d)).contiguous()
                kls.append(kl)
                vls.append(vl)         
            elif self.downsampling == "avgpool":
                kl = self.pool_k[l](k.transpose(1,2)) # out: (b,d,np/w)
                assert(kl.shape == (b, d, (n//w)*p)), f"conv{l} out shape{kl.shape} neq {(b, d*p, n//w)}."
                kl = kl.transpose(1, 2).contiguous()# out: (b, np/w, d)
                vl = self.pool_v[l](v.transpose(1,2)) # out: (b,d,np/w)
                assert(vl.shape == (b, d, (n//w)*p)), f"conv{l} out shape{vl.shape} neq {(b, d*p, n//w)}."
                vl = vl.transpose(1, 2).contiguous()# out: (b, np/w, d)
                kls.append(kl)
                vls.append(vl)     

            # vl = self.conv_v[l](v.transpose(1,2))
            # # vl = vl.view((b,d,(n*2)//w))
            # vl = vl.transpose(1,2).contiguous()
            # vls.append(vl)

            # kl = torch.nn.functional.conv1d(
            #     k.transpose(1,2),
            #     self.conv_k[l],
            #     # self.conv_k[l].expand((self.d, 1, -1)),
            #     stride=stride, 
            #     padding=0, 
            #     dilation=1, 
            #     ).transpose(1,2)
            #     # groups=self.d).transpose(1,2)
            # kls.append(kl.contiguous())

            # vl = torch.nn.functional.conv1d(
            #     v.transpose(1,2),
            #     self.conv_v[l],
            #     # self.conv_v[l].expand((self.d, 1, -1)),
            #     stride=stride,
            #     padding=0,
            #     dilation=1,
            #     groups=self.d).transpose(1,2)
            # vls.append(vl.contiguous())
        return kls, vls

    def stage2(self, cs, vls):
        '''
            calculate c@v
        '''
        m = self.m
        for l in range(self.levels + 1):
            if l==0:
                result = diagonaled_mm_cv_fine_ltr(cs[l], vls[l], m)   
            else:
                result_l = diagonaled_mm_cv_coarse_ltr(cs[l], vls[l], self.p)
                # scaling already taken into account in attn matrix, don't scale again here
                # if scale_attn:
                #     result_l *= m*2**(l-1) # multiply by the grid size
                result += result_l
        return result

    def forward(self, q, k, v):
        q = q / math.sqrt(self.d)
        kls, vls = self.convolute(k, v)
        cs, attns = self.stage1( q, kls)
        result = self.stage2(cs, vls)
        # return result, cs
        # return result, attns
        return result, None
