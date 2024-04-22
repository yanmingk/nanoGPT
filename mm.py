import torch


@torch.jit.script
def qk_fine(q, k, c, num_blocks, m):
    '''
        q: (b, num_blocks, m, d)
        k: (b, num_blocks, m, d)
        c: (b, num_blocks, m, 3*m)
    '''
    for i in range(num_blocks):
        if i > 0:
            c[:, i, :, :m] = torch.bmm(q[:, i], k[:, i - 1].transpose(1, 2))
        c[:, i, :, m:2*m] = torch.bmm(q[:, i], k[:, i].transpose(1, 2))
        if i < num_blocks - 1:
            c[:, i, :, 2*m:] = torch.bmm(q[:, i], k[:, i + 1].transpose(1, 2))

@torch.jit.script
def qk_fine_auto(q, k, c, num_blocks, m):
    '''
        q: (b, num_blocks, m, d)
        k: (b, num_blocks, m, d)
        c: (b, num_blocks, m, 2*m)
    '''
    for i in range(num_blocks):
        if i > 0:
            c[:, i, :, :m] = torch.bmm(q[:, i], k[:, i - 1].transpose(1, 2))
        c[:, i, :, m:2*m] = torch.tril(torch.bmm(q[:, i], k[:, i].transpose(1, 2)))



@torch.jit.script
def qk_coarse(q, k, c, num_blocks, p):
    '''
        q: (b, num_blocks, ml, d)
        k: (b, num_blocks, p, d)
        c: (b, num_blocks, ml, 4*p)
    '''
    for i in range(num_blocks):
        if i > 1:
            if i&1==0:
                c[:, i, :, :p] = torch.bmm(q[:, i], k[:, i-2].transpose(1, 2))
            else:
                c[:, i, :, :p] = torch.bmm(q[:, i], k[:, i-3].transpose(1, 2))
                c[:, i, :, p:2*p] = torch.bmm(q[:, i], k[:, i-2].transpose(1, 2))
        if i < num_blocks - 2:
            if i&1==0:
                c[:, i, :, 2*p:3*p] = torch.bmm(q[:, i], k[:, i+2].transpose(1, 2))
                c[:, i, :, 3*p:] = torch.bmm(q[:, i], k[:, i+3].transpose(1, 2))
            else:
                c[:, i, :, 3*p:] = torch.bmm(q[:, i], k[:, i+2].transpose(1, 2))


@torch.jit.script
def qk_coarse_auto(q, k, c, num_blocks, p):
    '''
        q: (b, num_blocks, ml, d)
        k: (b, num_blocks, p, d)
        c: (b, num_blocks, ml, 2*p)
    '''
    for i in range(num_blocks):
        if i > 1:
            if i&1==0:
                c[:, i, :, :p] = torch.bmm(q[:, i], k[:, i-2].transpose(1, 2))
            else:
                c[:, i, :, :p] = torch.bmm(q[:, i], k[:, i-3].transpose(1, 2))
                c[:, i, :, p:2*p] = torch.bmm(q[:, i], k[:, i-2].transpose(1, 2))


@torch.jit.script
def cv_fine(c, v, o, num_blocks, m):
    '''
        c: (b, num_blocks, m, 2*m)
        v: (b, num_blocks, m, d)
        o: (b, n, d)
    '''
    for i in range(num_blocks):
        o[:, i*m:(i+1)*m, :] = torch.bmm(c[:, i, :, m:2*m], v[:, i])

        if i > 0:
            o[:, i*m:(i+1)*m, :] += torch.bmm(c[:, i, :, :m], v[:, i-1])

        if i < num_blocks - 1:
            o[:, i*m:(i+1)*m, :] += torch.bmm(c[:, i, :, 2*m:], v[:, i+1])


@torch.jit.script
def cv_fine_auto(c, v, o, num_blocks, m):
    '''
        c: (b, num_blocks, m, 2*m)
        v: (b, num_blocks, m, d)
        o: (b, n, d)
    '''
    for i in range(num_blocks):
        o[:, i*m:(i+1)*m, :] = torch.bmm(c[:, i, :, m:2*m], v[:, i])

        if i > 0:
            o[:, i*m:(i+1)*m, :] += torch.bmm(c[:, i, :, :m], v[:, i-1])



@torch.jit.script
def cv_coarse(c, v, o, num_blocks: int, ml: int, p: int):
    '''
        c: (b, num_blocks, ml, 4*p)
        v: (b, num_blocks, p, d)
        o: (b, n, d)
    '''
    for i in range(num_blocks):
        if i > 1:
            if i&1==0:
                o[:, i*ml:(i+1)*ml, :] = torch.bmm(c[:, i, :, :p], v[:, i-2])
            else:
                o[:, i*ml:(i+1)*ml, :] = torch.bmm(c[:, i, :, :p], v[:, i-3])
                o[:, i*ml:(i+1)*ml, :] += torch.bmm(c[:, i, :, p:2*p], v[:, i-2])
        if i < num_blocks - 2:
            if i&1==0:
                o[:, i*ml:(i+1)*ml, :] += torch.bmm(c[:, i, :, 2*p:3*p], v[:, i+2])
                o[:, i*ml:(i+1)*ml, :] += torch.bmm(c[:, i, :, 3*p:], v[:, i+3])
            else:
                o[:, i*ml:(i+1)*ml, :] += torch.bmm(c[:, i, :, 3*p:], v[:, i+2])