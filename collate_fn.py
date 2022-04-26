import torch
from utils import fix_length
from einops import rearrange

def train_batch(batch, data_source, sampler, processor, loc2gpscode, max_len, n_neg):
    src_seq, trg_seq = zip(*batch)
    locs, gpscodes = [], []
    data_size = []
    for e in src_seq:
        _, l_, g_, _, _ = zip(*e)
        data_size.append(len(l_))
        locs.append(torch.tensor(l_))
        g_ = processor.numericalize(list(g_))
        gpscodes.append(g_)
    # (b, n)
    src_locs = fix_length(locs, 1, max_len, 'seq')
    # (b, n, l)
    src_gpscodes = fix_length(gpscodes, 2, max_len, 'seq')

    locs, gpscodes = [], []
    for i, seq in enumerate(trg_seq):
        pos = torch.tensor([[e[1]] for e in seq])
        neg = sampler(seq, n_neg, user=seq[0][0])
        cat_locs = torch.cat([pos, neg], dim=-1)
        locs.append(cat_locs)
        cat_gpscodes = []
        for loc in range(cat_locs.size(0)):
            g_ = []
            for idx in cat_locs[loc]:
                g_.append(loc2gpscode[idx])
            cat_gpscodes.append(processor.numericalize(list(g_)))
        gpscodes.append(torch.stack(cat_gpscodes))
    # (b, n, 1+k)
    trg_locs = fix_length(locs, 2, max_len, 'seq')
    # (b, n, 1+k, l)
    trg_gpscodes = fix_length(gpscodes, 3, max_len, 'seq')
    # (b, n*(1+k))
    trg_locs = rearrange(rearrange(trg_locs, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')
    # (b, n*(1+k), l)
    trg_gpscodes = rearrange(rearrange(trg_gpscodes, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    return src_locs, src_gpscodes, trg_locs, trg_gpscodes, data_size

def eval_batch(batch, data_source, sampler, processor, loc2gpscode, max_len, n_neg):
    src_seq, trg_seq = zip(*batch)
    locs, gpscodes, = [], []
    data_size = []
    for e in src_seq:
        _, l_, g_, _, _ = zip(*e)
        data_size.append(len(l_))
        locs.append(torch.tensor(l_))
        g_ = processor.numericalize(list(g_))
        gpscodes.append(g_)
    # (b, n)
    src_locs = fix_length(locs, 1, max_len, 'seq')
    # (b, n, l)
    src_gpscodes = fix_length(gpscodes, 2, max_len, 'seq')

    locs, gpscodes = [], []
    for i, seq in enumerate(trg_seq):
        pos = torch.tensor([[e[1]] for e in seq])
        neg = sampler(seq, n_neg, user=seq[0][0])
        cat_locs = torch.cat([pos, neg], dim=-1)
        locs.append(cat_locs)
        cat_gpscodes = []
        for loc in range(cat_locs.size(0)):
            g_ = []
            for idx in cat_locs[loc]:
                g_.append(loc2gpscode[idx])
            cat_gpscodes.append(processor.numericalize(list(g_)))
        gpscodes.append(torch.stack(cat_gpscodes))
    # (b, n, 1+k)
    trg_locs = fix_length(locs, 2, max_len, 'loc')
    # (b, n, 1+k, l)
    trg_gpscodes = fix_length(gpscodes, 3, max_len, 'loc')
    # (b, n*(1+k))
    trg_locs = rearrange(rearrange(trg_locs, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')
    # (b, n*(1+k), l)
    trg_gpscodes = rearrange(rearrange(trg_gpscodes, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    return src_locs, src_gpscodes, trg_locs, trg_gpscodes, data_size