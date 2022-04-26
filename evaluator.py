import numpy as np
import torch
from collate_fn import eval_batch
from utils import reset_random_seed
from torch.utils.data import DataLoader
from collections import Counter


def evaluate(model, data, sampler, bsz, n_neg, processor, loc2gpscode, max_len, device, n_worker):
    model.eval()
    reset_random_seed(1)
    loader = DataLoader(data,
                        batch_size=bsz,
                        num_workers=n_worker,
                        collate_fn=lambda e: eval_batch(e, data, sampler, processor, loc2gpscode, max_len, n_neg))
    cnt = Counter()
    array = np.zeros(1 + n_neg)
    with torch.no_grad():
        for _, (src_loc, src_gps, trg_loc, trg_gps, data_size) in enumerate(loader):
            src_loc = src_loc.to(device)
            src_gps = src_gps.to(device)
            trg_loc = trg_loc.to(device)
            trg_gps = trg_gps.to(device)
            output = model(src_loc, src_gps, trg_loc, trg_gps, data_size)
            idx = output.sort(descending=True, dim=1)[1]
            order = idx.topk(k=1, dim=1, largest=False)[1]
            cnt.update(order.squeeze().tolist())
    for k, v in cnt.items():
        array[k] = v
    Hit_Rate = array.cumsum()
    NDCG = 1 / np.log2(np.arange(0, n_neg + 1) + 2)
    NDCG = NDCG * array
    NDCG = NDCG.cumsum() / Hit_Rate.max()
    Hit_Rate = Hit_Rate / Hit_Rate.max()
    return Hit_Rate, NDCG