import time
import torch
from einops import rearrange
from tqdm import tqdm
from evaluator import evaluate
from collate_fn import train_batch, eval_batch
from utils import reset_random_seed, LadderSampler, fix_length
from torch.utils.data import DataLoader


def train(model, data, sampler, bsz, n_epoch, n_neg, eval_data, eval_sampler, eval_bsz, eval_n_neg, processor, loc2gpscode,
          max_len, optimizer, loss_fn, device, n_worker, log_path, result_path):
    reset_random_seed(1)
    for epoch in range(n_epoch):
        start_time = time.time()
        running_loss = 0.
        processed_batch = 0.
        data_loader = DataLoader(data,
                                 sampler=LadderSampler(data, bsz),
                                 num_workers=n_worker,
                                 batch_size=bsz,
                                 collate_fn=lambda e:
                                 train_batch(e,
                                             data,
                                             sampler,
                                             processor,
                                             loc2gpscode,
                                             max_len,
                                             n_neg))
        print("=====epoch {:>2d}=====".format(epoch + 1))
        batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)
        model.train()
        for batch_idx, (src_loc, src_gps, trg_loc, trg_gps, data_size) in batch_iterator:
            src_loc = src_loc.to(device)
            src_gps = src_gps.to(device)
            trg_loc = trg_loc.to(device)
            trg_gps = trg_gps.to(device)
            optimizer.zero_grad()
            output = model(src_loc, src_gps, trg_loc, trg_gps, data_size)
            output = rearrange(rearrange(output, 'b (k n) -> b k n', k=1+n_neg), 'b k n -> b n k')
            pos_scores, neg_scores = output.split([1, n_neg], -1)
            loss = loss_fn(pos_scores, neg_scores)
            keep = [torch.ones(e, dtype=torch.float32).to(device) for e in data_size]
            keep = fix_length(keep, 1, max_len, 'data size')
            loss = torch.sum(loss * keep) / torch.sum(torch.tensor(data_size).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            processed_batch += 1
            batch_iterator.set_postfix_str(f"loss={loss.item():.4f}")

        epoch_time = time.time() - start_time
        avg_loss = running_loss / processed_batch
        f = open(log_path, 'a+')
        print("time taken: {:.2f} sec".format(epoch_time))
        print("avg. loss: {:.4f}".format(running_loss / processed_batch))
        print("epoch={:d}, loss={:.4f}".format(epoch+1, avg_loss), file=f)
        f.close()
        if (epoch+1) % 1 == 0:
            print("=====evaluation for {:.4f}=====".format(epoch+1))
            hr, ndcg = evaluate(model, eval_data, eval_sampler, eval_bsz, eval_n_neg, processor, loc2gpscode, max_len, device, n_worker)
            print("Hit@1: {:.4f}, Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f} "
                  .format(hr[0], hr[4], ndcg[4], hr[9], ndcg[9]))

    print("training completed!")
    print("")
    print("=====evaluation under sampled metric (100 nearest un-visited locations)=====")
    hr, ndcg = evaluate(model, eval_data, eval_sampler, eval_bsz, eval_n_neg, processor, loc2gpscode, max_len, device, n_worker)
    print("Hit@1: {:.4f}, Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f} "
          .format(hr[0], hr[4], ndcg[4], hr[9], ndcg[9]))

    f = open(result_path, 'a+')
    print("=====evaluation under sampled metric (100 nearest un-visited locations)=====", file=f)
    print("Hit@1: {:.4f}, Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f}"
          .format(hr[0], hr[4], ndcg[4], hr[9], ndcg[9]), file=f)
    print('\n', file=f)
    f.close()
