import os
import torch
import numpy as np
from data_process import LBSNData
from loc_query_system import Location_Query_System
from negative_sampler import UniformSampler, KNNSampler
from loss_fn import BCELoss
from trainer import train
from utils import serialize, unserialize, get_visited_locs
from model import *


if __name__ == "__main__":
    "Path Definition"
    prefix = '/data/home/scv6856/run/'
    data_name = 'yelp'
    raw_data_path = prefix + 'LBSNData' + '/' + data_name + '/' + data_name + '.inter'
    clean_data_path = prefix + 'LBSNData' + '/' + data_name + '/' + data_name + '.data'
    loc_query_path = prefix + 'LBSNData' + '/' + data_name + '/' + data_name + '_tree.pkl'
    log_path = prefix + 'log' + '/' + data_name + '.txt'
    result_path = prefix + 'result' + '/' + data_name + '.txt'

    '''Data Pre-Process'''
    if os.path.exists(clean_data_path):
        dataset = unserialize(clean_data_path)
    else:
        cold_user = 20
        cold_loc = 10
        dataset = LBSNData(raw_data_path, cold_loc, cold_user, 10)
        serialize(dataset, clean_data_path)
    max_len = 128
    train_data, eval_data = dataset.data_partition(max_len)
    count = 0
    length = []
    for seq in dataset.user_seq:
        count += len(seq)
        length.append(len(seq))
    print("dataset:", data_name)
    print("#check-ins:", count)
    print("#users:", dataset.n_user - 1)
    print("#locations:", dataset.n_loc - 1)
    print("#average seq len:", np.mean(np.array(length)))
    print("sparsity:", 1 - count / ((dataset.n_user - 1) * (dataset.n_loc - 1)))

    '''Location Query System'''
    query_tree = Location_Query_System()
    if os.path.exists(loc_query_path):
        query_tree.load(loc_query_path)
    else:
        n_neighbour = 2000
        query_tree.build_tree(dataset)
        query_tree.prefetch(n_neighbour)
        query_tree.save(loc_query_path)

    '''Train & Eval Detail'''
    user_visited_locs = get_visited_locs(dataset)
    n_epoch = 200
    train_bsz = 1024
    train_neg = 1
    train_neg_sampler = KNNSampler(query_tree, 2000, user_visited_locs, 'training', False)
    eval_bsz = 2048
    eval_neg = 99
    eval_neg_sampler = KNNSampler(query_tree, 2000, user_visited_locs, 'evaluating', True)
    loss_fn = BCELoss()
    n_worker = 10
    device = 'cuda:0'

    "Constructing Model"
    model = C_MLP(  n_loc=dataset.n_loc,
                    n_gps=dataset.n_gpscode,
                    length=max_len,
                    loc_dim=64,
                    gps_dim=64,
                    exp_factor=4,
                    n_set=4,
                    n_head=2,
                    geo_depth=2,
                    seq_depth=2,
                    drop_ratio=0.5)
    model.to(device)
    processor = dataset.GPSCODE
    loc2gpscode = dataset.loc2gpscode
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    train(model, train_data, train_neg_sampler, train_bsz, n_epoch, train_neg,
          eval_data, eval_neg_sampler, eval_bsz, eval_neg, processor, loc2gpscode,
          max_len, optimizer, loss_fn, device, n_worker, log_path, result_path)