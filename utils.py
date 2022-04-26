import os
import json
import torch
import torch
import numpy as np
import random
import math
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence
from einops import reduce, repeat

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def serialize(obj, path, in_json=False):
    if in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)


def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)


def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LadderSampler(Sampler):
    def __init__(self, data_source, batch_sz, fix_order=False):
        super(LadderSampler, self).__init__(data_source)
        self.data = [len(e[0]) for e in data_source]
        self.batch_size = batch_sz * 100
        self.fix_order = fix_order

    def __iter__(self):
        if self.fix_order:
            d = zip(self.data, np.arange(len(self.data)), np.arange(len(self.data)))
        else:
            d = zip(self.data, np.random.permutation(len(self.data)), np.arange(len(self.data)))
        d = sorted(d, key=lambda e: (e[1] // self.batch_size, e[0]), reverse=True)
        return iter(e[2] for e in d)

    def __len__(self):
        return len(self.data)


def fix_length(sequences, n_axies, max_len, dtype):
    if dtype != 'loc':
        padding_term = torch.zeros_like(sequences[0])
        length = padding_term.size(0)
        # (l, any) -> (1, any) -> (max_len any)
        if n_axies == 1:
            padding_term = reduce(padding_term, '(h l) -> h', 'max', l=length)
            padding_term = repeat(padding_term, 'h -> (repeat h)', repeat=max_len)
        elif n_axies == 2:
            padding_term = reduce(padding_term, '(h l) any -> h any', 'max', l=length)
            padding_term = repeat(padding_term, 'h any -> (repeat h) any', repeat=max_len)
        else:
            padding_term = reduce(padding_term, '(h l) any_1 any_2 -> h any_1 any_2', 'max', l=length)
            padding_term = repeat(padding_term, 'h any_1 any_2 -> (repeat h) any_1 any_2', repeat=max_len)

        sequences.append(padding_term)
        tensor = pad_sequence(sequences, True)
        return tensor[:-1]
    else:
        tensor = pad_sequence(sequences, True)
        return tensor


def get_visited_locs(dataset):
    user_visited_locs = {}
    for u in range(len(dataset.user_seq)):
        seq = dataset.user_seq[u]
        user = seq[0][0]
        user_visited_locs[user] = set()
        for i in reversed(range(len(seq))):
            if not seq[i][-1]:
                break
        user_visited_locs[user].add(seq[i][1])
        seq = seq[:i]
        for check_in in seq:
            user_visited_locs[user].add(check_in[1])
    return user_visited_locs