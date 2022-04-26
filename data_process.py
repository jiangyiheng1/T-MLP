import copy
import math
import os.path
import re
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from gps_encode import encode
from utils import serialize, unserialize
from torchtext.data import Field


class LBSNData(Dataset):
    def __init__(self, path, min_loc, min_user, map_level):
        self.loc2idx = {'<pad>': 0}
        self.loc2gps = {'<pad>': (0.0, 0.0)}
        self.idx2loc = {0: '<pad>'}
        self.idx2gps = {0: (0.0, 0.0)}
        self.loc2count = {}
        self.n_loc = 1
        self.loc_vocab(path, min_loc)
        self.user_seq, self.user2idx, self.n_user, self.gpscode2idx, self.n_gpscode, self.gpscode_idx2loc_idx = self.pre_process(path, min_user, map_level)

    def loc_vocab(self, path, min_loc):
        for line in open(path, encoding='utf-8'):
            # user, loc, time, lon, lat
            line = line.strip().split('\t')
            if line[0] == "user_id:token":
                continue
            loc = line[1]
            coordinate = float(line[3]), float(line[4])
            self.add_loc(loc, coordinate)
        self.n_loc = 1
        self.loc2idx = {'<pad>': 0}
        self.idx2loc = {0: '<pad>'}
        self.idx2gps = {0: (0.0, 0.0)}
        for loc in self.loc2count:
            if self.loc2count[loc] >= min_loc:
                self.add_loc(loc, self.loc2gps[loc])
        self.freq = np.zeros(self.n_loc - 1, dtype=np.int32)
        for idx, loc in self.idx2loc.items():
            if idx != 0:
                self.freq[idx - 1] = self.loc2count[loc]

    def add_loc(self, loc, coordinate):
        if loc not in self.loc2idx:
            self.loc2idx[loc] = self.n_loc
            self.loc2gps[loc] = coordinate
            self.idx2loc[self.n_loc] = loc
            self.idx2gps[self.n_loc] = coordinate
            if loc not in self.loc2count:
                self.loc2count[loc] = 1
            self.n_loc += 1
        else:
            self.loc2count[loc] += 1

    def pre_process(self, path, min_user, map_level):
        user_seq = {}
        user_seq_array = []
        user2idx = {}
        n_user = 1
        gpscode2idx = {}
        idx2gpscode = {}
        g_idx2l_idx = defaultdict(set)
        n_gpscode = 1

        for line in open(path, encoding='utf-8'):
            line = line.strip().split('\t')
            # user, loc, time, lon, lat
            if line[0] == "user_id:token":
                continue
            user = line[0]
            loc = line[1]
            timestamp = float(line[2])
            lon = float(line[3])
            lat = float(line[4])
            if loc not in self.loc2idx:
                continue
            loc_idx = self.loc2idx[loc]
            gpscode = encode(lat, lon, map_level)
            if gpscode not in gpscode2idx:
                gpscode2idx[gpscode] = n_gpscode
                idx2gpscode[n_gpscode] = gpscode
                n_gpscode += 1
            gpscode_idx = gpscode2idx[gpscode]
            g_idx2l_idx[gpscode_idx].add(loc_idx)

            if user not in user_seq:
                user_seq[user] = []
            user_seq[user].append([loc_idx, gpscode, timestamp])

        for user, seq in user_seq.items():
            if len(seq) >= min_user:
                user2idx[user] = n_user
                user_idx = n_user
                seq_new = []
                tmp = set()
                cnt = 0
                for loc_idx, gpscode, timestamp in sorted(seq, key=lambda e:e[-1]):
                    if loc_idx in tmp:
                        seq_new.append((user_idx, loc_idx, gpscode, timestamp, True))
                    else:
                        seq_new.append((user_idx, loc_idx, gpscode, timestamp, False))
                        tmp.add(loc_idx)
                        cnt += 1
                if cnt >= min_user / 2:
                    n_user += 1
                    user_seq_array.append(seq_new)

        all_gpscode = []
        for u in range(len(user_seq_array)):
            seq = user_seq_array[u]
            for i in range(len(seq)):
                check_in = seq[i]
                user_idx = check_in[0]
                loc_idx = check_in[1]
                gpscode = check_in[2]
                timestamp = check_in[3]
                bool = check_in[4]
                gpscode = re.findall(".{2}", gpscode)
                bi_gpscode = '\t'.join(gpscode)
                bi_gpscode = bi_gpscode.split('\t')
                all_gpscode.append(bi_gpscode)
                user_seq_array[u][i] = (user_idx, loc_idx, bi_gpscode, timestamp, bool)

        self.loc2gpscode = ['NULL']
        for loc_idx in range(1, self.n_loc):
            lon, lat = self.idx2gps[loc_idx]
            gpscode = encode(lat, lon, map_level)
            gpscode = re.findall(".{2}", gpscode)
            bi_gpscode = '\t'.join(gpscode)
            bi_gpscode = bi_gpscode.split('\t')
            self.loc2gpscode.append(bi_gpscode)
            all_gpscode.append(bi_gpscode)

        self.GPSCODE = Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            unk_token=None,
            preprocessing=str.split
        )
        self.GPSCODE.build_vocab(all_gpscode)
        return user_seq_array, user2idx, n_user, gpscode2idx, n_gpscode, g_idx2l_idx

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx]

    def data_partition(self, max_len):
        train_data = copy.copy(self)
        eval_data = copy.copy(self)
        train_seq = []
        eval_seq = []
        for user in range(len(self)):
            seq = self[user]
            for i in reversed(range(len(seq))):
                if not seq[i][-1]:
                    break
            eval_trg_loc = seq[i: i+1]
            eval_src_seq = seq[max(0, i-max_len): i]
            eval_seq.append((eval_src_seq, eval_trg_loc))

            num_inst = math.floor((i+max_len-1) / max_len)
            for k in range(num_inst):
                if (i-k*max_len) > max_len * 1.1:
                    train_trg_seq = seq[i-(k+1)*max_len: i-k*max_len]
                    train_src_seq = seq[i-(k+1)*max_len-1: i-k*max_len-1]
                    train_seq.append((train_src_seq, train_trg_seq))
                else:
                    train_trg_seq = seq[max(0, i-(k+1)*max_len): i-k*max_len]
                    train_src_seq = seq[max(0, i-(k+1)*max_len-1): i-k*max_len-1]
                    train_seq.append((train_src_seq, train_trg_seq))
                    break

        train_data.user_seq = train_seq
        eval_data.user_seq = eval_seq
        return train_data, eval_data


if __name__ == "__main__":
    prefix = 'LBSNData'
    data_name = 'gowalla'
    cold_user = 20
    cold_loc = 10
    raw_data_path = prefix + '/' + data_name + '/' + data_name + '.inter'
    clean_data_path = prefix + '/' + data_name + '/' + data_name + '.data'
    if os.path.exists(clean_data_path):
        dataset = unserialize(clean_data_path)
    else:
        dataset = LBSNData(raw_data_path, cold_loc, cold_user, 10)
        serialize(dataset, clean_data_path)

    count = 0
    length = []
    for seq in dataset.user_seq:
        count += len(seq)
        length.append(len(seq))
    print("#check-ins:", count)
    print("#users:", dataset.n_user - 1)
    print("#locations:", dataset.n_loc - 1)
    print("#average seq len:", np.mean(np.array(length)))
    print("sparsity:", 1 - count / ((dataset.n_user - 1) * (dataset.n_loc - 1)))