import numpy as np
import os
from data_process import  LBSNData
from sklearn.neighbors import BallTree
from tqdm import tqdm
from utils import serialize, unserialize
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


class Location_Query_System:
    def __init__(self):
        self.coordinates = []
        self.tree = None
        self.n = None
        self.nbr_locs = None

    def build_tree(self, dataset):
        self.coordinates = np.zeros((len(dataset.idx2gps)-1, 2), dtype=np.float64)
        for idx, (lon, lat) in dataset.idx2gps.items():
            if idx != 0:
                self.coordinates[idx - 1] = [lon, lat]
        self.tree = BallTree(self.coordinates, leaf_size=1, metric='haversine')

    def prefetch(self, n_nbr):
        self.n = n_nbr
        self.nbr_locs = np.zeros((self.coordinates.shape[0], self.n), dtype=np.int32)
        for idx, gps in tqdm(enumerate(self.coordinates), total=len(self.coordinates), leave=True):
            trg_gps = gps.reshape(1, -1)
            _, nbr = self.tree.query(trg_gps, self.n+1)
            nbr = nbr[0, 1:]
            nbr += 1
            self.nbr_locs[idx] = nbr

    def retrieve(self, trg_loc_idx, k):
        knn_locs = self.nbr_locs[trg_loc_idx-1][:k]
        return knn_locs
    
    def save(self, path):
        data = {"coordinates": self.coordinates,
                "tree": self.tree,
                "neighbour": self.n,
                "neighbour_location": self.nbr_locs}
        serialize(data, path)
    
    def load(self, path):
        data = unserialize(path)
        self.coordinates = data["coordinates"]
        self.tree = data["tree"]
        self.n = data["neighbour"]
        self.nbr_locs = data["neighbour_location"]


if __name__ == "__main__":
    prefix = 'LBSNData'
    data_name = 'yelp'
    cold_user = 20
    cold_loc = 10
    n_neighbour = 2000
    raw_data_path = prefix + '/' + data_name + '/' + data_name + '.inter'
    clean_data_path = prefix + '/' + data_name + '/' + data_name + '.data'
    loc_query_path = prefix + '/' + data_name + '/' + data_name + '_tree.pkl'

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
    print("#POIs:", dataset.n_loc - 1)
    print("#average seq len:", np.mean(np.array(length)))
    print("sparsity:", 1 - count / ((dataset.n_user - 1) * (dataset.n_loc - 1)))

    query_tree = Location_Query_System()
    if os.path.exists(loc_query_path):
        query_tree.load(loc_query_path)
    else:
        query_tree.build_tree(dataset)
        query_tree.prefetch(n_neighbour)
        query_tree.save(loc_query_path)