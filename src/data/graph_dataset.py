import glob
import os
import warnings

import numpy as np
import torch
import torch.utils.data as data

from argparse import ArgumentParser
from tqdm import tqdm

from src.models.core import parse_t_f


# Dataset class for GINR and spatial graph INR models
class GraphDataset(data.Dataset):
    def __init__(self, dataset_dir, emb_dir, emb_name, model,
                 n_fourier = 5, n_nodes_in_sample = 1000,time = False,
                 cache_fourier = True, in_memory = True, cut = -1, **kwargs):
        self.dataset_dir = dataset_dir
        self.emb_dir = emb_dir
        self.emb_name = emb_name
        self.model = model
        self.n_fourier = n_fourier
        self.n_nodes_in_sample = n_nodes_in_sample
        self.time = time
        self.cache_fourier = cache_fourier
        self._fourier = None
        self._fourier_path = os.path.join(dataset_dir, 'fourier.npy')
        self.in_memory = in_memory
        self.cut = cut

        self.ginr_filenames = self.get_filenames(dataset_dir)
        if cut > 0:
            self.ginr_filenames = self.ginr_filenames[:cut]
        self.ginr_npzs = [np.load(f) for f in self.ginr_filenames]
        self.embs = [np.load(f'{self.emb_dir}/{self.emb_name}_emb.npz')] if self.model == 'Spatial_Graph_INR' else None
        self._data = None
        if in_memory:
            print('Loading dataset')
            self._data = [self.load_data(i) for i in tqdm(range(len(self)))]

    def load_data(self, index):
        data = {}
        data['inputs'] = self.get_inputs(index)
        data['target'] = self.get_target(index)
        return data
    
    def get_coordinates(self, index):
        arr = self.ginr_npzs[index]['points']
        return torch.from_numpy(arr).float()       
    
    def get_fourier(self, index):
        if self.cache_fourier and os.path.exists(self._fourier_path):
            if self._fourier is None:
                self._fourier = np.load(self._fourier_path)
                self._fourier = torch.from_numpy(self._fourier).float()
                self._fourier = self._fourier[:, : self.n_fourier]
            return self._fourier
        else:
            arr = self.ginr_npzs[index]['fourier'][:, : self.n_fourier]
            return torch.from_numpy(arr).float()
        
    def get_embbeding(self, index):
        emb_list = []
        
        if self.embs[index]['hyp'] is not None:
            hyp_emb = torch.Tensor(self.embs[index]['hyp'])
            emb_list.append(hyp_emb)
        if self.embs[index]['sph'] is not None:
            sph_emb = torch.Tensor(self.embs[index]['sph'])
            emb_list.append(sph_emb)
        if self.embs[index]['euc'] is not None:
            euc_emb = torch.Tensor(self.embs[index]['euc'])
            emb_list.append(euc_emb)
            
        emb_data = torch.cat(emb_list, 1).float()
        return emb_data

    def get_time(self, index):
        arr = self.ginr_npzs[index]['time']
        return torch.from_numpy(arr).float()

    @staticmethod
    def add_time(points, time):
        n_points = points.shape[-2]
        time = time.unsqueeze(0).repeat(n_points, 1)
        return torch.cat([points, time], dim = -1)
    
    def add_emb_npz(self, emb_data):
        self.emb_datas.append(emb_data)

    def get_inputs(self, index):
        if self.model == 'INR':
            arr = self.get_coordinates(index)
        elif self.model == 'GINR':
            arr = self.get_fourier(index)
        elif self.model == 'Spatial_Graph_INR':
            arr = self.get_embbeding(index)
        if self.time:
            time = self.get_time(index)
            arr = self.add_time(arr, time)
        return arr

    def get_target(self, index):
        arr = self.ginr_npzs[index]['target']
        return torch.from_numpy(arr).float()

    def get_data(self, index):
        if self.in_memory:
            return self._data[index]
        else:
            return self.load_data(index)

    def __getitem__(self, index):
        data = self.get_data(index)
        data_out = dict()

        n_points = data['inputs'].shape[0]
        
        if self.n_nodes_in_sample == -1:
            points_idx = self.get_subsampling_idx(n_points, 0, n_points)
        else:
            points_idx = self.get_subsampling_idx(n_points, 0, self.n_nodes_in_sample)
            
        data_out['inputs'] = data['inputs'][points_idx]
        data_out['target'] = data['target'][points_idx]
        data_out['index'] = index

        return data_out

    def __len__(self):
        return len(self.ginr_filenames)

    @property
    def target_dim(self):
        return self.get_data(0)['target'].shape[-1]

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents = [parent_parser], add_help = False)

        parser.add_argument('--dataset_dir', default = 'data/', type = str, help = 'Directory for .npz files to load')
        parser.add_argument('--emb_dir', default = '', type = str, help = 'Directory for embedding information files to load')
        parser.add_argument('--emb_name', default = '', type = str, help = 'Communal name of embedding information files to load')
        parser.add_argument('--model', default = 'GINR', type = str, help = 'Type of model to use - GINR or Spatial_Graph_INR')
        parser.add_argument('--n_fourier', default = 5, type = int, help = 'Number of fourier basis to use - GINR only')
        parser.add_argument('--n_nodes_in_sample', default = 1000, type = int, help = 'Number of nodes to load for each training step')
        parser.add_argument('--time', type = parse_t_f, default = False, help = 'Whether the data is time-varying or not')
        parser.add_argument('--in_memory', type = parse_t_f, default = True, help = 'Whether the data is loaded in memory or not')
        parser.add_argument('--cut', default = -1, type = int, help = 'Maximum number of .npz files to use')

        return parser

    @staticmethod
    def get_filenames(dataset_dir, subset = None):
        if subset is None:
            subset = ['*']

        if isinstance(subset, str):
            subset = open(subset).read().splitlines()
        elif isinstance(subset, list):
            pass
        else:
            raise TypeError(
                f'Unsupported type {type(subset)} for subset. '
                f'Expected string or list.'
            )

        npz_dir = os.path.join(dataset_dir, 'npz_files')
        npz_filenames = []
        for f in subset:
            npz_filenames += glob.glob(os.path.join(npz_dir, f'{f}.npz'))

        npz_filenames = sorted(npz_filenames, key = lambda s: s.split('/')[-1])

        return npz_filenames

    @staticmethod
    def get_subsampling_idx(n_points, start_index, to_keep):
        if n_points >= to_keep:
            idx = torch.randperm(n_points)[:to_keep]
        else:
            # Sample some indices more than once
            idx = (torch.randperm(n_points * int(np.ceil(to_keep / n_points)))[:to_keep] % n_points)
        idx = torch.add(idx, start_index)

        return idx


# Graph dataset class specialized for training and testing
# Assuming no memory issues
class SubGraphDataset(data.Dataset):
    def __init__(self, data, len, n_nodes_in_sample, ginr_npzs, **kwargs):
        self._data = data
        self.len = len
        self.n_nodes_in_sample = n_nodes_in_sample
        self.ginr_npzs = ginr_npzs

    def get_target(self, index):
        arr = self.ginr_npzs[index]['target']
        return torch.from_numpy(arr).float()

    def get_data(self, index):
        return self._data[index]

    def __getitem__(self, index):
        data = self.get_data(index)
        data_out = dict()

        n_points = data['inputs'].shape[0]
        
        if self.n_nodes_in_sample == -1:
            points_idx = self.get_subsampling_idx(n_points, 0, n_points)
        else:
            points_idx = self.get_subsampling_idx(n_points, 0, self.n_nodes_in_sample)
            
        data_out['inputs'] = data['inputs'][points_idx]
        data_out['target'] = data['target'][points_idx]
        data_out['index'] = index

        return data_out

    def __len__(self):
        return self.len

    @staticmethod
    def get_subsampling_idx(n_points, start_index, to_keep):
        if n_points >= to_keep:
            idx = torch.randperm(n_points)[:to_keep]
        else:
            # Sample some indices more than once
            idx = (torch.randperm(n_points * int(np.ceil(to_keep / n_points)))[:to_keep] % n_points)
        idx = torch.add(idx, start_index)

        return idx


def split_graphdataset(graphdataset, split_ratio):
    if(len(split_ratio) != 3):
        raise ValueError('The split ratio should have exactly 3 components - train, validation, and test ratios')
    
    if sum(split_ratio) != 1:
        raise ValueError('The sum of split ratio values should be 1')
    
    subset_lengths = []
    n_points = graphdataset.get_data(0)['inputs'].shape[0]
    
    for i, frac in enumerate(split_ratio):
        if frac < 0 or frac > 1:
            raise ValueError(f'Fraction at index {i} is not between 0 and 1')
        if i < (len(split_ratio) - 1):
            n_items_in_split = int(n_points * frac)
            subset_lengths.append(n_items_in_split)
        else:
            remainder = n_points - sum(subset_lengths)
            subset_lengths.append(remainder)
            
    for i, length in enumerate(subset_lengths):
        if length == 0:
            if i == 0:
                warnings.warn(f'Length of split at index {i} is 0. This will result in an empty training dataset.')
            elif i == 1:
                warnings.warn(f'Length of split at index {i} is 0. This will result in an empty validation dataset.')
            else:
                warnings.warn(f'Length of split at index {i} is 0. This will result in an empty test dataset.')
            
    indices = torch.randperm(n_points).tolist()
    train_indices = indices[:subset_lengths[0]]
    validation_indices = indices[subset_lengths[0]:(subset_lengths[0] + subset_lengths[1])]
    test_indices = indices[(subset_lengths[0] + subset_lengths[1]):]
    
    dataset_len = len(graphdataset)
    
    train_data = []
    validation_data = []
    test_data = []
    
    for i in range(dataset_len):
        data = graphdataset.get_data(i)
        
        train_dict = {}
        validation_dict = {}
        test_dict = {}
        
        for key in data.keys():
            train_dict[key] = data[key][train_indices]
            validation_dict[key] = data[key][validation_indices]
            test_dict[key] = data[key][test_indices]
        
        train_data.append(train_dict)
        validation_data.append(validation_dict)
        test_data.append(test_dict)
        
    train_dataset = SubGraphDataset(train_data, dataset_len, graphdataset.n_nodes_in_sample, graphdataset.ginr_npzs)
    validation_dataset = SubGraphDataset(validation_data, dataset_len, subset_lengths[1], graphdataset.ginr_npzs)
    test_dataset = SubGraphDataset(test_data, dataset_len, subset_lengths[2], graphdataset.ginr_npzs)
        
    return train_dataset, validation_dataset, test_dataset
