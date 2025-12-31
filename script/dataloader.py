import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import pickle
import torch_geometric
from torch_geometric.nn import radius_graph, TransformerConv
import torch.utils.data as data
import random
import torch.nn.functional as F
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from Bio.PDB import PDBParser, PPBuilder
from torch_geometric.transforms import AddLaplacianEigenvectorPE



class colabMotifDataset(Dataset):
    def __init__(self, dict_path, emb_dir, structure_dir,
                 train=False, window_size=100,
                 t2_label_id=2,
                 t2_aug_prob=0.3,
                 t2_feat_drop_p=0.05,
                 t2_noise_std=0.05):
        self.emb_dir = emb_dir
        self.window_size = window_size

        self.struct_dir = structure_dir
        
        with open(dict_path, 'rb') as f:
            self.data_dict = pickle.load(f)


        self.data_list = list(self.data_dict.keys())

        self.train = train
        self.t2_label_id = t2_label_id
        self.t2_aug_prob = t2_aug_prob
        self.t2_feat_drop_p = t2_feat_drop_p
        self.t2_noise_std = t2_noise_std
        
    def __len__(self):
        return len(self.data_list)

    def get_labels(self, indices=None):
        if indices is None:
            target_labels = [self.data_dict[name][-1] for name in self.data_list]
        else:
            target_labels = [self.data_dict[self.data_list[i]][-1] for i in indices]
        return np.array(target_labels)

    def _feat_dropout(self, x, p):
        if p <= 0: 
            return x
        mask = (np.random.rand(*x.shape) > p).astype(np.float32)
        return x * mask

    def _gaussian_noise(self, x, std):
        if std <= 0:
            return x
        return x + np.random.normal(0, std, size=x.shape).astype(np.float32)

    def __getitem__(self, idx):
        name = self.data_list[idx]
        label = self.data_dict[name][-1]
        path = os.path.join(self.emb_dir, f"{name}.npy")

        emb = np.load(path)  # [L, 1280]
        L, D = emb.shape

        global_feat = np.mean(emb, axis=0)

        n_seq = np.zeros((self.window_size, D), dtype=np.float32)
        if L < self.window_size:
            n_seq[:L, :] = emb
        else:
            n_seq[:, :] = emb[:self.window_size, :]

        c_seq = np.zeros((self.window_size, D), dtype=np.float32)
        if L < self.window_size:
            c_seq[:L, :] = emb
        else:
            c_seq[:, :] = emb[-self.window_size:, :]

        n_seq = n_seq.transpose(1, 0)  # [D, window]
        c_seq = c_seq.transpose(1, 0)



        subdir = os.path.join(self.struct_dir, name)
        single_emb_path = os.path.join(subdir, "single_repr.npy")
        
        single_emb = np.load(single_emb_path)

        L, D = single_emb.shape


        K = 50
        n_struct = np.zeros((K, D), dtype=np.float32)
        if L < K:
            n_struct[:L, :] = single_emb
        else:
            n_struct[:, :] = single_emb[:K, :]

        c_struct = np.zeros((K, D), dtype=np.float32)
        if L < K:
            c_struct[:L, :] = single_emb
        else:
            c_struct[:, :] = single_emb[-K:, :]

        nc_feats = np.concatenate([n_struct, c_struct], axis=0)

        nc_feats = nc_feats.max(axis=0)

        # 1. 全局特征 (Global)
        s_mean_feat = np.mean(single_emb, axis=0) # [256,]
        s_max_feat = np.max(single_emb, axis=0)
        # s_global_feat = np.concatenate([s_mean_feat, s_max_feat],axis=-1)


        # ss_global_feat = np.concatenate([s_global_feat, nc_feats],axis=-1)


        # ===== 只对 T2SE 做增强 =====
        if self.train and label != 0:
            if random.random() < self.t2_aug_prob:
                n_seq = self._gaussian_noise(n_seq, self.t2_noise_std)
                c_seq = self._gaussian_noise(c_seq, self.t2_noise_std)
                global_feat = self._gaussian_noise(global_feat, self.t2_noise_std)
                s_mean_feat = self._gaussian_noise(s_mean_feat, self.t2_noise_std)

        return (
            torch.tensor(global_feat, dtype=torch.float32),
            torch.tensor(n_seq, dtype=torch.float32),
            torch.tensor(c_seq, dtype=torch.float32),
            torch.tensor(s_mean_feat, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )



class CLIPfDataset(Dataset):
    def __init__(self, dict_path, emb_dir, structure_dir,
                 train=False, window_size=200):
        self.emb_dir = emb_dir
        self.window_size = window_size

        self.struct_dir = structure_dir
        
        with open(dict_path, 'rb') as f:
            self.data_dict = pickle.load(f)

        self.data_list = list(self.data_dict.keys())

       
    def __len__(self):
        return len(self.data_list)

    def get_labels(self, indices=None):
        if indices is None:
            target_labels = [self.data_dict[name][-1] for name in self.data_list]
        else:
            target_labels = [self.data_dict[self.data_list[i]][-1] for i in indices]
        return np.array(target_labels)

    def __getitem__(self, idx):
        name = self.data_list[idx]
        label = self.data_dict[name][-1]
        path = os.path.join(self.emb_dir, f"{name}.npy")

        emb = np.load(path)  # [L, 1280]
        L, D = emb.shape

        global_feat = np.mean(emb, axis=0)


        subdir = os.path.join(self.struct_dir, name)
        single_emb_path = os.path.join(subdir, "single_repr.npy")
        
        single_emb = np.load(single_emb_path)
        
        s_feat = single_emb.mean(axis=0)

        return (
            torch.tensor(global_feat, dtype=torch.float32),
            torch.tensor(s_feat, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


