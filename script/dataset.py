"""
dataset.py — 蛋白质数据集加载
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle


class ProteinDataset(Dataset):
    def __init__(self, dict_path, emb_dir, window_size=100, **kwargs):
        self.emb_dir = emb_dir
        self.window_size = window_size

        with open(dict_path, 'rb') as f:
            self.data_dict = pickle.load(f)
        self.data_list = list(self.data_dict.keys())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        name = self.data_list[idx]
        label = self.data_dict[name][-1]
        emb = np.load(os.path.join(self.emb_dir, f"{name}.npy"))
        L, D = emb.shape
        W = self.window_size

        global_feat = np.mean(emb, axis=0)

        n_seq = np.zeros((W, D), dtype=np.float32)
        if L < W:
            n_seq[:L, :] = emb
        else:
            n_seq[:, :] = emb[:W, :]

        c_seq = np.zeros((W, D), dtype=np.float32)
        if L < W:
            c_seq[:L, :] = emb
        else:
            c_seq[:, :] = emb[-W:, :]

        n_seq = n_seq.transpose(1, 0)
        c_seq = c_seq.transpose(1, 0)

        return (
            torch.tensor(global_feat, dtype=torch.float32),
            torch.tensor(n_seq, dtype=torch.float32),
            torch.tensor(c_seq, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )