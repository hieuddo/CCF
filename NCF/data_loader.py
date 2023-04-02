#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
# import pandas as pd
import torch
from torch.utils import data
import pickle5 as pickle


class DataFolder(data.Dataset):
    """Load Data for Iterator. """

    def __init__(self, data_path, neg_path, neg_cnt, buffer=None, df=None):
        """Initializes image paths and preprocessing module."""
        self.neg_list = torch.tensor(np.load(neg_path))
        self.neg_cnt = neg_cnt

        if df is not None:
            self.data = torch.tensor(df.to_numpy(dtype="int64"))
        else:
            # self.data = torch.tensor(pd.read_pickle(data_path).values)
            with open(data_path, 'rb') as f:
                df = pickle.load(f).to_numpy(dtype="int64")
            if buffer is not None:
                # breakpoint()
                with open(buffer, 'rb') as bf:
                    buffer_df = pickle.load(bf).to_numpy(dtype="int64")
                if len(buffer_df):
                    df = np.concatenate((buffer_df, df), axis=0)
            self.data = torch.tensor(df)

    def __getitem__(self, index):
        """Reads an Data and Neg Sample from a file and returns."""
        src = self.data[index]
        usr = int(src[0])-1
        neg = self.neg_list[usr]
        return src, neg

    def __len__(self):
        """Returns the total number of font files."""
        return self.data.size(0)


def get_loader(root, data_path, neg_path, neg_cnt, batch_size, shuffle=False, num_workers=0, buffer=None, df=None):
    """Builds and returns Dataloader."""
    if buffer:
        buffer = root + buffer
    dataset = DataFolder(root+data_path, root+neg_path, neg_cnt, buffer, df)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader


class MERDataFolder(data.Dataset):
    """Load Data for Iterator. """

    def __init__(self, data_path, neg_path, neg_cnt, buffer, seed=123):
        """Initializes image paths and preprocessing module."""
        self.neg_list = torch.tensor(np.load(neg_path))
        self.neg_cnt = neg_cnt

        # self.data = torch.tensor(pd.read_pickle(data_path).values)
        with open(data_path, 'rb') as f:
            self.data = torch.tensor(pickle.load(f).to_numpy(dtype="int64"))
        with open(buffer, 'rb') as bf:
            self.buffer = torch.tensor(pickle.load(bf).to_numpy(dtype="int64"))

    def __getitem__(self, index):
        """Reads an Data and Neg Sample from a file and returns."""
        src = self.data[index]
        usr = int(src[0])-1
        neg = self.neg_list[usr]
        return src, neg

    def __len__(self):
        """Returns the total number of font files."""
        return self.data.size(0)


def get_mer_loader(root, data_path, neg_path, neg_cnt, batch_size, shuffle=False, num_workers=0, buffer=None, df=None):
    """Builds and returns Dataloader."""
    if buffer:
        buffer = root + buffer
    dataset = DataFolder(root+data_path, root+neg_path, neg_cnt, buffer, df)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    train_loader = get_loader("data/ml-1m/", "train_0.pkl",
                              "neg_0.npy", 100, 1024, True, 0, buffer="buffer_0.pkl")
