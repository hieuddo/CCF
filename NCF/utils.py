import numpy as np
from sklearn.cluster import KMeans
from time import time
from copy import deepcopy
import torch
import argparse
import random
import os
import pickle
import pandas as pd


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_pkl_df(path, root="data/ml-1m/"):
    with open(root+path, "rb") as f:
        return pickle.load(f)


def read_buffer_df(path, train_size, batch_size, root="data/ml-1m/"):
    df = read_pkl_df(path, root)
    buffer = deepcopy(df)
    need = train_size * (batch_size-1)
    duplicate = (need + len(df)-1) // len(df)
    for _ in range(duplicate):
        buffer = pd.concat([buffer, df], axis=0).reset_index(drop=True)
    return buffer


def split_df_by_user(train_df, val_df, previous_model, n_clusters=10, seed=123, max_iter=5000, prev_centroids="random"):
    user_emb = np.array(previous_model.u_emb.weight.data.cpu())
    kmeans = KMeans(n_clusters=n_clusters, init=prev_centroids,
                    random_state=seed, max_iter=max_iter).fit(user_emb)
    prev_centroids = kmeans.cluster_centers_
    labels = kmeans.predict(user_emb)
    for lb in range(n_clusters):
        user_set = np.where(labels == lb)[0]
        yield train_df[train_df.user_id.isin(user_set)], val_df[val_df.user_id.isin(user_set)], prev_centroids


def calculate_cont_metric(metric_mat):
    """
    Return continual metrics respectively: learning average, average backward, and backward transfer
    """
    learning_acc = np.average(np.diag(metric_mat))
    average_backward = np.average(metric_mat[-1])
    bwt = np.average(metric_mat[-1][:-1] - np.diag(metric_mat)[:-1])
    return learning_acc, average_backward, bwt


def linear_update(old_factors, new_factors, gamma=0.5):
    for idx in range(len(old_factors)):
        new_factors[idx] = (1-gamma) * old_factors[idx] + \
            gamma * new_factors[idx]

    return new_factors


if __name__ == '__main__':
    data_folder = "data/movies/"
    files = os.listdir(data_folder)
    for file in files:
        if "pkl" in file:
            df = read_pkl_df(file, data_folder)
            if len(df) > 0:
                print(file)
                print(df.user_id.max(), df.item_id.max())
                breakpoint()
                # df.to_pickle(data_folder+file)
            # breakpoint()

    # metric_mat = [[0.03303383, 0., 0., 0., 0.],
    #               [0.03614681, 0.02035107, 0., 0., 0.],
    #               [0.01893968, 0.00999501, 0.0191098, 0., 0.],
    #               [0.01065627, 0.00632919, 0.00702464, 0.0221086, 0.],
    #               [0.01560896, 0.00898273, 0.00731747, 0.01719842, 0.01740954]]
    # print(calculate_cont_metric(metric_mat))
