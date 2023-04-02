import os
import sys
from time import time
from typing import Dict, List, Tuple
import pandas as pd
from pandas import DataFrame
from argparser import set_env
import pickle
import numpy as np
import torch
from reservoir import Reservoir
from copy import deepcopy


def read_raw_data(data_path: str, user_key: str, item_key: str, time_key: str, rating: str) -> DataFrame:
    df = pd.read_csv(data_path, names=[
                     "item_id", "user_id", "rating", "timestamp"])
    name_map = {
        user_key: "user_id",
        item_key: "item_id",
        rating: "rating",
        time_key: "timestamp",
    }
    return df.rename(columns=name_map).filter(items=["user_id", "item_id", "rating", "timestamp"])


def read_movielens(data_path: str, user_key: str, item_key: str, time_key: str, rating: str) -> DataFrame:
    df = pd.read_csv(data_path, names=["user_id", "item_id", "rating", "timestamp"],
                     sep="::", engine='python', encoding='latin-1')
    name_map = {
        user_key: "user_id",
        item_key: "item_id",
        rating: "rating",
        time_key: "timestamp",
    }
    return df.rename(columns=name_map).filter(items=["user_id", "item_id", "rating", "timestamp"])


def drop_duplicate(df: DataFrame) -> DataFrame:
    df = df.sort_values(by=["timestamp", "user_id"]).reset_index(drop=True)
    return df.drop_duplicates(subset=["user_id", "item_id"], keep="last").reset_index(drop=True)


def drop_spare(df: DataFrame, min_thres: int = 10) -> DataFrame:
    current_length = 0

    while current_length != len(df):
        current_length = len(df)
        # drop spare items
        item_pop = df.groupby("item_id")["user_id"].count()
        good_items = item_pop.index[item_pop >= min_thres]
        df = df[df["item_id"].isin(good_items)]

        # drop spare items
        user_pop = df.groupby("user_id")["user_id"].count()
        good_users = user_pop.index[user_pop >= min_thres]
        df = df[df["user_id"].isin(good_users)]

    return df


def train_test_split_by_time(df: DataFrame, test_ratio: float = 0.2) -> Tuple[DataFrame, DataFrame]:
    """Train/Test split by timestamp. Sort all ratings by time, the earlier ones are training set, the remaining is test set.

    Args:
        df (DataFrame): Interaction dataframe
        test_ratio (float, optional): Defaults to 0.25.

    Returns:
        Tuple[DataFrame, DataFrame]: Train and Test df
    """
    pivot = round(len(df) * (1-test_ratio))
    return df.head(pivot).reset_index(drop=True), df.tail(-pivot).reset_index(drop=True)


def train_test_split_by_users(df: DataFrame, test_ratio: float = 0.2) -> Tuple[DataFrame, DataFrame]:
    """Train/Test split based on users. For each user, the former interactions are for training, while the remaining is for test.

    Args:
        df (DataFrame): Interaction dataframe
        test_ratio (float, optional): Defaults to 0.25.

    Returns:
        Tuple[DataFrame, DataFrame]: _description_
    """
    last_item = df.groupby("user_id")["item_id"].transform("last")
    train = df[df["item_id"] != last_item].reset_index(drop=True)
    test = df[df["item_id"] == last_item].reset_index(drop=True)
    # cnt = 0
    # train_users = train.user_id.unique()
    # for user in test.user_id.unique():
    #     if user in train_users:
    #         cnt += 1
    # breakpoint()

    return train, test


def train_val_test_split_by_users(train_df: DataFrame, val_ratio: float = 0.1, test_ratio: float = 0.2) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Train/Val/Test split based on users. For each user, the former interactions are for training, the middle ones are for validation, while the remaining is for test.

    Args:
        df (DataFrame): Interaction dataframe
        val_ratio (float, optional): Defaults to 0.1.
        test_ratio (float, optional): Defaults to 0.2.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: _description_
    """
    train_ratio = 1 - val_ratio - test_ratio
    train_lst = []
    val_lst = []
    test_lst = []

    for uid in train_df["user_id"].unique():
        watches = train_df.loc[train_df.user_id == uid]

        train_lst.append(watches.iloc[:int(len(watches)*train_ratio)])
        val_lst.append(
            watches.iloc[int(len(watches)*train_ratio):int(len(watches)*(1-test_ratio))])
        test_lst.append(watches.iloc[int(len(watches)*(1-test_ratio)):])

    train = pd.concat(train_lst).reset_index(drop=True)
    val = pd.concat(val_lst).reset_index(drop=True)
    test = pd.concat(test_lst).reset_index(drop=True)

    return train, val, test


def reservoir_sampling(train_dfs, mem_size, seed):
    """Reservoir sampling.

    Args:
        train_dfs (list): list of training dataframes
        mem_size (int): memory size
        seed (int): random seed

    Returns:
        list: list of training dataframes
    """

    start_time = time()
    reservoir = Reservoir(mem_size, seed)
    task_buffers = []
    task_buffers.append(deepcopy(reservoir.buffer))  # first task no buffer

    for task in range(0, len(train_dfs)-1):
        # sample from previous task
        reservoir.update_new_task(train_dfs[task])
        task_buffers.append(deepcopy(reservoir.buffer))

    print("Reservoir sampling runtime:", time() - start_time)
    return task_buffers


def split_tasks_by_user(df: DataFrame, n_tasks: int = 5, mem_size: int = 5000, type: str = "uniform", test_ratio: int = 0.2, seed: int = 123) -> Dict:
    """Split tasks in Continual Learning setting, with memory buffer

    Args:
        df (DataFrame): filtered rating DataFrame
        n_tasks (int, optional): #task. Defaults to 5.
        mem_size (float, optional): memory buffer size. Absolute or relative value (percentage). Default is 5%.
        type (str, optional): task split strategy. Defaults to "uniform".
            - "uniform": same number of new users and items each task.
        test_ratio (int, optional): Train/Test split ratio. Defaults to 0.25 (75/25).
        seed (int, optional): random seed. Defaults to 123.

    Returns:
        Dict: dictionary contains dataset meta data and split Tasks, each divided into Train/Valid/Test.
    """
    # universal params
    n_users = df.user_id.nunique()
    n_items = df.item_id.nunique()
    print("Whole dataset:")
    print("#interactions:", len(df))
    print("#users:", n_users)
    print("#items:", n_items)
    dataset = {
        "n_tasks": n_tasks,
        "n_users": n_users,
        "n_items": n_items,
    }

    # split tasks
    # divide users and items to "equal" subsets
    user_sets = np.array_split(df.user_id.unique(), n_tasks)
    item_sets = np.array_split(df.item_id.unique(), n_tasks)
    cur_user_list = np.array([])
    cur_item_list = np.array([])

    task_dfs = []
    train_dfs = []
    valid_dfs = []
    test_dfs = []
    memory_dfs = []
    memory_dfs.append(pd.DataFrame())  # first task no buffer

    for task in range(n_tasks):
        # combine previous tasks' users and items with new ones
        cur_user_list = np.concatenate(
            (cur_user_list, user_sets[task]))
        cur_item_list = np.concatenate(
            (cur_item_list, item_sets[task]))

        # get sub-df upto current task, then remove previous tasks' ratings
        task_df = df[df.user_id.isin(cur_user_list)
                     & df.item_id.isin(cur_item_list)] \
            .reset_index(drop=True)
        for idx in range(task):
            task_df = pd.concat([task_df, task_dfs[idx]]) \
                        .drop_duplicates(keep=False)\
                        .reset_index(drop=True)

        task_dfs.append(task_df)

        # train_df, test_df = train_test_split_by_users(
        #     task_df, test_ratio=test_ratio)
        # train_df, valid_df = train_test_split_by_users(
        #     train_df, test_ratio=test_ratio)

        # train_df, test_df = train_test_split_by_time(
        #     task_df, test_ratio=test_ratio)
        # train_df, valid_df = train_test_split_by_time(
        #     train_df, test_ratio=test_ratio)

        train_df, valid_df, test_df = train_val_test_split_by_users(task_df)

        train_dfs.append(train_df)
        valid_dfs.append(valid_df)
        test_dfs.append(test_df)

        print(f"Task {task+1}:")
        print("#interactions:", len(task_df))
        print("#users:", task_df.user_id.nunique())
        print("#items:", task_df.item_id.nunique())
        print("#train={}, #valid={}, #test={}".format(
            len(train_df), len(valid_df), len(test_df)))

    # store train/valid/test tasks' df
    dataset["train"] = train_dfs
    dataset["valid"] = valid_dfs
    dataset["test"] = test_dfs

    # get memory buffer for each task
    task_buffers = reservoir_sampling(train_dfs, mem_size, seed)

    dataset["buffers"] = task_buffers

    return dataset


def split_tasks_by_time(df: DataFrame, n_tasks: int = 5, mem_size: int = 5000, type: str = "uniform", test_ratio: int = 0.2, seed: int = 123) -> Dict:
    """Split tasks in Continual Learning setting, with memory buffer

    Args:
        df (DataFrame): filtered rating DataFrame
        n_tasks (int, optional): #task. Defaults to 5.
        mem_size (float, optional): memory buffer size. Absolute or relative value (percentage). Default is 5%.
        type (str, optional): task split strategy. Defaults to "uniform".
            - "uniform": same number of new users and items each task.
        test_ratio (int, optional): Train/Test split ratio. Defaults to 0.25 (75/25).
        seed (int, optional): random seed. Defaults to 123.

    Returns:
        Dict: dictionary contains dataset meta data and split Tasks, each divided into Train/Valid/Test.
    """
    # universal params
    n_users = df.user_id.nunique()
    n_items = df.item_id.nunique()
    print("Whole dataset:")
    print("#interactions:", len(df))
    print("#users:", n_users)
    print("#items:", n_items)
    dataset = {
        "n_tasks": n_tasks,
        "n_users": n_users,
        "n_items": n_items,
    }

    # split tasks
    task_dfs = np.array_split(df, n_tasks)
    train_dfs = []
    valid_dfs = []
    test_dfs = []
    memory_dfs = []
    memory_dfs.append(pd.DataFrame())  # first task no buffer

    for task in range(n_tasks):
        train_df, test_df = train_test_split_by_time(
            df=task_dfs[task], test_ratio=test_ratio)
        train_df, valid_df = train_test_split_by_time(
            df=train_df, test_ratio=test_ratio)

        train_dfs.append(train_df)
        valid_dfs.append(valid_df)
        test_dfs.append(test_df)

        print(f"Task {task+1}:")
        print("#interactions:", len(task_dfs[task]))
        print("#users:", task_dfs[task].user_id.nunique())
        print("#items:", task_dfs[task].item_id.nunique())

    # store train/valid/test tasks' df
    dataset["train"] = train_dfs
    dataset["valid"] = valid_dfs
    dataset["test"] = test_dfs

    # get memory buffer for each task
    task_buffers = reservoir_sampling(train_dfs, mem_size, seed)

    dataset["buffers"] = task_buffers

    return dataset


def count_intersection(list1, list2):
    """Count intersection of two lists

    Args:
        list1 (list): list 1
        list2 (list): list 2

    Returns:
        int: intersection size
    """
    return len(set(list1) & set(list2))


if __name__ == "__main__":
    # data source: http://jmcauley.ucsd.edu/data/amazon/index_2014.html / rating only
    # original names: Digital_Music.csv  Kindle_Store.csv  Movies_and_TV.csv  ratings_Books.csv
    # changed to: music, kindle, movies, books

    args = set_env()

    # load data
    if args.dataset == "movielens":
        inter_df = read_movielens(
            os.environ["interaction_dir"],
            user_key=args.user_key, item_key=args.item_key, time_key=args.time_key, rating=args.rating)
    else:
        inter_df = read_raw_data(
            os.environ["interaction_dir"],
            user_key=args.user_key, item_key=args.item_key, time_key=args.time_key, rating=args.rating)
    # refine data
    # breakpoint()
    parsed_inter_df = inter_df \
        .pipe(drop_duplicate) \
        .pipe(drop_spare, min_thres=args.min_thres) \
        .reset_index(drop=True)\
        .drop(columns=["timestamp"])

    if args.split == "user":
        dataset_dict = split_tasks_by_user(parsed_inter_df, n_tasks=args.n_tasks, type=args.split_type,
                                           test_ratio=args.test_ratio, mem_size=args.buffer_size, seed=args.seed)
    elif args.split == "time":
        dataset_dict = split_tasks_by_time(parsed_inter_df, n_tasks=args.n_tasks, type=args.split_type,
                                           test_ratio=args.test_ratio, mem_size=args.buffer_size, seed=args.seed)

    torch.save(dataset_dict,
               f"dataset/processed/{args.dataset}.pt")
