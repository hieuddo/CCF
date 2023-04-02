import cornac
from cornac.models import MF
from cornac.metrics import MAE, MSE
from cornac.eval_methods import BaseMethod
import numpy as np
import pandas as pd
from copy import deepcopy
from time import time
from argparser import set_env
import os
import torch
from utils import group_users, calculate_cont_metric


if __name__ == "__main__":
    args = set_env()

    dataset_dict = torch.load(
        f"dataset/processed/{args.dataset}.pt")

    # init variables
    use_gpu = True if args.gpu else False
    seed = args.seed
    k_latent = args.hidden_dim
    beta = args.beta
    gamma = args.gamma
    net_structure = args.net_structure
    n_tasks = dataset_dict["n_tasks"]
    train_dfs = dataset_dict["train"]
    valid_dfs = dataset_dict["valid"]
    test_dfs = dataset_dict["test"]
    if args.buffer_size > 0:
        buffer_dfs = dataset_dict["buffers"]
    else:
        buffer_dfs = [pd.DataFrame([]) for _ in range(n_tasks)]

    # # metrics matrices
    # MF
    mae = np.ones((n_tasks, n_tasks), dtype=float) * 5
    mse = np.ones((n_tasks, n_tasks), dtype=float) * 5

    # init models
    # predefine fixed parameters
    mf_model = MF(k=k_latent, use_bias=True, early_stop=True,
                  verbose=False, max_iter=500, seed=seed)

    agg_train_df = pd.concat([train_dfs[idx] for idx in range(n_tasks)])
    agg_train_list = agg_train_df.values.tolist()
    global_eval_method = BaseMethod.from_splits(agg_train_list, agg_train_list, rating_threshold=3.0,
                                                exclude_unknowns=True, seed=seed, verbose=False)
    global_uid_map = global_eval_method.global_uid_map
    global_iid_map = global_eval_method.global_iid_map
    list_eval_method = []

    previous_mf_model = None

    start_time = time()

    for task in range(n_tasks):
        # training
        mf_model.trainable = True
        # vae_model.trainable = True
        print("Training task", task+1)
        if task > 0:
            group_id = 1
            train_dff = pd.concat([train_dfs[task], buffer_dfs[task]])\
                .reset_index(drop=True)
            for dfs in group_users(mf_model, train_dff, valid_dfs[task], test_dfs[task], n_groups=10, seed=seed):
                print("Training group", group_id)
                group_id += 1
                train_df, valid_df, test_df = dfs
                # breakpoint()
                train_list = train_df.values.tolist()
                valid_list = valid_df.values.tolist()
                valid_list = None
                test_list = test_df.values.tolist()
                if len(test_list) == 0:
                    test_list = train_list
                eval_method = BaseMethod.from_splits(train_list, test_list, valid_list, rating_threshold=3.0, verbose=False,
                                                     exclude_unknowns=True, seed=seed, uid_map=global_uid_map, iid_map=global_iid_map)
                exp = cornac.Experiment(eval_method=eval_method,
                                        models=[mf_model, ],
                                        metrics=[
                                            MAE(), MSE(),
                                        ],
                                        user_based=False, save_dir="CornacExp",)
                exp.run()

        # refine with whole task
        train_set = pd.concat(
            [train_dfs[task], buffer_dfs[task]]).values.tolist()
        valid_set = valid_dfs[task].values.tolist()
        test_set = test_dfs[task].values.tolist()
        # breakpoint()
        eval_method = BaseMethod.from_splits(train_set, test_set, valid_set, rating_threshold=3.0, verbose=False,
                                             exclude_unknowns=True, seed=seed, uid_map=global_uid_map, iid_map=global_iid_map)
        list_eval_method.append(deepcopy(eval_method))

        exp = cornac.Experiment(eval_method=eval_method,
                                models=[mf_model, ],
                                metrics=[
                                    MAE(), MSE(),
                                ],
                                user_based=False, save_dir="CornacExp",)

        exp.run()

        # backward testing
        mf_model.trainable = False

        for t in range(0, task+1):  # from first to current task
            print(f"Testing task {t+1} at time {task+1}")
            # test_set = test_dfs[t].values.tolist()
            eval_method = BaseMethod.from_splits(agg_train_list, test_set, rating_threshold=3.0, verbose=False, exclude_unknowns=True,
                                                 seed=seed, uid_map=global_uid_map, iid_map=global_iid_map)
            exp = cornac.Experiment(eval_method=eval_method,
                                    models=[mf_model, ],
                                    metrics=[
                                        MAE(), MSE(),
                                    ],
                                    user_based=False, save_dir="CornacExp",)
            exp.run()
            # breakpoint()
            # exp.result[0]: MF results
            mae[task][t] = exp.result[0].metric_avg_results["MAE"]
            mse[task][t] = exp.result[0].metric_avg_results["MSE"]

    # finalize metrics
    print("Running time:", time() - start_time)

    print(mae, "\n\n", mse, "\n\n")

    print("Continual metrics: LA/Avg/BWT")
    print(calculate_cont_metric(mae))
    print(calculate_cont_metric(mse))
