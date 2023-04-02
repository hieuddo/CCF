# from cornac.utils import get_rng
# from cornac.utils.init_utils import normal
import numpy as np
from sklearn.cluster import KMeans
# from time import time
from copy import deepcopy
import torch
import torch.nn as nn
import argparse
from cornac.metrics.rating import RatingMetric
from cornac.models import MF, VAECF
from cornac.metrics import MSE, Recall
from cornac.eval_methods import rating_eval, ranking_eval


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def vaecf_get_init_params(old_model, new_train_set, seed, model_type='vaecf'):
    if not old_model:
        return None

    old_iid_map = old_model.train_set.iid_map
    new_iid_map = new_train_set.iid_map
    device = old_model.device

    new_vae = deepcopy(old_model.vae).to(device)
    hidden_size = old_model.autoencoder_structure[0]
    new_num_items = new_train_set.num_items
    new_vae.encoder.fc0 = nn.Linear(
        new_num_items, hidden_size).to(device)
    new_vae.decoder.fc1 = nn.Linear(
        hidden_size, new_num_items).to(device)

    # remap item_embedding
    with torch.no_grad():
        for key in old_iid_map.keys():
            if new_iid_map.get(key):
                old_mapped_iid = old_iid_map.get(key)
                new_mapped_iid = new_iid_map.get(key)
                new_vae.encoder.fc0.weight[:, new_mapped_iid] \
                    = old_model.vae.encoder.fc0.weight[:, old_mapped_iid]
                new_vae.decoder.fc1.weight[new_mapped_iid] = old_model.vae.decoder.fc1.weight[old_mapped_iid]
    return new_vae


def calculate_cont_metric(metric_mat):
    """
    Return continual metrics respectively: learning average, average backward, and backward transfer
    """
    learning_acc = np.average(np.diag(metric_mat))
    average_backward = np.average(metric_mat[-1])
    bwt = np.average(metric_mat[-1][:-1] - np.diag(metric_mat)[:-1])
    return learning_acc, average_backward, bwt


def consolidate_mf_model(old_model: MF, model: MF, beta: float, algo: str = "none"):
    """_summary_

    Args:
        model (MF): _description_
        beta (float): _description_
        gamma (float): _description_
        algo (str, optional): meta update algorithm. Defaults to "none". Options: ["none", "gacf", "gacf+"]
    """
    clone_old = deepcopy(old_model)
    clone_new = deepcopy(model)

    clone_new.u_factors = clone_old.u_factors + \
        beta * (clone_new.u_factors - clone_old.u_factors)
    clone_new.u_biases = clone_old.u_biases + \
        beta * (clone_new.u_biases - clone_old.u_biases)
    clone_new.i_factors = clone_old.i_factors + \
        beta * (clone_new.i_factors - clone_old.i_factors)
    clone_new.i_biases = clone_old.i_biases + \
        beta * (clone_new.i_biases - clone_old.i_biases)

    return clone_new


def consolidate_vae_model(old_model: VAECF, model: VAECF, beta: float):
    """
    """
    before = old_model.vae.state_dict()
    after = model.vae.state_dict()
    return {name: before[name] + ((after[name] - before[name]) * beta) for name in before}


def centroid_vae(item_embeddings, gamma=0.15, sampling_ratio=1.0, seed=123):
    """_summary_

    Args:
        item_embeddings (_type_): _description_
        gamma (float, optional): _description_. Defaults to 0.15.
        sampling_ratio (float, optional): _description_. Defaults to 1.0.
        seed (int, optional): _description_. Defaults to 123.

    Returns:
        _type_: _description_
    """
    np.random.seed(seed)

    item_mat = deepcopy(item_embeddings)
    item_mat = item_mat.T
    for idx, i_embed in enumerate(item_mat):
        mask = np.random.choice([True, False], len(item_mat),
                                p=[sampling_ratio, 1-sampling_ratio])
        selected_mat = item_mat[mask]
        weights = torch.sum(selected_mat * i_embed, dim=1)
        zeros = torch.zeros(weights.shape).to(weights.device)
        weights = torch.where(weights > 0, weights, zeros)
        # weights[idx] = 0
        weights = weights/torch.sum(weights)
        false_centroid = torch.sum(selected_mat.T * weights, dim=1)
        item_embeddings[:, idx] = item_embeddings[:, idx] + \
            gamma * (false_centroid-item_embeddings[:, idx])
    return item_embeddings


def centroid_vae_reg(model: VAECF, gamma=0.15, sampling_ratio=1.0, seed=123):
    item_mat = deepcopy(model.vae.encoder.fc0.weight.data)
    return centroid_vae(item_mat, gamma=gamma, sampling_ratio=sampling_ratio, seed=seed)


def centroid_reg(factors_mat, gamma=0.1, n_clusters=10, prev_centroids="random", seed=123, sampling_ratio=0.5, max_iter=5000):
    if n_clusters < 0:  # n_cluster == -1
        return weighted_centroid(factors_mat, gamma, sampling_ratio=sampling_ratio, seed=seed), None

    kmeans = KMeans(n_clusters=n_clusters, init=prev_centroids,
                    random_state=seed, max_iter=max_iter).fit(factors_mat)
    prev_centroids = kmeans.cluster_centers_
    labels = kmeans.predict(factors_mat)
    for idx, lb in enumerate(labels):
        factors_mat[idx] = (1-gamma) * factors_mat[idx] + \
            gamma * prev_centroids[lb]
    return factors_mat, prev_centroids


def translate_uid_back(user_set, uid_map):
    """Translate uids from cornac.train_set back to original

    Args:
        user_set (np.array): set of uids in cornac.trainset
        train_set (_type_): cornac.trainset

    Returns:
        list: list of original uids
    """
    uids = []
    for key, item in uid_map.items():
        if np.isin(item, user_set):
            uids.append(key)
    return uids


def group_users(model, train_df, val_df, test_df, n_groups=10, seed=123):
    factor_mat = model.u_factors
    uid_map = model.train_set.uid_map
    kmeans = KMeans(n_clusters=n_groups, random_state=seed,
                    max_iter=5000).fit(factor_mat)
    labels = kmeans.predict(factor_mat)
    for group in range(n_groups):
        user_set = np.where(labels == group)
        user_ids = translate_uid_back(user_set, uid_map)
        yield train_df[train_df.user_id.isin(user_ids)], val_df[val_df.user_id.isin(user_ids)], test_df[test_df.user_id.isin(user_ids)]


def search_params(old_model, model, train_set, val_set, model_type="MF", sampling_ratio=1.0, seed=123):
    """
    select best model based on val_set
    """
    if model_type == "MF":
        higher_better = False
        metric = MSE()
    elif model_type == "VAECF":
        higher_better = True
        metric = Recall(k=50)

    compare_op = np.greater if higher_better else np.less

    # beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    beta_values = [0, 0.25, 0.5, 0.75, 1.0]
    gamma_values = [0, 0.25, 0.5, 0.75, 1.0]

    best_model = deepcopy(model)
    best_score = -np.inf if higher_better else np.inf
    best_beta = 0
    best_gamma = 0

    for beta in beta_values:
        for gamma in gamma_values:
            # if not old_model:
            #     exp_model = deepcopy(model)
            # else:
            if model_type == "MF":
                exp_model = consolidate_mf_model(old_model, model, beta)

                exp_model.u_factors, _ = centroid_reg(
                    exp_model.u_factors, gamma, 10, "random", seed=seed)
                # best_model from hyperopt has no train_set
                exp_model.train_set = deepcopy(model.train_set)
            elif model_type == "VAECF":
                new_state_dict = consolidate_vae_model(
                    old_model, model, beta)
                exp_model = deepcopy(model)
                exp_model.vae.load_state_dict(new_state_dict)
                exp_model.vae.encoder.fc0.weight.data = centroid_vae_reg(
                    exp_model, gamma, sampling_ratio, seed)
                exp_model.train_set = train_set

            if isinstance(metric, RatingMetric):
                score = rating_eval(exp_model, [metric], val_set)[0][0]
            else:
                score = ranking_eval(
                    exp_model,
                    [metric],
                    train_set,
                    val_set,
                    rating_threshold=3.0,
                    exclude_unknowns=True,
                    verbose=False,
                )[0][0]

            if compare_op(score, best_score):
                best_score = score
                best_model = deepcopy(exp_model)
                best_beta = beta
                best_gamma = gamma

            # del exp_model

    print("Best score: {}={}; beta={}, gamma={}".format(
        metric.name, best_score, best_beta, best_gamma))

    return best_model


def linear_update(old_factors, new_factors, gamma=0.5):
    for idx in range(len(old_factors)):
        new_factors[idx] = (1-gamma) * old_factors[idx] + \
            gamma * new_factors[idx]

    return new_factors


def weighted_centroid(factors_mat, gamma=0.2, sampling_ratio=0.5, seed=123):
    np.random.seed(seed)

    for idx in range(len(factors_mat)):
        mask = np.random.choice([True, False], len(factors_mat),
                                p=[sampling_ratio, 1-sampling_ratio])

        user_factor = factors_mat[idx]
        neighbors = factors_mat[mask]
        weights = np.sum(np.multiply(user_factor, neighbors), axis=1)
        weights = weights/np.sum(weights)
        weights = np.maximum(weights, 0)
        # print(weights)
        false_centroid = np.average(neighbors, axis=0, weights=weights)
        u_factor = (1-gamma) * user_factor + gamma * false_centroid
        factors_mat[idx] = u_factor

    return factors_mat


if __name__ == '__main__':

    # factors_mat = np.random.rand(10000, 20) - 0.5
    # # print(factors_mat)

    # start_time = time()
    # centroid_reg(factors_mat, gamma=0.2, n_clusters=-1, sampling_ratio=0.1,
    #              prev_centroids='random', seed=123)
    # print(time() - start_time)
    # # breakpoint()

    metric_mat = [[0.03303383, 0., 0., 0., 0.],
                  [0.03614681, 0.02035107, 0., 0., 0.],
                  [0.01893968, 0.00999501, 0.0191098, 0., 0.],
                  [0.01065627, 0.00632919, 0.00702464, 0.0221086, 0.],
                  [0.01560896, 0.00898273, 0.00731747, 0.01719842, 0.01740954]]
    print(calculate_cont_metric(metric_mat))
