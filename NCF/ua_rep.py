from copy import deepcopy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from time import time
import pickle5 as pickle

from model import *
from config import get_args
from data_loader import get_loader
from utils import calculate_cont_metric, split_df_by_user
from seed_everything import seed_everything


start_time = time()
args = get_args()
seed_everything(args.seed)
float_formatter = "{:.6f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Getting the number of users and movies
if 'ml' in args.data_path:
    num_users = 6041
    num_movies = 3953
elif 'books' in args.data_path:
    num_users = 38122
    num_movies = 35737
elif 'movies' in args.data_path:
    num_users = 16003
    num_movies = 9775
elif 'kindle' in args.data_path:
    num_users = 8534
    num_movies = 10069

# Creating the architecture of the Neural Network
if args.model == 'NCF':
    model = NCF(num_users, num_movies, args.emb_dim, args.layers)

if torch.cuda.is_available():
    model.cuda()
before = deepcopy(model.state_dict())
"""Print out the network information."""
num_params = 0
for p in model.parameters():
    num_params += p.numel()
print(model)
print("The number of parameters: {}".format(num_params))

criterion = nn.BCEWithLogitsLoss()  # CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)


def train(train_loader, val_loader):
    best_loss = 9999.
    best_model = None

    # Training
    for epoch in range(args.start_epoch, args.n_epochs):
        train_loss = 0
        model.train()
        for s, (x, n) in enumerate(train_loader):
            x = x.long().to(device)
            n = n.long().to(device)
            u = Variable(x[:, 0])
            v = Variable(x[:, 1])
            # r = Variable(x[:,2]).float()

            pred, neg_pred = model(u, v, n)
            loss = criterion(pred, torch.ones(pred.size(0)).to(device)) \
                + criterion(neg_pred, torch.zeros(neg_pred.size(0)).to(device))
            train_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: '+str(epoch+1)+' loss: '+str(train_loss/(s+1)))
        if (epoch+1) % args.val_step == 0:
            # Validation
            model.eval()
            val_loss = 0
            val_hits = 0
            n_users = 0
            with torch.no_grad():
                for s, (x, n) in enumerate(val_loader):
                    x = x.long().to(device)
                    n = n.long().to(device)
                    u = Variable(x[:, 0])
                    v = Variable(x[:, 1])
                    # r = Variable(x[:,2]).float()
                    n_users += len(x)

                    pred, neg_pred = model(u, v, n)
                    loss = criterion(pred, torch.ones(pred.size(0)).to(device)) \
                        + criterion(neg_pred,
                                    torch.zeros(neg_pred.size(0)).to(device))
                    val_loss += loss.item()

                    # Hit Ratio
                    pred = torch.cat(
                        (pred.unsqueeze(1), neg_pred.view(-1, args.neg_cnt)), 1)
                    _, topk = torch.sort(pred, 1, descending=True)
                    val_hits += sum([0 in topk[k, :args.at_k]
                                    for k in range(topk.size(0))])

            val_loss = val_loss / (s+1)
            if val_loss == 0:
                val_loss = 9999.1
            print('[val loss] : '+(str(val_loss)) +
                  ' [val hit ratio] : '+str(val_hits/n_users))
            if best_loss > (val_loss):
                best_loss = (val_loss)
                model_dir = args.model_path + args.model
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                best_model = deepcopy(model)

    return best_model


def test(test_loader, model):
    # Test
    model.eval()
    test_loss = 0
    test_hits = 0
    n_users = 0
    with torch.no_grad():
        for s, (x, n) in enumerate(test_loader):
            x = x.long().to(device)
            n = n.long().to(device)
            u = Variable(x[:, 0])
            v = Variable(x[:, 1])
            # r = Variable(x[:,2]).float()
            n_users += len(x)

            pred, neg_pred = model(u, v, n)
            loss = criterion(pred, torch.ones(pred.size(0)).to(device)) \
                + criterion(neg_pred, torch.zeros(neg_pred.size(0)).to(device))
            test_loss += loss.item()

            # Hit Ratio
            pred = torch.cat(
                (pred.unsqueeze(1), neg_pred.view(-1, args.neg_cnt)), 1)
            _, topk = torch.sort(pred, 1, descending=True)
            test_hits += sum([0 in topk[k, :args.at_k]
                             for k in range(topk.size(0))])

    t_loss = test_loss/(s+1)
    t_hits = test_hits/n_users

    print('[test loss] : '+str(t_loss) +
          ' [test hit ratio] : '+str(t_hits))
    return t_loss, t_hits


if __name__ == '__main__':
    n_tasks = args.n_tasks
    seed = args.seed
    beta = args.beta
    loss_mat = np.ones((n_tasks, n_tasks)) * 9999.
    hit_mat = np.zeros((n_tasks, n_tasks))
    prev_centroids = "random"

    for task in range(n_tasks):
        print('Task: '+str(task+1))
        # user cluster data split
        if task == 0:
            train_loader = get_loader(args.data_path, f"train_{task}.pkl", f"neg_{task}.npy",
                                      args.neg_cnt, args.batch_size, args.data_shuffle)
            val_loader = get_loader(args.data_path, f"val_{task}.pkl", f"neg_{task}.npy",
                                    args.neg_cnt, args.batch_size, False)

            best_model = train(train_loader, val_loader)
            model.load_state_dict(best_model.state_dict())
        else:  # task > 0
            for step in range(args.reptile_steps):
                before = deepcopy(model.state_dict())

                with open(args.data_path+f"train_{task}.pkl", 'rb') as f:
                    train_df = pickle.load(f)
                with open(args.data_path+f"buffer_{task}.pkl", 'rb') as f:
                    buffer_df = pickle.load(f)
                train_df = pd.concat([train_df, buffer_df], axis=0)
                with open(args.data_path+f"val_{task}.pkl", 'rb') as f:
                    val_df = pickle.load(f)

                # learn by each group of similar users
                group_id = 1
                va_df = None
                for t_df, v_df, centroids in split_df_by_user(train_df, val_df, best_model, seed=seed, n_clusters=args.n_clusters, prev_centroids=prev_centroids):
                    if len(t_df) == 0:
                        continue
                    print(f"Group {group_id}")
                    # breakpoint()
                    group_id += 1
                    prev_centroids = centroids
                    if va_df is not None:
                        v_df = pd.concat([va_df, v_df]).reset_index(drop=True)
                    train_loader = get_loader(args.data_path, "", f"neg_{task}.npy", args.neg_cnt,
                                              args.batch_size, args.data_shuffle, df=t_df)
                    val_loader = get_loader(args.data_path, "", f"neg_{task}.npy", args.neg_cnt,
                                            args.batch_size, False, df=v_df)
                    best_model = train(train_loader, val_loader)
                    va_df = deepcopy(v_df)

                    after = best_model.state_dict()
                    model.load_state_dict(
                        {name: before[name] + ((after[name] - before[name]) * beta) for name in before})
                    before = deepcopy(model.state_dict())
                # breakpoint()
            print("Done learning by each group of similar users")
            print("Refine whole dataset")

            train_loader = get_loader(args.data_path, f"buffer_{task}.pkl", f"neg_{task}.npy",
                                      args.neg_cnt, args.batch_size, args.data_shuffle)
            val_loader = get_loader(args.data_path, f"val_{task}.pkl", f"neg_{task}.npy",
                                    args.neg_cnt, args.batch_size, False)
            best_model = train(train_loader, val_loader)
            after = best_model.state_dict()
            model.load_state_dict(
                {name: before[name] + ((after[name] - before[name]) * beta) for name in before})
            before = deepcopy(model.state_dict())

            train_loader = get_loader(args.data_path, f"train_{task}.pkl", f"neg_{task}.npy",
                                      args.neg_cnt, args.batch_size, args.data_shuffle)
            best_model = train(train_loader, val_loader)
            after = best_model.state_dict()
            model.load_state_dict(
                {name: before[name] + ((after[name] - before[name]) * beta) for name in before})
            before = deepcopy(model.state_dict())
        # breakpoint()

        # test
        for t in range(task+1):
            print(f"Testing task {t+1} at time {task+1}")
            test_loader = get_loader(args.data_path, f"test_{t}.pkl", f"neg_{t}.npy",
                                     args.neg_cnt, args.batch_size, False)
            loss, hit = test(test_loader, best_model)
            loss_mat[task][t] = loss
            hit_mat[task][t] = hit

    print('Loss Matrix: \n', loss_mat)
    print('Continual metrics: ', calculate_cont_metric(loss_mat))
    print('Hit Matrix: \n', hit_mat)
    print('Continual metrics: ', calculate_cont_metric(hit_mat))
    print("Running time: ", time() - start_time)
