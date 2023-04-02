import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # algorithm
    parser.add_argument('--er', action="store_true")
    parser.add_argument('--reptile', action="store_true")
    parser.add_argument('--mer', action="store_true")
    parser.add_argument('--user_align', action="store_true")

    # data
    parser.add_argument('--mode', type=str, default="train",
                        help='train / test')
    parser.add_argument('--model-path', type=str, default="./model_")
    parser.add_argument('--dataset', type=str, default="movielens")
    parser.add_argument("--user_key", default="user_id", type=str)
    parser.add_argument("--item_key", default="item_id", type=str)
    parser.add_argument("--time_key", default="timestamp", type=str)
    parser.add_argument("--rating", default="rating", type=str)
    parser.add_argument('--data_path', type=str, default="./data/ml-1m/")
    parser.add_argument('--data_shuffle', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--val-step', type=int, default=5)
    parser.add_argument('--test-epoch', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--neg_cnt', type=int, default=100)
    parser.add_argument('--at-k', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument("--reptile_steps", default=3, type=int)

    # preprocess
    parser.add_argument("--min_thres", default=10, type=int)
    parser.add_argument("--n_clusters", default=10, type=int)
    parser.add_argument("--n_tasks", default=5, type=int)
    parser.add_argument("--split_type", default='uniform', type=str)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    parser.add_argument("--buffer_size", default=50000, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument('--model', type=str, default='NCF',
                        help='NCF / ONCF / CCF')

    parser.add_argument('--emb-dim', type=int, default=16)
    parser.add_argument('--layers', default=[32, 32, 16, 8])
    parser.add_argument('--outer-layers', default=[1, 16, 16, 16, 16])
    parser.add_argument('--conv-layers', default=[2, 32, 16, 8])
    #parser.add_argument('--conv-layers', default=[2,16,16,16,16])
    parser.add_argument('--user-cnt', type=int, default=25678)  # 6041)
    parser.add_argument('--item-cnt', type=int, default=25816)  # 3954)

    # parser.add_argument('--train-path', type=str, default='train_score.pkl')
    # parser.add_argument('--val-path', type=str, default='val_score.pkl')
    # parser.add_argument('--test-path', type=str, default='test_score.pkl')
    # parser.add_argument('--neg-path', type=str, default='neg_score.npy')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    if args.er:
        print("ER")
    else:
        print("Not ER")
