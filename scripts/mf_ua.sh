dataset=$1
seed=$2

python -u mf_ua.py --dataset $dataset --seed $seed >results/mmf/$dataset/ua_$seed.txt

