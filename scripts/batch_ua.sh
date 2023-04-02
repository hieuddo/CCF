for dataset in books kindle movies movielens
do
    for seed in 12 21 24 42 88
    do
        # ./scripts/plusmf.sh $dataset $seed &
        ./scripts/mf_ua.sh $dataset $seed &
    done
done
