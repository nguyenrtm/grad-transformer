#!/bin/bash

python src/idx_split.py \
--save_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat \
--dataset_path deepmind/aqua_rat \
--num_splits 10 \
--train_test_split 0.9

python src/gen_idx.py \
--save_fname aqua_rat_10/distributed_10_split_0.9_aqua_rat_shadow_datasets.pkl \
--subset_size 1024 \
--num_sets 100000 \
--dataset_path deepmind/aqua_rat \
--initial_idx_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_5.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_6.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_7.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_8.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_9.pkl