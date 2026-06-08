#!/bin/bash

python src/gen_idx.py \
--subset_size 1024 \
--num_sets 100000 \
--dataset_path deepmind/aqua_rat \
--split 0.5 \
--single_client