#!/bin/bash

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes=1 --main_process_port 20001 src/gen_gradients.py \
--model_path Qwen/Qwen2.5-3B-Instruct \
--dataset_path deepmind/aqua_rat \
--indices_path ../grad-transformer/indices/split_0.5_aqua_rat_1024_100000.pkl \
--starting_indices 1 \
--ending_indices 50 \
--num_training_steps 2200 \
--converged_step 2000 \
--lora_rank 2 \
--batch_size 1 \
--lr 1e-4 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--max_length 4096 \
--tune_option all_attention_layers

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes=1 --main_process_port 20000 src/gen_gradients.py \
--model_path Qwen/Qwen2.5-7B-Instruct \
--dataset_path deepmind/aqua_rat \
--indices_path ../grad-transformer/indices/split_0.5_aqua_rat_1024_100000.pkl \
--starting_indices 1 \
--ending_indices 50 \
--num_training_steps 1400 \
--converged_step 1200 \
--lora_rank 2 \
--batch_size 1 \
--lr 1e-4 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--max_length 4096 \
--tune_option all_attention_layers