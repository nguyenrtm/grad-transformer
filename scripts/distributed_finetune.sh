#!/bin/bash

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes=1 --main_process_port 20000 src/finetune_aquarat_qwen.py \
--train_indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_0.pkl \
--test_indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_0.pkl \
--num_samples_per_val_dataset 100 \
--dataset_path deepmind/aqua_rat \
--model_path Qwen/Qwen2.5-3B-Instruct \
--num_training_steps 20000 \
--lr 1e-5 \
--lora_rank 2 \
--batch_size 1 \
--weight_decay 0.01 \
--max_length 4096 \
--max_length_eval 4096 \
--max_new_tokens 4096 \
--evaluate_every 500 \
--tune_option all_attention_layers \
--scheduler \
--seed 42

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes=1 --main_process_port 20000 src/finetune_aquarat_qwen.py \
--train_indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_1.pkl \
--test_indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_1.pkl \
--num_samples_per_val_dataset 100 \
--dataset_path deepmind/aqua_rat \
--model_path Qwen/Qwen2.5-3B-Instruct \
--num_training_steps 20000 \
--lr 1e-5 \
--lora_rank 2 \
--batch_size 1 \
--weight_decay 0.01 \
--max_length 4096 \
--max_length_eval 4096 \
--max_new_tokens 4096 \
--evaluate_every 500 \
--tune_option all_attention_layers \
--scheduler \
--seed 42

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes=1 --main_process_port 20000 src/finetune_aquarat_qwen.py \
--train_indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_2.pkl \
--test_indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_2.pkl \
--num_samples_per_val_dataset 100 \
--dataset_path deepmind/aqua_rat \
--model_path Qwen/Qwen2.5-3B-Instruct \
--num_training_steps 20000 \
--lr 1e-5 \
--lora_rank 2 \
--batch_size 1 \
--weight_decay 0.01 \
--max_length 4096 \
--max_length_eval 4096 \
--max_new_tokens 4096 \
--evaluate_every 500 \
--tune_option all_attention_layers \
--scheduler \
--seed 42

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes=1 --main_process_port 20000 src/finetune_aquarat_qwen.py \
--train_indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_3.pkl \
--test_indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_3.pkl \
--num_samples_per_val_dataset 100 \
--dataset_path deepmind/aqua_rat \
--model_path Qwen/Qwen2.5-3B-Instruct \
--num_training_steps 20000 \
--lr 1e-5 \
--lora_rank 2 \
--batch_size 1 \
--weight_decay 0.01 \
--max_length 4096 \
--max_length_eval 4096 \
--max_new_tokens 4096 \
--evaluate_every 500 \
--tune_option all_attention_layers \
--scheduler \
--seed 42

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes=1 --main_process_port 20000 src/finetune_aquarat_qwen.py \
--train_indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_4.pkl \
--test_indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_4.pkl \
--num_samples_per_val_dataset 100 \
--dataset_path deepmind/aqua_rat \
--model_path Qwen/Qwen2.5-3B-Instruct \
--num_training_steps 20000 \
--lr 1e-5 \
--lora_rank 2 \
--batch_size 1 \
--weight_decay 0.01 \
--max_length 4096 \
--max_length_eval 4096 \
--max_new_tokens 4096 \
--evaluate_every 500 \
--tune_option all_attention_layers \
--scheduler \
--seed 42