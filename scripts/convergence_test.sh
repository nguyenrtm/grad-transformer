#!/bin/bash

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes=1 --main_process_port 20000 src/finetune_aquarat_qwen.py \
--dataset_path deepmind/aqua_rat \
--save_fname convergence_test \
--model_path Qwen/Qwen2.5-7B-Instruct \
--num_training_steps 30000 \
--num_samples_per_train_dataset 1024 \
--num_samples_per_val_dataset 100 \
--lr 1e-4 \
--lora_rank 2 \
--batch_size 1 \
--weight_decay 0.01 \
--max_length 4096 \
--max_length_eval 4096 \
--max_new_tokens 4096 \
--evaluate_every 100 \
--tune_option all_attention_layers \
--early_stopping_patience 1000 \
--early_stopping_min_delta 0.0001 \
--seed 42