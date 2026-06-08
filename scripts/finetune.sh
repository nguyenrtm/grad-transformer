#!/bin/bash

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes=1 --main_process_port 30000 src/finetune_aquarat_qwen.py \
--num_samples_per_val_dataset 100 \
--dataset_path deepmind/aqua_rat \
--split 0.5 \
--model_path Qwen/Qwen2.5-3B-Instruct \
--num_training_steps 20000 \
--gradient_accumulation_steps 16 \
--lr 4e-5 \
--lora_rank 2 \
--batch_size 1 \
--weight_decay 0.01 \
--max_grad_norm 1.0 \
--max_length 4096 \
--max_length_eval 4096 \
--max_new_tokens 4096 \
--evaluate_every 500 \
--tune_option all_attention_layers \
--scheduler \
--seed 42