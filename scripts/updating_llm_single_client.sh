#!/bin/bash

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes=1 --num_processes 1 --main_process_port 14000 src/inference.py \
--dataset_path deepmind/aqua_rat \
--small_model_path Qwen/Qwen2.5-3B-Instruct \
--large_model_path Qwen/Qwen2.5-7B-Instruct \
--base_model_path google/flan-t5-large \
--save_fname test_aquarat \
--transform_model_path ../grad-transformer/models/transform_models/aqua_rat-3B-7B-flan-t5-large.pt \
--small_model_gradient_path ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct_aqua_rat_all_attention_layers_split_0.5_none_seed_42/final_lora_gradients.pt \
--max_length_eval 4096 \
--max_new_tokens 4096 \
--small_tune_option all_attention_layers \
--large_tune_option all_attention_layers \
--lora_rank 2 \
--l_out 28 \
--input_dim 12800 \
--output_dim 22528 \
--seed 42