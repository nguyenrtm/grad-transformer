#!/bin/bash

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes 1 --main_process_port 20000 src/inference_mult_clients.py \
--dataset_path deepmind/aqua_rat \
--indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_0.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_1.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_2.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_3.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_4.pkl  \
--small_model_path Qwen/Qwen2.5-3B-Instruct \
--large_model_path Qwen/Qwen2.5-7B-Instruct \
--base_model_path google/flan-t5-large \
--save_fname 7B_client_0 \
--transform_model_path ../grad-transformer/models/transform_models/aqua_rat-3B-7B-flan-t5-large.pt \
--small_model_gradient_path ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_0-0/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_1-1/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_2-2/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_3-3/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_4-4/final_lora_gradients.pt \
--max_length_eval 4096 \
--max_new_tokens 512 \
--small_tune_option all_attention_layers \
--large_tune_option all_attention_layers \
--lora_rank 2 \
--l_out 48 \
--input_dim 12800 \
--output_dim 22528 \
--idx 0 \
--seed 42

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes 1 --main_process_port 20000 src/inference_mult_clients.py \
--dataset_path deepmind/aqua_rat \
--indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_0.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_1.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_2.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_3.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_4.pkl  \
--small_model_path Qwen/Qwen2.5-3B-Instruct \
--large_model_path Qwen/Qwen2.5-7B-Instruct \
--base_model_path google/flan-t5-large \
--save_fname 7B_client_1 \
--transform_model_path ../grad-transformer/models/transform_models/aqua_rat-3B-7B-flan-t5-large.pt \
--small_model_gradient_path ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_0-0/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_1-1/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_2-2/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_3-3/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_4-4/final_lora_gradients.pt \
--max_length_eval 4096 \
--max_new_tokens 512 \
--small_tune_option all_attention_layers \
--large_tune_option all_attention_layers \
--lora_rank 2 \
--l_out 48 \
--input_dim 12800 \
--output_dim 22528 \
--idx 1 \
--seed 42

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes 1 --main_process_port 20000 src/inference_mult_clients.py \
--dataset_path deepmind/aqua_rat \
--indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_0.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_1.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_2.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_3.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_4.pkl  \
--small_model_path Qwen/Qwen2.5-3B-Instruct \
--large_model_path Qwen/Qwen2.5-7B-Instruct \
--base_model_path google/flan-t5-large \
--save_fname 7B_client_2 \
--transform_model_path ../grad-transformer/models/transform_models/aqua_rat-3B-7B-flan-t5-large.pt \
--small_model_gradient_path ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_0-0/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_1-1/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_2-2/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_3-3/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_4-4/final_lora_gradients.pt \
--max_length_eval 4096 \
--max_new_tokens 512 \
--small_tune_option all_attention_layers \
--large_tune_option all_attention_layers \
--lora_rank 2 \
--l_out 48 \
--input_dim 12800 \
--output_dim 22528 \
--idx 2 \
--seed 42

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes 1 --main_process_port 20000 src/inference_mult_clients.py \
--dataset_path deepmind/aqua_rat \
--indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_0.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_1.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_2.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_3.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_4.pkl  \
--small_model_path Qwen/Qwen2.5-3B-Instruct \
--large_model_path Qwen/Qwen2.5-7B-Instruct \
--base_model_path google/flan-t5-large \
--save_fname 7B_client_3 \
--transform_model_path ../grad-transformer/models/transform_models/aqua_rat-3B-7B-flan-t5-large.pt \
--small_model_gradient_path ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_0-0/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_1-1/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_2-2/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_3-3/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_4-4/final_lora_gradients.pt \
--max_length_eval 4096 \
--max_new_tokens 512 \
--small_tune_option all_attention_layers \
--large_tune_option all_attention_layers \
--lora_rank 2 \
--l_out 48 \
--input_dim 12800 \
--output_dim 22528 \
--idx 3 \
--seed 42

accelerate launch --config_file "../grad-transformer/configs/accumulate_16_gpu_1_bf16.yaml" \
--num_processes 1 --main_process_port 20000 src/inference_mult_clients.py \
--dataset_path deepmind/aqua_rat \
--indices_path ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_0.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_1.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_2.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_3.pkl ../grad-transformer/indices/aqua_rat_10/distributed_10_split_0.9_aqua_rat_4.pkl  \
--small_model_path Qwen/Qwen2.5-3B-Instruct \
--large_model_path Qwen/Qwen2.5-7B-Instruct \
--base_model_path google/flan-t5-large \
--save_fname 7B_client_4 \
--transform_model_path ../grad-transformer/models/transform_models/aqua_rat-3B-7B-flan-t5-large.pt \
--small_model_gradient_path ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_0-0/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_1-1/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_2-2/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_3-3/final_lora_gradients.pt ../grad-transformer/models/aqua_rat/Qwen2.5-3B-Instruct-distributed_10_split_0.9_aqua_rat_4-4/final_lora_gradients.pt \
--max_length_eval 4096 \
--max_new_tokens 512 \
--small_tune_option all_attention_layers \
--large_tune_option all_attention_layers \
--lora_rank 2 \
--l_out 48 \
--input_dim 12800 \
--output_dim 22528 \
--idx 4 \
--seed 42