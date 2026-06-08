#!/bin/bash

accelerate launch --config_file "../grad-transformer/configs/accumulate_1_gpu_1_bf16.yaml" \
--num_processes=1 --main_process_port 20000 src/transformer.py \
--num_samples_per_val_dataset 100 \
--save_path ../grad-transformer/models/transform_models/aqua_rat-3B-7B-flan-t5-large.pt \
--base_model_path google/flan-t5-large \
--small_gradients_paths ../grad-transformer/gradients/grad-transformer_all_attention_layers_Qwen2.5-3B-Instruct_aqua_rat/sep \
--large_gradients_paths ../grad-transformer/gradients/grad-transformer_all_attention_layers_Qwen2.5-7B-Instruct_aqua_rat/sep \
--split 0.95 \
--batch_size 32 \
--num_epochs 50 \
--lr 4e-5 \
--l_out 28 \
--dataset_size 3000