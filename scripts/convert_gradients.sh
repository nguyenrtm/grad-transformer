#!/bin/bash

python src/convert_gradients.py \
--gradients_path ../grad-transformer/gradients/grad-transformer_all_attention_layers_Qwen2.5-3B-Instruct_aqua_rat \
--model_name qwen \
--merge_option by_layer \
--starting_indices_shadow_dataset 1 \
--ending_indices_shadow_dataset 15 \
--starting_indices_step 2000 \
--ending_indices_step 2200 \
--saving_option sep

python src/convert_gradients.py \
--gradients_path ../grad-transformer/gradients/grad-transformer_all_attention_layers_Qwen2.5-7B-Instruct_aqua_rat \
--model_name qwen \
--merge_option by_layer \
--starting_indices_shadow_dataset 1 \
--ending_indices_shadow_dataset 15 \
--starting_indices_step 1200 \
--ending_indices_step 1400 \
--saving_option sep