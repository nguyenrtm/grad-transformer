import os
import re
from collections import defaultdict
import datetime
import argparse
import pickle as pkl
import torch
from tqdm import tqdm

def process_gradient_folder(gradient_folder):
    path = f"{gradient_folder}/lora_gradients.pt"
    tensors_dict = torch.load(path, weights_only=True)
        
    return tensors_dict

def flatten_and_merge(tensors_dict):
    for k, v in tensors_dict.items():
        flattened = v.flatten()
        tensors_dict[k] = flattened
    
    merged_tensor = torch.cat(list(tensors_dict.values()))
    return merged_tensor

def parse_key(k):
    layer_match = re.search(r'layers\.(\d+)', k)
    layer_idx = int(layer_match.group(1)) if layer_match else 999
    return (layer_idx, k)

def group_by_layer_and_merge(
        sorted_items, 
        model_name: str, 
        merge_option: str = 'by_layer', 
        start_layer: int = -1,
        end_layer: int = -1,
        print_layers: bool = False
    ):
    if ('t5' in model_name.lower() or 'bert' in model_name.lower() or 'qwen' in model_name.lower()) and merge_option == 'by_layer':
        if start_layer != -1 or end_layer != -1:
            sorted_items = sorted_items[start_layer*4:end_layer*4]
        
        if print_layers:
            for i in sorted_items:
                print(i[0])

        layer_lst = []
        tmp_lst = []

        for i, (k, v) in enumerate(sorted_items):
            v = torch.flatten(v)
            tmp_lst.append(v)

            if len(tmp_lst) == 4:
                layer_tensor = torch.cat(tmp_lst, dim=0)
                layer_lst.append(layer_tensor)
                tmp_lst = []

    final_tensor = torch.stack(layer_lst, dim=0)
    return final_tensor

def split_tensor_to_lora(tensor, keys, model_name: str, lora_rank=4):
    new_dict = {}

    sorted_items = sorted(keys, key=lambda x: parse_key(x[0]))
    
    if 'qwen2.5-0.5b' in model_name.lower():
        for i, k in enumerate(sorted_items):
            if 'q_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, :lora_rank*896].reshape(lora_rank, 896)
            elif 'q_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*896:lora_rank*896*2].reshape(896, lora_rank)
            if 'v_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, lora_rank*896*2:lora_rank*896*3].reshape(lora_rank, 896)
            elif 'v_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*896*3:].reshape(128, lora_rank)
    elif 'qwen2.5-1.5b' in model_name.lower():
        for i, k in enumerate(sorted_items):
            if 'q_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, :lora_rank*1536].reshape(lora_rank, 1536)
            elif 'q_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*1536:lora_rank*1536*2].reshape(1536, lora_rank)
            if 'v_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, lora_rank*1536*2:lora_rank*1536*3].reshape(lora_rank, 1536)
            elif 'v_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*1536*3:].reshape(256, lora_rank)
    elif 'qwen2.5-3b' in model_name.lower():
        for i, k in enumerate(sorted_items):
            if 'q_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, :lora_rank*2048].reshape(lora_rank, 2048)
            elif 'q_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*2048:lora_rank*2048*2].reshape(2048, lora_rank)
            if 'v_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, lora_rank*2048*2:lora_rank*2048*3].reshape(lora_rank, 2048)
            elif 'v_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*2048*3:].reshape(256, lora_rank)
    elif 'qwen2.5-7b' in model_name.lower():
        for i, k in enumerate(sorted_items):
            if 'q_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, :lora_rank*3584].reshape(lora_rank, 3584)
            elif 'q_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*3584:lora_rank*3584*2].reshape(3584, lora_rank)
            if 'v_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, lora_rank*3584*2:lora_rank*3584*3].reshape(lora_rank, 3584)
            elif 'v_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*3584*3:].reshape(512, lora_rank)
    elif 'qwen2.5-14b' in model_name.lower():
        for i, k in enumerate(sorted_items):
            if 'q_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, :lora_rank*5120].reshape(lora_rank, 5120)
            elif 'q_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*5120:lora_rank*5120*2].reshape(5120, lora_rank)
            if 'v_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, lora_rank*5120*2:lora_rank*5120*3].reshape(lora_rank, 5120)
            elif 'v_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*5120*3:].reshape(1024, lora_rank)
    elif 'qwen2.5-32b' in model_name.lower():
        for i, k in enumerate(sorted_items):
            if 'q_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, :lora_rank*5120].reshape(lora_rank, 5120)
            elif 'q_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*5120:lora_rank*5120*2].reshape(5120, lora_rank)
            if 'v_proj.lora_A' in k:
                new_dict[k] = tensor[i // 4, lora_rank*5120*2:lora_rank*5120*3].reshape(lora_rank, 5120)
            elif 'v_proj.lora_B' in k:
                new_dict[k] = tensor[i // 4, lora_rank*5120*3:].reshape(1024, lora_rank)
    return new_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert gradients for model')

    parser.add_argument(
        '--gradients_path',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--start_layer',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--end_layer',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--merge_option',
        type=str,
        required=True
    )

    parser.add_argument(
        '--starting_indices_shadow_dataset',
        type=int,
        required=True
    )

    parser.add_argument(
        '--ending_indices_shadow_dataset',
        type=int,
        required=True
    )

    parser.add_argument(
        '--starting_indices_step',
        type=int,
        required=True
    )

    parser.add_argument(
        '--ending_indices_step',
        type=int,
        required=True
    )

    parser.add_argument(
        '--saving_option',
        type=str,
        required=True
    )


    args = parser.parse_args()
    if args.saving_option == 'full':
        full_gradients = []

    for i in tqdm(range(args.starting_indices_shadow_dataset, args.ending_indices_shadow_dataset+1)):
        shadow_dataset_path = f"shadow_dataset_{i}"
        shadow_dataset_folder = os.path.join(args.gradients_path, shadow_dataset_path)
        for j in range(args.starting_indices_step+1, args.ending_indices_step+1):
            iteration_path = f"step_{j}"
            idx = (args.ending_indices_step - args.starting_indices_step) * (i - 1) + j - args.starting_indices_step
            gradient_folder = os.path.join(shadow_dataset_folder, iteration_path)
            tensors_dict = process_gradient_folder(gradient_folder)
            if args.merge_option == 'by_layer':
                if i == args.starting_indices_shadow_dataset and j == args.starting_indices_step+1:
                    sorted_items = sorted(tensors_dict.items(), key=lambda x: parse_key(x[0]))
                    merged_tensor = group_by_layer_and_merge(
                        sorted_items, 
                        args.model_name, 
                        args.merge_option,
                        args.start_layer,
                        args.end_layer,
                        print_layers=True
                    )
                else:
                    merged_tensor = group_by_layer_and_merge(
                        sorted_items, 
                        args.model_name, 
                        args.merge_option,
                        args.start_layer,
                        args.end_layer
                    )
            elif args.merge_option == 'flatten':
                merged_tensor = flatten_and_merge(tensors_dict)
            
            if args.saving_option == 'sep':
                save_path = os.path.join(args.gradients_path, f"sep/gradients_{idx}.pt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(merged_tensor, save_path)
                
            if args.saving_option == 'full':
                full_gradients.append(merged_tensor)

    if args.saving_option == 'full':
        full_gradients = torch.stack(full_gradients, dim=0)
        print(f"Final exported shape: {full_gradients.shape}")
        save_path = os.path.join(args.gradients_path, f'merged/gradients_{args.start_layer}_{args.end_layer}_{args.merge_option}_{args.starting_indices_shadow_dataset}_{args.ending_indices_shadow_dataset}_{args.starting_indices_step}_{args.ending_indices_step}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(full_gradients, save_path)