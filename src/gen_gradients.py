import os
import time
import datetime
import argparse
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import set_seed
import gc
import pickle as pkl
from trl import DataCollatorForCompletionOnlyLM
from tqdm import tqdm
import logging
from accelerate.logging import get_logger

from dataset import CustomDataset
from datasets import Dataset
from utils import Prompter, generate_and_tokenize_prompt, get_hms
from accelerate.logging import get_logger
from eval_math import *

parser = argparse.ArgumentParser(description='Generate gradients for model')

print("Experiment started:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "EST")
start_time = time.time()

parser.add_argument(
    '--model_path',
    type=str,
    required=True,
)

parser.add_argument(
    '--dataset_path',
    type=str,
    required=True,
)

parser.add_argument(
    '--seed',
    type=int,
    default=42,
)

parser.add_argument(
    '--indices_path',
    type=str,
    required=True,
)

parser.add_argument(
    '--starting_indices',
    type=int,
    required=True,
)

parser.add_argument(
    '--ending_indices',
    type=int,
    required=True,
)

parser.add_argument(
    '--num_training_steps',
    type=int,
    required=True,
)

parser.add_argument(
    '--converged_step',
    type=int,
    required=True,
)

parser.add_argument(
    '--mixed_precision',
    type=str,
    default="bf16"
)

parser.add_argument(
    '--lora_rank',
    type=int,
    required=True,
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=16
)

parser.add_argument(
    '--lr',
    type=float,
    default=2e-4
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.01
)

parser.add_argument(
    '--shuffle_train',
    action='store_true',
)

parser.add_argument(
    '--gradient_accumulation_steps',
    type=int,
    default=1
)

parser.add_argument(
    '--tune_option',
    type=str,
    default='all_layers'
)

parser.add_argument(
    '--max_length',
    type=int,
    required=True
)

parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.0
)

parser.add_argument(
    '--quantize',
    action='store_true',
)

parser.add_argument(
    '--save_fname',
    type=str,
    default=None
)

# Parse arguments and setup name of output file with forgetting stats
args = parser.parse_args()

safe_model_path = args.model_path.split('/')[1]
safe_dataset_path = args.dataset_path.split('/')[1]
safe_indices_path = args.indices_path.split('/')[1].split('.pkl')[0]

args_dict = vars(args)
if args.save_fname:
    save_fname = args.save_fname
else:
    save_fname = f"{safe_indices_path}_{args.tune_option}_{safe_model_path}_{safe_dataset_path}"

accelerator = Accelerator(
    log_with="wandb", 
    mixed_precision="bf16",
    gradient_accumulation_steps=args.gradient_accumulation_steps,
)

device = accelerator.device
logger = get_logger(__name__)
logger.setLevel(logging.INFO)

accelerator.init_trackers("gen_gradients", 
    config=args_dict,
    init_kwargs={
        "wandb": {
            "name": save_fname,
        }
    }
)

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(args.seed)

#-----------------------------------------------------------------------------------------
dataset = CustomDataset(path=args.dataset_path)
train_dataset, val_dataset, test_dataset = dataset.split_dataset()

if val_dataset == None:
    val_dataset = test_dataset

tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')

if 'commonsenseqa' in args.dataset_path.lower():
    train_dataset = train_dataset.map(dataset.format_commonsenseqa, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    val_dataset = val_dataset.map(dataset.format_commonsenseqa, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
elif 'drop' in args.dataset_path.lower():
    train_dataset = train_dataset.map(dataset.format_drop, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    val_dataset = val_dataset.map(dataset.format_drop, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
elif 'aqua_rat' in args.dataset_path.lower():
    train_dataset = train_dataset.map(dataset.format_aquarat, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    val_dataset = val_dataset.map(dataset.format_aquarat, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
elif 'gsm8k' in args.dataset_path.lower():
    train_dataset = train_dataset.map(dataset.format_gsm8k, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    val_dataset = val_dataset.map(dataset.format_gsm8k, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
elif 'samsum' in args.dataset_path.lower():
    train_dataset = train_dataset.map(dataset.format_samsum, batched=True, load_from_cache_file=False)
    val_dataset = val_dataset.map(dataset.format_samsum, batched=True, load_from_cache_file=False)
elif 'dialogsum' in args.dataset_path.lower():
    train_dataset = train_dataset.map(dataset.format_dialogsum, batched=True, load_from_cache_file=False)
    val_dataset = val_dataset.map(dataset.format_dialogsum, batched=True, load_from_cache_file=False)

prompter = Prompter()

train_dataset = Dataset.from_dict(generate_and_tokenize_prompt(train_dataset, prompter, tokenizer))
val_dataset = Dataset.from_dict(generate_and_tokenize_prompt(val_dataset, prompter, tokenizer))

tokenized_train_dataset = train_dataset.map(
    dataset.preprocess_function_causal,
    batched=True,
    remove_columns=train_dataset.column_names,
    load_from_cache_file=False,
    fn_kwargs={
        'prefix': '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n',
        'postfix': "<|im_end|>\n<|im_start|>assistant\n",
        'max_length': args.max_length,
        'padding': True,
        'truncation': True,
        'tokenizer': tokenizer,
        'input_key': 'instruction',
        'target_key': 'output',
        'mode': 'train',
    },
)

data_collator = DataCollatorForCompletionOnlyLM(
    response_template='<|im_start|>assistant', 
    tokenizer=tokenizer
)


#-----------------------------------------------------------------------------------------

with open(args.indices_path, 'rb') as f:
    indices_lst = pkl.load(f)

#-----------------------------------------------------------------------------------------
if args.quantize:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    # Big memory saver for training
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    # Required for stable k-bit finetuning
    base_model = prepare_model_for_kbit_training(base_model)
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )

lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

i = 0
start = time.time()
starting_indices = args.starting_indices
ending_indices = args.ending_indices

for i in range(starting_indices, ending_indices+1):
    indices = indices_lst[i-1]
    print(f"Training shadow model {i} started")
    now = time.time()
    step = 0

    if i >= starting_indices + 1:
        base_model = accelerator.unwrap_model(model).unload()
        del base_model
        del model
        del optimizer
        accelerator.free_memory()
        torch.cuda.empty_cache()
        gc.collect()

        if args.quantize:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
            )

            # Big memory saver for training
            base_model.config.use_cache = False
            base_model.gradient_checkpointing_enable()

            # Required for stable k-bit finetuning
            base_model = prepare_model_for_kbit_training(base_model)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
            )

    train_dataloader = DataLoader(tokenized_train_dataset.select(indices), batch_size=args.batch_size, shuffle=args.shuffle_train, collate_fn=data_collator)

    set_seed(args.seed)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    if accelerator.is_main_process:
        initial_lora_params = {
            name: param.clone().detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        for k, v in initial_lora_params.items():
            print(f"{k}: {v[0]}")
    
    accelerator.wait_for_everyone()

    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    progress_bar = tqdm(range(args.num_training_steps), disable=not accelerator.is_local_main_process)

    while step <= args.num_training_steps:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                start_time = time.time()
                progress_bar.update(1)
                step += 1
                model.train()
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                
                if step >= args.converged_step:
                    if accelerator.is_main_process:
                        lora_grads = {
                            name: param.clone().detach().cpu()
                            for name, param in model.module.named_parameters()
                            if param.requires_grad
                        }

                        delta_lora_grads = {
                            name: (lora_grads[name] - initial_lora_params[name])
                            for name in lora_grads.keys()
                        }

                        delta_save_path = os.path.join(f"gradients/{save_fname}/shadow_dataset_{i}/step_{step}", "lora_gradients.pt")
                        os.makedirs(os.path.dirname(delta_save_path), exist_ok=True)
                        torch.save(delta_lora_grads, delta_save_path)

                        iter_elapsed = time.time() - now
                        total_elapsed = time.time() - start

                    accelerator.wait_for_everyone()
            
                if step >= args.num_training_steps:
                    print(f"Training shadow model {i} ended; Training time: {get_hms(iter_elapsed)}; Total time: {get_hms(total_elapsed)}")
                    break

accelerator.end_training()