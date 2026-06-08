import os
import sys
import time
import datetime
import argparse
import random
import torch
import numpy as np
import evaluate
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from transformers import get_scheduler
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.nn.utils.rnn import pad_sequence
from trl import DataCollatorForCompletionOnlyLM
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from dataset import CustomDataset, random_sample
from datasets import Dataset
from utils import init_lora_B_gaussian, generate_and_tokenize_prompt, Prompter
from accelerate.logging import get_logger
from eval_math import *
from calflops import calculate_flops

torch.set_printoptions(threshold=float('inf'))

parser = argparse.ArgumentParser(description='Fine-tuning script for model')

# Print datetime
start_time = time.time()

parser.add_argument(
    '--model_path',
    type=str,
    required=True,
)

parser.add_argument(
    '--train_indices_path',
    nargs='+',
    type=str,
)

parser.add_argument(
    '--test_indices_path',
    nargs='+',
    type=str,
)

parser.add_argument(
    '--seed',
    type=int,
)

parser.add_argument(
    '--num_samples_per_train_dataset',
    type=int,
    default=None,
)

parser.add_argument(
    '--num_samples_per_val_dataset',
    type=int,
    default=None,
)

parser.add_argument(
    '--num_training_steps',
    type=int,
    default=10000,
)

parser.add_argument(
    '--warmup_ratio',
    type=float,
    default=0.0,
)

parser.add_argument(
    '--dataset_path',
    type=str,
    default='deepmind/aqua_rat'
)

parser.add_argument(
    '--lora_rank',
    type=int,
    required=True
)

parser.add_argument(
    '--scheduler',
    action='store_true'
)

parser.add_argument(
    '--physical_batch_size',
    type=int,
)

parser.add_argument(
    '--evaluate_every',
    type=int,
    default=1000
)

parser.add_argument(
    '--max_length',
    type=int,
    default=1024
)

parser.add_argument(
    '--max_length_eval',
    type=int,
    default=8192
)

parser.add_argument(
    '--max_new_tokens',
    type=int,
    default=2048
)

parser.add_argument(
    '--lr',
    type=float,
    default=1e-5,
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.01
)

parser.add_argument(
    '--not_quantized',
    action='store_true'
)

parser.add_argument(
    '--lora_lm_head',
    action='store_true'
)

parser.add_argument(
    '--tune_option',
    type=str,
    default=None,
)

parser.add_argument(
    '--logical_batch_size',
    type=int,
    default=1
)

parser.add_argument(
    '--save_fname',
    type=str,
    default=None
)

parser.add_argument(
    '--early_stopping_patience',
    type=int,
    default=None,
)

parser.add_argument(
    '--early_stopping_min_delta',
    type=float,
    default=0.0,
)

parser.add_argument(
    '--split',
    type=float,
    default=None
)

parser.add_argument(
    '--dp', 
    action='store_true'
)

parser.add_argument(
    '--max_grad_norm', 
    type=float, 
    default=1.0
)

parser.add_argument(
    '--noise_multiplier', 
    type=float, 
    default=0.0
)

parser.add_argument(
    '--dp_epsilon', 
    type=float,
    default=None
)

parser.add_argument(
    '--target_delta', 
    type=float, 
    default=None
)

# Parse arguments and setup name of output file with forgetting stats
args = parser.parse_args()

# Set random seed for initialization
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(seed)

safe_model_path = args.model_path.split('/')[1]
safe_dataset_path = args.dataset_path.split('/')[1]

args_dict = vars(args)
if args.save_fname:
    save_fname = args.save_fname
else:
    if args.split:
        save_fname = f"{safe_model_path}_{safe_dataset_path}_{args.tune_option}_split_{args.split}_seed_{args.seed}_C_{args.max_grad_norm}_eps_{args.dp_epsilon}_split_{args.split}"
    elif args.train_indices_path:
        path = [x.split('/')[-1][-5] for x in args.train_indices_path]
        prefix = args.train_indices_path[0].split('/')[-1].replace('.pkl', '')
        save_fname = safe_model_path + '-' + prefix + '-' + '-'.join(path) + f'seed_{args.seed}_C_{args.max_grad_norm}_eps_{args.dp_epsilon}'
    else:
        save_fname = f"{safe_model_path}_{safe_dataset_path}_{args.tune_option}_seed_{args.seed}_C_{args.max_grad_norm}_eps_{args.dp_epsilon}"
    
print(f"Saving to {save_fname}")

privacy_engine = None
accelerator = Accelerator(
    log_with="wandb", 
    mixed_precision="bf16",
)

device = accelerator.device
logger = get_logger(__name__)
logger.setLevel(logging.INFO)

accelerator.init_trackers("seckt", 
    config=args_dict,
    init_kwargs={
        "wandb": {
            "name": save_fname,
        }
    }
)

# Print the arguments
print("Parameter configuration:")
for k, v in args_dict.items():
    print(f"{k}: {v}")

#-----------------------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')

lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

set_seed(seed)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("The following weights are tuned:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)
        print(name, param[0])
        break

model.to(device)

dataset = CustomDataset(path=args.dataset_path)
train_dataset, val_dataset, test_dataset = dataset.split_dataset()

if args.train_indices_path:
    import pickle
    train_idx = []
    test_idx = []

    train_dataset_tmp = train_dataset

    for indices in args.train_indices_path:
        with open(indices, 'rb') as file:
            idx = pickle.load(file)
        train_idx += idx['train']
        print(len(train_idx))
    
    for indices in args.test_indices_path:
        with open(indices, 'rb') as file:
            idx = pickle.load(file)
        test_idx += idx['test']
    
    train_dataset = train_dataset_tmp.select(train_idx)
    val_dataset = train_dataset_tmp.select(test_idx)

if args.split:
    train_dataset = train_dataset.select(range(int(len(train_dataset)*args.split), len(train_dataset)))
    print(f"Cut the first {args.split*100}% of the training set")

if args.num_samples_per_train_dataset:
    train_dataset = random_sample(train_dataset, n_samples=args.num_samples_per_train_dataset)

if val_dataset == None:
    val_dataset = test_dataset

if args.num_samples_per_val_dataset:
    val_dataset = random_sample(val_dataset, n_samples=args.num_samples_per_val_dataset)

print(f"Final length train set: {len(train_dataset)}, validation set: {len(val_dataset)}")

train_dataset = train_dataset.map(dataset.format_samsum, batched=True, load_from_cache_file=False)
val_dataset = val_dataset.map(dataset.format_samsum, batched=True, load_from_cache_file=False)

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

tokenized_val_dataset_for_loss = val_dataset.map(
    dataset.preprocess_function_causal,
    batched=True,
    remove_columns=train_dataset.column_names,
    load_from_cache_file=False,
    fn_kwargs={
        'prefix': "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
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

tokenized_val_dataset = val_dataset.map(
    dataset.preprocess_function_causal,
    batched=True,
    remove_columns=val_dataset.column_names,
    load_from_cache_file=False,
    fn_kwargs={
        'prefix': "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        'postfix': "<|im_end|>\n<|im_start|>assistant\n",
        'max_length': args.max_length_eval,
        'padding': True,
        'truncation': True,
        'tokenizer': tokenizer,
        'input_key': 'instruction',
        'target_key': 'output',
        'mode': 'inference'
    },
)

#-----------------------------------------------------------------------------------------
data_collator = DataCollatorForCompletionOnlyLM(
    response_template='<|im_start|>assistant', 
    tokenizer=tokenizer
)

rouge = evaluate.load("rouge", keep_in_memory=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    records_table = wandb.Table(columns=["labels", "prediction"])
    for label, pred in zip(decoded_labels, decoded_preds):
        records_table.add_data(label, pred)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return [{k: round(v, 4).item() for k, v in result.items()}, records_table]

def evaluate_model(model, val_dataloader):
    generated_sequences = []
    label_sequences = []
    for eval_batch in tqdm(val_dataloader):
        input_ids = eval_batch["input_ids"].to(device)
        attention_mask = eval_batch["attention_mask"].to(device)
        label_ids = eval_batch["labels"].to(device)

        generated_tokens = model._module.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
            use_cache=True,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True
        )
        
        generated_tokens = generated_tokens.sequences[:, input_ids.shape[1]:]
        generated_sequences.extend(generated_tokens.cpu())
        label_sequences.extend(label_ids.cpu())

    predictions = pad_sequence(generated_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(label_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)

    result, records_table = compute_metrics((predictions, labels))
    return result, records_table

def evaluate_loss(model, val_dataloader):
    model.eval()
    losses = []
    for eval_batch in val_dataloader:
        with torch.no_grad():
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            outputs = model(**eval_batch)
            loss = outputs.loss
        losses.append(loss.item())
    return np.mean(losses)

LOGICAL_BATCH_SIZE = args.logical_batch_size * args.physical_batch_size
MAX_PHYSICAL_BATCH_SIZE = args.physical_batch_size

train_dataloader = DataLoader(tokenized_train_dataset, batch_size=LOGICAL_BATCH_SIZE, shuffle=True, collate_fn=data_collator)
val_dataloader = DataLoader(tokenized_val_dataset, batch_size=8, shuffle=False, collate_fn=default_data_collator)
val_dataloader_for_loss = DataLoader(tokenized_val_dataset_for_loss, batch_size=8, shuffle=False, collate_fn=data_collator)

optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

num_training_steps = args.num_training_steps

if args.dp:
    # Instantiate PrivacyEngine and make model + optimizer + train_dataloader private
    privacy_engine = PrivacyEngine()

    # Opacus will replace the sampler in `train_dataloader` with a DP-compatible one
    if args.dp_epsilon:
        model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            epochs=num_training_steps / len(train_dataset),
            target_epsilon=args.dp_epsilon,
            target_delta=args.target_delta,
            max_grad_norm=args.max_grad_norm,
        )
    elif args.noise_multiplier:
        model, optimizer, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            target_delta=args.target_delta,
        )

if args.scheduler:
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_training_steps*args.warmup_ratio,
        num_training_steps=args.num_training_steps,
    )

model.to(device)
model.train()

progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

step = 0
start = time.time()

initial_lora_params = {
    name: param.clone().detach().cpu()
    for name, param in model._module.named_parameters()
    if param.requires_grad
}

initial_save_path = os.path.join(f"models/{safe_dataset_path}/dp-sgd/{save_fname}", "lora_initial.pt")
os.makedirs(os.path.dirname(initial_save_path), exist_ok=True)
torch.save(initial_lora_params, initial_save_path)

best_metric = -1
best_eval_loss = float('inf')

if accelerator.is_main_process:
    with torch.no_grad():
        eval_loss = evaluate_loss(model, val_dataloader_for_loss)
        best_eval_loss = eval_loss
        result, records_table = evaluate_model(model, val_dataloader)
        print(f"Step 0 Eval loss: {eval_loss:.6f}")
        print(f"Step 0 samsum: {result}")
        accelerator.log({f"samsum_{k}": v for k, v in result.items()}, step=0)
        accelerator.log({"eval_loss": eval_loss}, step=0)
        accelerator.log({f"samsum_records_table": records_table}, step=0)
        best_metric = result['rouge1']
        accelerator.log({"best_metric": best_metric}, step=step)

accelerator.wait_for_everyone()

print('TRAINING STARTED...')

# Early stopping state (eval loss)
best_eval_loss = float('inf')
early_stop_counter = 0
stop_training = False

while step <= num_training_steps:
    avg_train_loss = 0

    with BatchMemoryManager(
        data_loader=train_dataloader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer,
    ) as memory_safe_data_loader:
        for batch in memory_safe_data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            start_time = time.time()
            progress_bar.update(1)
            step += 1
            model.train()
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
                
            optimizer.step()
            if args.scheduler:
                scheduler.step()
            avg_train_loss += loss.item()

            if args.scheduler:
                accelerator.log({"learning_rate": scheduler.get_last_lr()[0]}, step=step)
        
            if step % 100 == 0 and step != 0:
                avg_train_loss /= 100
                accelerator.log({"train_loss": avg_train_loss}, step=step)
                avg_train_loss = 0
                eval_loss = evaluate_loss(model, val_dataloader_for_loss)
                print(f"Step {step} Eval loss: {eval_loss:.6f}")
                accelerator.log({"eval_loss": eval_loss}, step=step)

                # Log current (ε, δ) if DP is enabled
                if args.dp and privacy_engine is not None and args.target_delta is not None:
                    epsilon = privacy_engine.accountant.get_epsilon(delta=args.target_delta)
                    print(f"Step {step}: (ε, δ)=({epsilon:.4f}, {args.target_delta})")
                    accelerator.log({"epsilon": epsilon}, step=step)

                # Early stopping on eval loss
                if args.early_stopping_patience is not None:
                    improved = (best_eval_loss - eval_loss) > args.early_stopping_min_delta
                    if improved:
                        best_eval_loss = eval_loss
                        early_stop_counter = 0
                        accelerator.log({"best_eval_loss": best_eval_loss}, step=step)
                    else:
                        early_stop_counter += 100
                        accelerator.log({"early_stop_counter": early_stop_counter}, step=step)
                        if early_stop_counter >= args.early_stopping_patience:
                            print(
                                f"Early stopping triggered at step {step}: "
                                f"no eval loss improvement for {args.early_stopping_patience} checks."
                            )
                            stop_training = True

            if step >= num_training_steps or stop_training and step != 0:
                stop_training = True
                if accelerator.is_main_process:
                    model.eval()
                    with torch.no_grad():
                        result, records_table = evaluate_model(model, val_dataloader)
                        print(f"Step {step} samsum: {result}")
                        accelerator.log({f"samsum_{k}": v for k, v in result.items()}, step=step)
                        accelerator.log({f"samsum_records_table": records_table}, step=step)

                        unwrapped_model = model._module
                        unwrapped_model.save_pretrained(
                            f"models/{safe_dataset_path}/dp-sgd/{save_fname}",
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )

                        lora_grads = {
                            name: param.clone().detach().cpu()
                            for name, param in model._module.named_parameters()
                            if param.requires_grad
                        }

                        delta_lora_grads = {
                            name: (lora_grads[name] - initial_lora_params[name])
                            for name in lora_grads.keys()
                        }

                        delta_save_path = os.path.join(f"models/{safe_dataset_path}/dp-sgd/{save_fname}", "final_lora_gradients.pt")
                        os.makedirs(os.path.dirname(delta_save_path), exist_ok=True)
                        torch.save(delta_lora_grads, delta_save_path)

                        if result['rouge1'] > best_metric:
                            best_metric = result['rouge1']
                            delta_save_path = os.path.join(f"models/{safe_dataset_path}/dp-sgd/{save_fname}", "best_lora_gradients.pt")
                            os.makedirs(os.path.dirname(delta_save_path), exist_ok=True)
                            torch.save(delta_lora_grads, delta_save_path)
                            print(f"Best model saved at step {step} to models/{safe_dataset_path}/dp-sgd/{save_fname}/best_lora_gradients.pt; {result}")
                    
                    accelerator.log({"best_metric": best_metric}, step=step)
                        
                accelerator.wait_for_everyone()
                break

    if stop_training == True:
        break

accelerator.end_training()