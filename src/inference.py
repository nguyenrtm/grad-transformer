import argparse
import random
import torch
import numpy as np
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import wandb
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.nn.utils.rnn import pad_sequence
from datasets import concatenate_datasets
from dataset import CustomDataset, random_sample
from safetensors import safe_open
import copy
from accelerate.logging import get_logger
import logging
from datasets import Dataset

from src.transformer import Embedding2EmbeddingT5
from convert_gradients import group_by_layer_and_merge, split_tensor_to_lora, parse_key
from eval_math import *
from utils import Prompter, generate_and_tokenize_prompt

def init_weights(peft_model, lora_weights, device="cuda"):
    for name, param in peft_model.named_parameters():
        if param.requires_grad and name in lora_weights.keys():
            weight = lora_weights[name].to(device)
            print(weight[0], param.data[0])
            param.data = weight.to(param.device).to(param.dtype)
    return peft_model

def apply_gradients(peft_model, lora_gradients, device="cuda"):
    for name, param in peft_model.named_parameters():
        if param.requires_grad and name in lora_gradients:
            grad = lora_gradients[name].to(device)
            if 'layers.27.self_attn.q_proj.lora_A' in name:
                if len(lora_gradients.keys()) == 144:
                    print("Applied to small")
                else:
                    print("Applied to large")
                print(grad)
            param.data = param.data + grad.to(param.device).to(param.dtype)
    return peft_model

def dp_noise_gradients(lora_gradients, noise_option, clipping_threshold=2.5, delta=1e-5, eps=10., device="cuda"):
    tmp = []
    print('=' * 20)
    for k, v in lora_gradients.items():
        tmp.append(v.view(-1))
    all_grads = torch.cat(tmp)
    if noise_option == 'gaussian':
        grad_norm = torch.norm(all_grads, p=2)
    elif noise_option == 'laplace':
        grad_norm = torch.norm(all_grads, p=1)
    clipping_scale = max(1.0, (grad_norm.item() / clipping_threshold))
    new_norm = grad_norm.item() / clipping_scale
    for k in lora_gradients.keys():
        if 'layers.27.self_attn.q_proj.lora_A' in k:
            print("Before clipping:")
            print(lora_gradients[k][0])
        lora_gradients[k] = lora_gradients[k] / clipping_scale

    print(f"Gradient norm before clipping   : {grad_norm.item()}")
    print(f"Clipping gradient with threshold: {clipping_threshold}")
    print(f"Clipping gradient with scale    : {clipping_scale}")
    print(f"Norm after clipping             : {grad_norm.item() / clipping_scale}")

    if noise_option == 'gaussian':
        noise_std = clipping_threshold * torch.sqrt(torch.log(torch.tensor(1.25 / delta)) * 2.0) / eps
        print(f"Adding Gaussian noise N(0, {noise_std ** 2})")
        print(f"Variance of noise added: {noise_std ** 2}")
        print(f"Providing DP with (eps, delta): ({eps}, {delta})")

        for k in lora_gradients.keys():
            grad = lora_gradients[k].to(device)
            noise = torch.normal(0, noise_std, size=grad.size()).to(device)
            lora_gradients[k] = (grad + noise).to(grad.device).to(grad.dtype)
            if 'layers.27.self_attn.q_proj.lora_A' in k:
                print("After adding noise:")
                print(lora_gradients[k][0])
    elif noise_option == 'laplace':
        noise_beta = clipping_threshold / eps
        print(f"Adding Laplace noise L(0, {noise_beta})")
        print(f"Variance of noise added: {2 * (noise_beta ** 2)}")
        print(f"Providing DP with eps: {eps}")

        for k in lora_gradients.keys():
            grad = lora_gradients[k].to(device)
            noise = torch.distributions.Laplace(0, noise_beta).sample(grad.size()).to(device)
            lora_gradients[k] = (grad + noise).to(grad.device).to(grad.dtype)
            if 'layers.27.self_attn.q_proj.lora_A' in k:
                print("After adding noise:")
                print(lora_gradients[k][0])
    
    print('=' * 20)

    return lora_gradients

def noise_gradients(lora_gradients, noise_option, mean, noise_std, device="cuda"):
    print('=' * 20)
    if noise_option == 'nondp_gaussian':
        print(f"Adding Gaussian noise N(0, {noise_std ** 2})")
        print(f"Noise std: {noise_std}")
        for k in lora_gradients.keys():
            if 'layers.27.self_attn.q_proj.lora_A' in k:
                print("Before adding noise:")
                print(lora_gradients[k][0])
            grad = lora_gradients[k].to(device)
            noise = torch.normal(mean, noise_std, size=grad.size()).to(device)
            lora_gradients[k] = (grad + noise).to(grad.device).to(grad.dtype)
            if 'layers.27.self_attn.q_proj.lora_A' in k:
                print("After adding noise:")
                print(lora_gradients[k][0])
    print('=' * 20)

    return lora_gradients

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--save_fname',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--small_model_path',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--small_model_gradient_path',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--large_model_path',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--transform_model_path',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--seed',
        type=int,
        required=True
    )

    parser.add_argument(
        '--lora_rank',
        type=int,
        default=4,
    )

    parser.add_argument(
        '--num_samples_per_val_dataset',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--max_length_eval',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--init_small_path',
        type=str
    )

    parser.add_argument(
        '--init_large_path',
        type=str
    )

    parser.add_argument(
        '--input_dim',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--output_dim',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--l_out',
        type=int,
        default=None,
    )

    parser.add_argument(
        '--small_tune_option',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--large_tune_option',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--base_model_path',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--ans_template',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--ref_template',
        type=str,
        default=None
    )

    parser.add_argument(
        '--delta',
        type=float,
        default=1e-5
    )

    parser.add_argument(
        '--dp_eps',
        type=float
    )

    parser.add_argument(
        '--clipping_threshold',
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--mean',
        type=float,
    )

    parser.add_argument(
        '--noise_std',
        type=float,
    )

    parser.add_argument(
        '--noise_option',
        type=str
    )

    parser.add_argument(
        '--quantize',
        action='store_true'
    )

    parser.add_argument(
        '--random_gradients_scale',
        type=float,
        default=None
    )

    args = parser.parse_args()

    accelerator = Accelerator(
        log_with="wandb", 
        mixed_precision="bf16",
    )

    device = accelerator.device
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)

    # Print the arguments
    args_dict = vars(args)

    # Set random seed for initialization
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

    accelerator.init_trackers("evaluate-rvkd", 
        config=args_dict,
        init_kwargs={
            "wandb": {
                "name": args.save_fname,
            }
        }
    )

    rouge = evaluate.load("rouge", keep_in_memory=True)

    def compute_metrics(
            eval_pred, 
            dataset_path, 
            ans_template='####', 
            ref_template='####'
        ):
        if 'dialogsum' in dataset_path.lower() or 'samsum' in dataset_path.lower():
            predictions, labels = eval_pred
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            print(f"*****\n{decoded_preds[0]}\n====\n{decoded_labels[0]}\n*****")
            print(f"*****\n{decoded_preds[1]}\n====\n{decoded_labels[1]}\n*****")  
            print(f"*****\n{decoded_preds[2]}\n====\n{decoded_labels[2]}\n*****")          
            result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
            result["gen_len"] = np.mean(prediction_lens)
        else:
            predictions, labels = eval_pred
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            acc, records_table = evaluate_math_reasoning_accuracy(
                predictions=decoded_preds, 
                references=decoded_labels,
                ans_template=ans_template,
                ref_template=ref_template,
                table=False
            )
            result = {"accuracy": round(acc * 100, 2)}
        
        records_table = None
        
        return result, records_table

    def evaluate_model(model, val_dataloader, dataset_path, ans_template, ref_template):
        generated_sequences = []
        label_sequences = []
        for eval_batch in tqdm(val_dataloader):
            input_ids = eval_batch["input_ids"].to(device)
            attention_mask = eval_batch["attention_mask"].to(device)
            label_ids = eval_batch["labels"].to(device)

            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
                use_cache=True,
            )
            
            generated_tokens = generated_tokens[:, input_ids.shape[1]:]
            generated_sequences.extend(generated_tokens.cpu())
            label_sequences.extend(label_ids.cpu())

        predictions = pad_sequence(generated_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(label_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)

        result, records_table = compute_metrics((predictions, labels), dataset_path, ans_template, ref_template)
        return result, records_table

    if args.quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        large_model = AutoModelForCausalLM.from_pretrained(
            args.large_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )

        # Big memory saver for training
        large_model.config.use_cache = False
        large_model.gradient_checkpointing_enable()

        # Required for stable k-bit finetuning
        large_model = prepare_model_for_kbit_training(large_model)
    else:
        large_model = AutoModelForCausalLM.from_pretrained(
            args.large_model_path,
            torch_dtype=torch.bfloat16,
        )

    large_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    set_seed(seed)
    large_model = get_peft_model(large_model, large_lora_config)
    large_model.to(device)

    transform_model = Embedding2EmbeddingT5(input_dim=args.input_dim, output_dim=args.output_dim, base_model=args.base_model_path)
    transform_model.load_state_dict(torch.load(args.transform_model_path, weights_only=True))
    transform_model.to(device)
    transform_model.to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.large_model_path, padding_side='left')

    dataset = CustomDataset(path=args.dataset_path)
    train_dataset, val_dataset, test_dataset = dataset.split_dataset()

    if val_dataset == None:
        val_dataset = test_dataset

    if args.num_samples_per_val_dataset:
        val_dataset = random_sample(val_dataset, args.num_samples_per_val_dataset)
    
    if 'aqua_rat' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_aquarat, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'gsm8k' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_gsm8k, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'math_qa' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_mathqa, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'svamp' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_svamp, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'hotpot_qa' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_hotpotqa, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'commonsenseqa' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_commonsenseqa, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'drop' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_drop, batched=True, load_from_cache_file=False, fn_kwargs={'mode': 'inference'})
    elif 'dialogsum' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_dialogsum, batched=True, load_from_cache_file=False)
    elif 'samsum' in args.dataset_path.lower():
        val_dataset = val_dataset.map(dataset.format_samsum, batched=True, load_from_cache_file=False)

    prompter = Prompter()
    val_dataset = Dataset.from_dict(generate_and_tokenize_prompt(val_dataset, prompter, tokenizer))
    tokenized_val_dataset = val_dataset.map(
        dataset.preprocess_function_causal,
        batched=True,
        remove_columns=val_dataset.column_names,
        load_from_cache_file=False,
        fn_kwargs={
            'prefix': "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
            'postfix': "<|im_end|>\n<|im_start|>assistant\n",
            'input_key': 'instruction',
            'target_key': 'output',
            'max_length': args.max_length_eval,
            'padding': True,
            'truncation': True,
            'tokenizer': tokenizer,
            'mode': 'inference'
        },
    )

    val_dataloader = DataLoader(tokenized_val_dataset, batch_size=8, collate_fn=default_data_collator)

    small_lora_gradients = torch.load(args.small_model_gradient_path, weights_only=True)

    if args.random_gradients_scale:
        print("Using random gradients for testing")
        for k in small_lora_gradients.keys():
            small_lora_gradients[k] = torch.randn_like(small_lora_gradients[k]) * args.random_gradients_scale

    large_lora_keys = [name for name, param in large_model.named_parameters() if param.requires_grad]

    small_gradients = group_by_layer_and_merge(sorted(small_lora_gradients.items(), key=lambda x: parse_key(x[0])), model_name=args.small_model_path, print_layers=True)
    small_gradients = small_gradients.to(device).to(torch.bfloat16)
    small_gradients = torch.unsqueeze(small_gradients, dim=0)

    large_gradients = transform_model(small_gradients, use_teacher_forcing=False, L_out=args.l_out)
    large_gradients = torch.squeeze(large_gradients, dim=0)
    predicted_large_lora_gradients = split_tensor_to_lora(tensor=large_gradients, keys=large_lora_keys, model_name=args.large_model_path, lora_rank=args.lora_rank)

    with torch.no_grad():
        large_model.eval()
        predicted_large_model = apply_gradients(copy.deepcopy(large_model), predicted_large_lora_gradients)
        result = evaluate_model(predicted_large_model, val_dataloader, args.dataset_path, args.ans_template, args.ref_template)
        print(f"Predicted large model: {result}")