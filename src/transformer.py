import os
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5GemmaForConditionalGeneration
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import Dataset
import wandb
import numpy as np
import random
import logging
from calflops import calculate_flops


class Embedding2EmbeddingT5(nn.Module):
    def __init__(self, input_dim, output_dim, base_model='google/flan-t5-small', freeze_t5=False):
        super().__init__()
        if 'flan-t5' in base_model:
            self.t5 = T5ForConditionalGeneration.from_pretrained(base_model)
            self.d_model = self.t5.config.d_model  # T5 hidden size
        elif 'gemma' in base_model:
            self.t5 = T5GemmaForConditionalGeneration.from_pretrained(base_model)
            self.d_model = self.t5.config.hidden_size  # T5 hidden size

        # New layers
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.output_proj = nn.Linear(self.d_model, output_dim)
        self.embed_proj = nn.Linear(output_dim, self.d_model)

        # Initialize new layers
        self._init_weights()

        # Optionally freeze T5 weights
        if freeze_t5:
            for param in self.t5.parameters():
                param.requires_grad = False

    def _init_weights(self):
        # Xavier uniform initialization for all new Linear layers
        for layer in [self.input_proj, self.output_proj, self.embed_proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x, y=None, L_out=1, use_teacher_forcing=False):
        """
        Args:
            x: [B, L_in, input_dim] input embedding sequence
            y: [B, L_out, output_dim] target output embeddings
            use_teacher_forcing: whether to feed ground-truth decoder inputs
        Returns:
            predicted: [B, L_out, output_dim]
        """
        B, L_in, _ = x.shape

        # Project encoder inputs
        encoder_input = self.input_proj(x)
        encoder_attention_mask = torch.ones((B, L_in), dtype=torch.long, device=x.device)

        if use_teacher_forcing:
            assert y is not None, "y must be provided during teacher forcing"
            # Shift y right and embed it to T5 space
            decoder_inputs = torch.zeros_like(y)
            decoder_inputs[:, 1:] = y[:, :-1]  # shift right
            decoder_inputs_embeds = self.embed_proj(decoder_inputs)

            # Training path
            out = self.t5(
                inputs_embeds=encoder_input,
                decoder_inputs_embeds=decoder_inputs_embeds,
                attention_mask=encoder_attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            decoder_hidden = out.decoder_hidden_states[-1][:, :L_out, :]
            predicted = self.output_proj(decoder_hidden)
            return predicted
        else:
            # Inference (autoregressive decoding)
            decoder_inputs_embeds = torch.zeros((B, 1, self.d_model), device=x.device)
            outputs = []
            for t in range(L_out):
                out = self.t5(
                    inputs_embeds=encoder_input,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    attention_mask=encoder_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = out.decoder_hidden_states[-1][:, -1:, :]  # [B, 1, d_model]
                out_embed = self.output_proj(last_hidden)  # [B, 1, output_dim]
                outputs.append(out_embed)

                next_embed = self.embed_proj(out_embed)
                decoder_inputs_embeds = torch.cat([decoder_inputs_embeds, next_embed], dim=1)

            return torch.cat(outputs, dim=1)

class CustomGradientDataset(Dataset):
    def __init__(self, dataset_size, small_gradients_paths, large_gradients_paths, indices):
        self.dataset_size = dataset_size
        self.small_gradients_paths = small_gradients_paths
        self.large_gradients_paths = large_gradients_paths
        self.indices = indices
        self.size_per_task = args.dataset_size // len(small_gradients_paths)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        idx = self.indices[idx]
        try:
            idx_of_task = (idx // self.size_per_task).item()
            idx = idx % self.size_per_task
            x = torch.load(f"{self.small_gradients_paths[idx_of_task]}/gradients_{idx+1}.pt", weights_only=True)
            y = torch.load(f"{self.large_gradients_paths[idx_of_task]}/gradients_{idx+1}.pt", weights_only=True)
            return x, y
        except:
            print(f"Corrupted file: {self.small_gradients_paths[idx_of_task]}/gradients_{idx+1}.pt")

def custom_collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    x_batch = torch.stack(x_batch, dim=0).to(torch.bfloat16)
    y_batch = torch.stack(y_batch, dim=0).to(torch.bfloat16)

    return x_batch, y_batch

if __name__ == "__main__":
    import time
    import datetime
    import argparse
    from torch.utils.data import TensorDataset, DataLoader
    import torch.optim as optim
    from tqdm import tqdm

    print("Experiment started:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "EST")
    start_time = time.time()

    # Set random seed for initialization

    parser = argparse.ArgumentParser(description='Training transformer')

    parser.add_argument(
        '--freeze',
        action='store_false'
    )

    parser.add_argument(
        '--base_model_path',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--save_fname',
        type=str,
    )

    parser.add_argument(
        '--num_samples_per_val_dataset',
        type=int,
        default=None
    )

    parser.add_argument(
        '--small_gradients_paths',
        nargs='+',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--large_gradients_paths',
        nargs='+',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--split',
        type=float,
        required=True,
    )

    parser.add_argument(
        '--clip',
        type=float,
        default=None
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        required=True
    )

    parser.add_argument(
        '--save_path',
        type=str,
        required=True
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=None
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4
    )

    parser.add_argument(
        '--l_out',
        type=int,
        default=1
    )

    parser.add_argument(
        '--freeze_t5',
        action='store_true'
    )

    parser.add_argument(
        '--dataset_size',
        type=int
    )

    parser.add_argument(
        '--eval_per',
        type=int,
        default=None
    )

    args = parser.parse_args()
    args_dict = vars(args)

    for k, v in args_dict.items():
        print(f"{k}: {v}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision="bf16",
    )

    device = accelerator.device
    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)

    if args.save_fname:
        accelerator.init_trackers("embedding2embedding", 
            config=args_dict,
            init_kwargs={
                "wandb": {
                    "name": args.save_fname,
                }
            }
        )
    else:
        accelerator.init_trackers("embedding2embedding", config=args_dict)

    dummy_small_sample = torch.load(f"{args.small_gradients_paths[0]}/gradients_1.pt", weights_only=True)
    dummy_large_sample = torch.load(f"{args.large_gradients_paths[0]}/gradients_1.pt", weights_only=True)
    
    model = Embedding2EmbeddingT5(
        input_dim=dummy_small_sample.shape[-1], 
        output_dim=dummy_large_sample.shape[-1], 
        freeze_t5=args.freeze_t5,
        base_model=args.base_model_path,
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    dataset_size = args.dataset_size
    train_dataset_size = int(args.dataset_size * args.split)

    # Use reproducible shuffling with seed for train/val split
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(dataset_size, generator=generator)

    if args.split != 1.0:
        train_idx = perm[:train_dataset_size]
        val_idx = perm[train_dataset_size:]
        if args.num_samples_per_val_dataset:
            val_idx = val_idx[:args.num_samples_per_val_dataset]
            print(f"Length of validation set: {len(val_idx)}")

        train_dataset = CustomGradientDataset(
            len(train_idx), 
            args.small_gradients_paths, 
            args.large_gradients_paths,
            indices=train_idx
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size, 
            collate_fn=custom_collate_fn, 
            num_workers=4,
            shuffle=True,
        )

        val_dataset = CustomGradientDataset(
            len(val_idx), 
            args.small_gradients_paths, 
            args.large_gradients_paths,
            indices=val_idx
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            collate_fn=custom_collate_fn, 
            num_workers=4,
            shuffle=False,
        )
    else:
        print("WARNING: No train/val split!")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on: {device}")

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    num_epochs = args.num_epochs

    best_val_loss = float('inf')

    if args.patience:
        patience = args.patience
        patience_counter = 0

    step = 0
    steps_per_epoch = train_dataset_size / args.batch_size
    if args.eval_per:
        eval_per = args.eval_per
    else:
        eval_per = int(steps_per_epoch) + 1
    print(f"Evaluate every {eval_per} steps")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        layer_loss = [0]*args.l_out

        for xb, yb in tqdm(train_loader):
            step += 1
            optimizer.zero_grad()
            pred = model(x=xb, y=yb, L_out=args.l_out, use_teacher_forcing=True) # (batch_size, L_out, embedding_size)
            loss = criterion(pred, yb)
            accelerator.backward(loss)
            # tflops += tflops_per_batch * 3

            if args.clip is not None and args.clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()
            total_loss += loss.item()

            if step % eval_per == 0:
                model.eval()

                val_loss_sum = torch.tensor(0.0, device=accelerator.device)
                val_count = torch.tensor(0.0, device=accelerator.device)

                layer_loss_sum = torch.zeros(args.l_out, device=accelerator.device)
                layer_count = torch.zeros(args.l_out, device=accelerator.device)

                with torch.no_grad():
                    for xb, yb in val_loader:
                        pred = model(x=xb, y=yb, L_out=args.l_out, use_teacher_forcing=False)
                        loss = criterion(pred, yb)

                        bs = xb.size(0)
                        val_loss_sum += loss * bs
                        val_count += bs

                        for layer in range(args.l_out):
                            layer_loss_sum[layer] += criterion(pred[:, layer, :], yb[:, layer, :]) * bs
                            layer_count[layer] += bs

                # Reduce across all processes
                val_loss_sum = accelerator.reduce(val_loss_sum, reduction="sum")
                val_count = accelerator.reduce(val_count, reduction="sum")
                layer_loss_sum = accelerator.reduce(layer_loss_sum, reduction="sum")
                layer_count = accelerator.reduce(layer_count, reduction="sum")

                avg_val_loss = (val_loss_sum / val_count).item()
                avg_layer_loss = (layer_loss_sum / layer_count).tolist()

                avg_train_loss = total_loss / eval_per

                if accelerator.is_main_process:
                    accelerator.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=step)
                    for i, v in enumerate(avg_layer_loss):
                        accelerator.log({f"val_loss_layer_{i+1}": v}, step=step)
                    print(f"Step [{step}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

                    if avg_val_loss < best_val_loss:
                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model.state_dict(), args.save_path)
                        best_val_loss = avg_val_loss
                        if args.patience:
                            patience_counter = 0
                    else:
                        if args.patience:
                            patience_counter += eval_per

                total_loss = 0
                accelerator.wait_for_everyone()
                model.train()

            if args.patience:
                if patience_counter >= patience * eval_per:
                    print(f"Early stopping triggered at step {step+1}")
                    break
            
        
        # accelerator.log({f"tflops": tflops}, step=step+1)
        accelerator.log({f"Epoch": (step - 1) / steps_per_epoch}, step=step+1)
        
        if args.patience:
            if patience_counter >= patience * eval_per:
                print(f"Early stopping triggered at step {step+1}")
                break
                    
    accelerator.end_training()