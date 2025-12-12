import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm  # pip install tqdm

from data.collate import TranslationCollator
from data.datasets import TranslationDataset
from model.transformer import Transformer


@dataclass
class TrainConfig:
    vocab_size: int = 10000
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    pad_id: int = 0
    sos_id: int = 1
    eos_id: int = 2
    max_len: int = 256
    batch_size: int = 32
    num_epochs: int = 1
    grad_clip: Optional[float] = 1.0
    device: str = "cpu"
    warmup_steps: int = 4000
    accum_iter: int = 1

# 1. THE SCHEDULER LOGIC
# This implements the "Noam" scheduler
def get_lr_scheduler(optimizer, d_model, warmup_steps=4000):
    def lr_lambda(step_num):
        step_num += 1  # avoid zero division
        arg1 = step_num ** -0.5
        arg2 = step_num * (warmup_steps ** -1.5)
        return (d_model ** -0.5) * min(arg1, arg2)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(
    model: Transformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],  # scheduler
    scaler: torch.cuda.amp.GradScaler,            # AMP scaler
    device: torch.device,
    grad_clip: Optional[float],
    accum_iter: int = 1,                          # gradient accumulation
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(pbar):
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)
        src_pad_mask = batch["src_pad_mask"].to(device)
        tgt_pad_mask = batch["tgt_pad_mask"].to(device)
        tgt_causal_mask = batch["tgt_causal_mask"].to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(
                src=src,
                tgt_input=tgt_input,
                src_pad_mask=src_pad_mask,
                tgt_pad_mask=tgt_pad_mask,
                tgt_causal_mask=tgt_causal_mask,
            )
            
            # Reshape for loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                tgt_output.view(-1),
            )
            
            # Normalize loss for gradient accumulation
            loss = loss / accum_iter

        scaler.scale(loss).backward()

        if (i + 1) % accum_iter == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            
            # Clip Gradients
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer Step (with scaler)
            scaler.step(optimizer)
            scaler.update()
            
            # Scheduler Step (Update LR every batch, not every epoch)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # --- Logging ---
        # Calculate actual loss for reporting (undoing the accumulation division)
        actual_loss = loss.item() * accum_iter
        
        # We assume ignore_index is handled by criterion, so we count non-pad tokens
        ignore_index = criterion.ignore_index if hasattr(criterion, "ignore_index") else None
        non_pad_count = (tgt_output != ignore_index).sum().item() if ignore_index is not None else tgt_output.numel()
        
        total_loss += actual_loss * non_pad_count
        total_tokens += non_pad_count
        
        # Update progress bar description with current perplexity/loss
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=f"{actual_loss:.4f}", lr=f"{current_lr:.6f}")

    return total_loss / max(total_tokens, 1)

def train(
    src_sequences: Iterable[Iterable[int]],
    tgt_sequences: Iterable[Iterable[int]],
    cfg: TrainConfig,
) -> Transformer:
    device = torch.device(cfg.device)

    # 1. Dataset & Loader (Increase num_workers for speed)
    dataset = TranslationDataset(src_sequences, tgt_sequences)
    collator = TranslationCollator(
        pad_id=cfg.pad_id, sos_id=cfg.sos_id, eos_id=cfg.eos_id, batch_first=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True if cfg.device == "cuda" else False,
    )

    model = Transformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pad_id=cfg.pad_id,
        max_len=cfg.max_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.pad_id, 
        label_smoothing=0.1 # <--- Helps generalization
    )

    optimizer = AdamW(
        model.parameters(), 
        lr=1.0, # Base LR is 1.0 because scheduler handles the scaling
        betas=(0.9, 0.98), 
        eps=1e-9, 
        weight_decay=1e-2
    )
    
    scheduler = get_lr_scheduler(optimizer, cfg.d_model, warmup_steps=cfg.warmup_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))

    print(f"Training on {device} with {sum(p.numel() for p in model.parameters())} parameters.")

    for epoch in range(cfg.num_epochs):
        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            grad_clip=cfg.grad_clip,
            accum_iter=cfg.accum_iter,
        )
        print(f"Epoch {epoch+1}/{cfg.num_epochs} | Loss: {avg_loss:.4f} | PPL: {math.exp(avg_loss):.2f}")
        
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")

    return model
