#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, random
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# --------------------------
# Dataset
# --------------------------

class HFContrastiveDataset(Dataset):
    """
    汎用：query / positive / negative(任意) をもつデータセットを正規化して返す。
    例: sentence-transformers/msmarco-msmarco-MiniLM-L6-v3 (subset=triplet)
        列: query, positive, negative
    """
    def __init__(self,
                 dataset_name: str,
                 split: str,
                 query_prefix: str,
                 doc_prefix: str,
                 max_negs: int = 1,
                 sample_limit: int = None,
                 subset: str = None,
                 hf_revision: str = None):
        ds_kwargs = {}
        if hf_revision:
            ds_kwargs["revision"] = hf_revision
        ds = load_dataset(dataset_name, subset, split=split, **ds_kwargs) if subset else load_dataset(dataset_name, split=split, **ds_kwargs)

        self.samples = []
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.max_negs = max_negs

        # ms marco triplet想定: query / positive / negative
        for e in ds:
            q = (e.get("query") or "").strip()
            p = (e.get("positive") or "").strip()
            n = (e.get("negative") or "")
            if not q or not p:
                continue
            negs = []
            if isinstance(n, str) and n.strip():
                negs = [n.strip()]
            elif isinstance(n, list):
                negs = [x.strip() for x in n if isinstance(x, str) and x.strip()]
            negs = negs[:max_negs]

            self.samples.append({
                "query":  self.query_prefix + q,
                "positive": self.doc_prefix + p,
                "negatives": [self.doc_prefix + x for x in negs]
            })

        if sample_limit:
            self.samples = self.samples[:sample_limit]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        # ここでは prefix 済みの文字列をそのまま返す（重複付与しない）
        return self.samples[idx]


@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_length: int
    def __call__(self, batch):
        queries = [b["query"] for b in batch]
        docs = [b["positive"] for b in batch]
        negs = [b["negatives"] for b in batch]

        q_tok = self.tokenizer(queries, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")

        flat_docs, doc_ptrs = [], []
        for n, d in zip(negs, docs):
            start = len(flat_docs)
            flat_docs.append(d)      # positive first
            for x in n:
                flat_docs.append(x)  # negatives
            doc_ptrs.append((start, 1 + len(n)))

        d_tok = self.tokenizer(flat_docs, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")

        return {
            "q_input_ids": q_tok["input_ids"],
            "q_attention_mask": q_tok["attention_mask"],
            "d_input_ids": d_tok["input_ids"],
            "d_attention_mask": d_tok["attention_mask"],
            "doc_ptrs": doc_ptrs
        }

# --------------------------
# Model
# --------------------------

class MeanPool(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-5)

class BiEncoder(nn.Module):
    def __init__(self, model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float,
                 target_modules: List[str] = None, load_in_8bit: bool=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        if load_in_8bit:
            prepare_model_for_kbit_training(self.encoder)

        if target_modules is None:
            target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"]
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, bias="none", task_type="FEATURE_EXTRACTION"
        )
        self.encoder = get_peft_model(self.encoder, lora_cfg)
        self.pool = MeanPool()

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out["last_hidden_state"] if isinstance(out, dict) else out[0]
        return self.pool(last_hidden, attention_mask)

# --------------------------
# Loss / Eval
# --------------------------

def info_nce(q, d_pos, temperature=0.05):
    q = nn.functional.normalize(q, dim=-1)
    d = nn.functional.normalize(d_pos, dim=-1)
    logits = q @ d.t() / temperature
    labels = torch.arange(q.size(0), device=q.device)
    loss = nn.functional.cross_entropy(logits, labels)
    return loss, logits

@torch.no_grad()
def evaluate_recall_at_k(model, tokenizer, dataset, k=5, max_eval=512, max_length=512, device="cuda"):
    model.eval()
    N = min(len(dataset), max_eval)
    queries = [dataset[i]["query"] for i in range(N)]
    pos_docs = [dataset[i]["positive"] for i in range(N)]
    q_tok = tokenizer(queries, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    d_tok = tokenizer(pos_docs, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    q_emb = model.encode(q_tok["input_ids"], q_tok["attention_mask"])
    d_emb = model.encode(d_tok["input_ids"], d_tok["attention_mask"])
    sims = q_emb @ d_emb.t()
    topk = sims.topk(k, dim=1).indices
    correct = torch.arange(N, device=device).unsqueeze(1)
    return (topk == correct).any(dim=1).float().mean().item()

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="nvidia/llama-3.2-nv-embedqa-1b-v2")
    ap.add_argument("--dataset_name", default="sentence-transformers/msmarco-msmarco-MiniLM-L6-v3",
                    help="HF dataset name. e.g., sentence-transformers/msmarco-msmarco-MiniLM-L6-v3")
    ap.add_argument("--dataset_subset", default="triplet", help="HF dataset subset/config. e.g., triplet")
    ap.add_argument("--dataset_split", default="train")
    ap.add_argument("--dataset_dev_split", default=None, help="e.g., dev (with same subset)")
    ap.add_argument("--hf_revision", default=None)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--gradient_accumulation", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--query_prefix", type=str, default="<|query|> ")
    ap.add_argument("--doc_prefix", type=str, default="<|document|> ")

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--train_sample_limit", type=int, default=None)
    ap.add_argument("--dev_sample_limit", type=int, default=1024)
    ap.add_argument("--max_negs", type=int, default=1)
    ap.add_argument("--load_in_8bit", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # ---- Load train/dev (subset対応) ----
    train_ds = HFContrastiveDataset(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
        max_negs=args.max_negs,
        sample_limit=args.train_sample_limit,
        subset=args.dataset_subset,
        hf_revision=args.hf_revision
    )
    collate = Collator(tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, collate_fn=collate, num_workers=2)

    # main() 内、dev ロードのところを置き換え
    dev_ds = None
    if args.dataset_dev_split:
        try:
            dev_ds = HFContrastiveDataset(
                dataset_name=args.dataset_name,
                split=args.dataset_dev_split,
                query_prefix=args.query_prefix,
                doc_prefix=args.doc_prefix,
                max_negs=args.max_negs,
                sample_limit=args.dev_sample_limit,
                subset=args.dataset_subset,
                hf_revision=args.hf_revision
            )
        except Exception as e:
            print(f"[WARN] dev split '{args.dataset_dev_split}' not found. Using a slice of train as dev. ({e})")
            from torch.utils.data import Subset
            n = min(args.dev_sample_limit, len(train_ds))
            dev_ds = Subset(train_ds, range(n))

    model = BiEncoder(args.model_name, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, target_modules=None,
                      load_in_8bit=args.load_in_8bit).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    global_step, best_recall = 0, 0.0
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            q_ids, q_msk = batch["q_input_ids"].to(device), batch["q_attention_mask"].to(device)
            d_ids, d_msk = batch["d_input_ids"].to(device), batch["d_attention_mask"].to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                q_emb = model.encode(q_ids, q_msk)
                d_emb_all = model.encode(d_ids, d_msk)
                pos_indices = torch.tensor([start for (start, cnt) in batch["doc_ptrs"]],
                                           device=device, dtype=torch.long)
                d_pos = d_emb_all.index_select(0, pos_indices)
                loss, _ = info_nce(q_emb, d_pos, temperature=args.temperature)

            scaler.scale(loss).backward()
            if (global_step + 1) % args.gradient_accumulation == 0:
                scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
            global_step += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()))

            if dev_ds and (global_step % args.eval_every == 0):
                r5 = evaluate_recall_at_k(model, tokenizer, dev_ds,
                                          k=5, max_eval=min(args.dev_sample_limit, len(dev_ds)),
                                          max_length=min(512, args.max_length), device=device)
                if r5 > best_recall:
                    best_recall = r5
                    model.encoder.save_pretrained(os.path.join(args.output_dir, "best_adapter"))
                torch.save({"step": global_step, "recall@5": r5, "loss": float(loss.detach().cpu())},
                           os.path.join(args.output_dir, "last_state.pt"))

    model.encoder.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Best Recall@5={best_recall:.4f} | Train={len(train_ds)} | Dev={len(dev_ds) if dev_ds else 0} | Saved: {args.output_dir}")

if __name__ == "__main__":
    main()
