#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm


# --------------------------
# Dataset
# --------------------------

class HFRerankDataset(Dataset):
    """
    Normalizes HF datasets that provide query / positive / negative columns for reranker fine-tuning.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        query_prefix: str,
        doc_prefix: str,
        max_negs: int = 3,
        sample_limit: Optional[int] = None,
        subset: Optional[str] = None,
        hf_revision: Optional[str] = None,
    ):
        ds_kwargs = {}
        if hf_revision:
            ds_kwargs["revision"] = hf_revision

        if subset:
            ds = load_dataset(dataset_name, subset, split=split, **ds_kwargs)
        else:
            ds = load_dataset(dataset_name, split=split, **ds_kwargs)

        self.samples: List[dict] = []
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.max_negs = max_negs

        for e in ds:
            q = (e.get("query") or "").strip()
            p = (e.get("positive") or "").strip()
            n = e.get("negative")
            if not q or not p:
                continue

            negs: List[str] = []
            if isinstance(n, str) and n.strip():
                negs = [n.strip()]
            elif isinstance(n, list):
                negs = [x.strip() for x in n if isinstance(x, str) and x.strip()]

            negs = negs[: max_negs or len(negs)]
            if not negs:
                continue

            self.samples.append(
                {
                    "query": self.query_prefix + q,
                    "positive": self.doc_prefix + p,
                    "negatives": [self.doc_prefix + x for x in negs],
                }
            )

        if sample_limit:
            self.samples = self.samples[:sample_limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


@dataclass
class PairCollator:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, batch: List[dict]) -> dict:
        queries: List[str] = []
        docs: List[str] = []
        group_ptrs: List[Tuple[int, int]] = []

        for sample in batch:
            candidates = [sample["positive"]] + sample["negatives"]
            start = len(queries)
            queries.extend([sample["query"]] * len(candidates))
            docs.extend(candidates)
            group_ptrs.append((start, len(candidates)))

        encoded = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        features = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "group_ptrs": group_ptrs,
        }
        if "token_type_ids" in encoded:
            features["token_type_ids"] = encoded["token_type_ids"]
        return features


# --------------------------
# Loss / Eval
# --------------------------

def groupwise_cross_entropy(logits: torch.Tensor, group_ptrs: List[Tuple[int, int]]) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for start, count in group_ptrs:
        group = logits[start : start + count].unsqueeze(0)
        target = torch.zeros(1, dtype=torch.long, device=logits.device)
        losses.append(torch.nn.functional.cross_entropy(group, target))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=logits.device)


@torch.no_grad()
def evaluate_top1_accuracy(
    model,
    tokenizer,
    dataset: Dataset,
    max_eval: int = 256,
    max_length: int = 512,
    device: str = "cuda",
) -> float:
    model.eval()
    total, correct = 0, 0
    limit = min(len(dataset), max_eval)
    for i in range(limit):
        sample = dataset[i]
        candidates = [sample["positive"]] + sample["negatives"]
        if len(candidates) < 2:
            continue
        encoded = tokenizer(
            [sample["query"]] * len(candidates),
            candidates,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded).logits.view(-1)
        if int(torch.argmax(logits)) == 0:
            correct += 1
        total += 1
    model.train()
    return correct / max(total, 1)


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="nvidia/llama-3.2-nv-rerankqa-1b-v2")
    ap.add_argument(
        "--dataset_name",
        default="sentence-transformers/msmarco-msmarco-MiniLM-L6-v3",
        help="HF dataset name that exposes query/positive/negative columns.",
    )
    ap.add_argument("--dataset_subset", default="triplet", help="HF dataset subset/config.")
    ap.add_argument("--dataset_split", default="train")
    ap.add_argument("--dataset_dev_split", default=None)
    ap.add_argument("--hf_revision", default=None)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gradient_accumulation", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", nargs="+", default=None)

    ap.add_argument("--query_prefix", type=str, default="<|query|> ")
    ap.add_argument("--doc_prefix", type=str, default="<|document|> ")

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--train_sample_limit", type=int, default=None)
    ap.add_argument("--dev_sample_limit", type=int, default=512)
    ap.add_argument("--max_negs", type=int, default=7)
    ap.add_argument("--load_in_8bit", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    train_ds = HFRerankDataset(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
        max_negs=args.max_negs,
        sample_limit=args.train_sample_limit,
        subset=args.dataset_subset,
        hf_revision=args.hf_revision,
    )
    if not len(train_ds):
        raise ValueError("Training dataset is empty. Please check dataset fields or filtering options.")

    collate = PairCollator(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        num_workers=2,
    )

    dev_ds: Optional[Dataset] = None
    if args.dataset_dev_split:
        try:
            dev_ds = HFRerankDataset(
                dataset_name=args.dataset_name,
                split=args.dataset_dev_split,
                query_prefix=args.query_prefix,
                doc_prefix=args.doc_prefix,
                max_negs=args.max_negs,
                sample_limit=args.dev_sample_limit,
                subset=args.dataset_subset,
                hf_revision=args.hf_revision,
            )
        except Exception as e:
            print(f"[WARN] dev split '{args.dataset_dev_split}' unavailable: {e}. Using a slice of train.")
            n = min(args.dev_sample_limit, len(train_ds))
            dev_ds = Subset(train_ds, range(n))

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=1,
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if args.load_in_8bit:
        prepare_model_for_kbit_training(model)

    target_modules = args.target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    global_step, best_acc = 0, 0.0
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            group_ptrs = batch.pop("group_ptrs")
            inputs = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=args.fp16):
                logits = model(**inputs).logits.view(-1)
                loss = groupwise_cross_entropy(logits, group_ptrs)

            scaler.scale(loss).backward()
            if (global_step + 1) % args.gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.set_postfix(loss=float(loss.detach().cpu()))

            if dev_ds and global_step % args.eval_every == 0:
                acc = evaluate_top1_accuracy(
                    model,
                    tokenizer,
                    dev_ds,
                    max_eval=min(args.dev_sample_limit, len(dev_ds)),
                    max_length=min(512, args.max_length),
                    device=device,
                )
                if acc > best_acc:
                    best_acc = acc
                    model.save_pretrained(os.path.join(args.output_dir, "best_adapter"))
                torch.save(
                    {"step": global_step, "accuracy@1": acc, "loss": float(loss.detach().cpu())},
                    os.path.join(args.output_dir, "last_state.pt"),
                )

    model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    tokenizer.save_pretrained(args.output_dir)
    print(
        f"Done. Best Accuracy@1={best_acc:.4f} | Train={len(train_ds)} "
        f"| Dev={len(dev_ds) if dev_ds else 0} | Saved: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
