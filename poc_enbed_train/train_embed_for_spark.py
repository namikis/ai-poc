import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from accelerate import Accelerator
from bitsandbytes.optim import PagedAdamW
from torch.utils.data import DataLoader

# -----------------------------------------------------
# Config
# -----------------------------------------------------

MODEL_NAME = "nvidia/nv-embedqa-e5-v5"  # ←必要なら変更
OUTPUT_DIR = "./lora-output"

BATCH_SIZE = 1                      # 絶対安全
GRAD_ACCUM = 8                      # 有効バッチ = 8
MAX_SEQ_LEN = 512                   # メモリを劇的削減
LR = 2e-4
NUM_EPOCHS = 1

# -----------------------------------------------------
# Tokenizer
# -----------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------------------------------
# 4bit QLoRA 設定
# -----------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# -----------------------------------------------------
# Model Load（bf16, device_map固定）
# -----------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},        # auto は Unified Memory で暴走するので禁止
)

# 勾配チェックポイント（必須）
model.gradient_checkpointing_enable()

# -----------------------------------------------------
# LoRA 設定（Q/V のみ）
# -----------------------------------------------------

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# -----------------------------------------------------
# Dataset
# -----------------------------------------------------

# 例として HF の dummy データ
dataset = load_dataset("yelp_review_full", split="train[:1%]")

def tokenize_fn(example):
    text = example["text"]
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

tokenized = dataset.map(tokenize_fn, batched=True)
tokenized.set_format(type="torch")

dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------------------------------
# Accelerator（Trainer より圧倒的に省メモリ）
# -----------------------------------------------------

accelerator = Accelerator()
model, dataloader = accelerator.prepare(model, dataloader)

# Optimizer（Unified Memory 向け）
optimizer = PagedAdamW(model.parameters(), lr=LR)

# -----------------------------------------------------
# Training Loop（超メモリ効率）
# -----------------------------------------------------

for epoch in range(NUM_EPOCHS):
    model.train()
    for step, batch in enumerate(dataloader):
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        accelerator.backward(loss)

        if (step + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()
            accelerator.print(f"Epoch {epoch}, Step {step}, Loss={loss.item()}")

# -----------------------------------------------------
# Save LoRA
# -----------------------------------------------------

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

accelerator.print("=== Training Finished ===")