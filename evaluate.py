# === Config ===
MODEL_NAME   = "nvidia/llama-3.2-nv-embedqa-1b-v2"
ADAPTER_DIR  = "runs/nv8bit_t4_mini/final_adapter"   # ←学習済みLoRAアダプタ保存先
MAX_LEN      = 192
QUERY_PREFIX = "<|query|> "
DOC_PREFIX   = "<|document|> "

# === Imports ===
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-5)

def build_encoder(model_name, load_adapter=False, adapter_dir=None):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base = AutoModel.from_pretrained(
        model_name, trust_remote_code=True,
        load_in_8bit=True, device_map="auto"
    )
    if load_adapter:
        base = PeftModel.from_pretrained(base, adapter_dir)
        base.to(device)
    return tok, base

def embed_texts(model, tok, texts, max_len=MAX_LEN):
    batch = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        last = out["last_hidden_state"] if isinstance(out, dict) else out[0]
        emb = mean_pool(last, batch["attention_mask"])
        emb = nn.functional.normalize(emb, dim=-1)
    return emb

def cosine(a, b):
    return (a * b).sum(-1).item()

# --- Load models ---
tok_base, model_base = build_encoder(MODEL_NAME, load_adapter=False)
tok_lora, model_lora = build_encoder(MODEL_NAME, load_adapter=True, adapter_dir=ADAPTER_DIR)

# --- Prepare sample pairs ---
# 1) Try to pull a few pairs from the triplet dataset (lightweight slice)
try:
    ds = load_dataset("sentence-transformers/msmarco-msmarco-MiniLM-L6-v3", "triplet", split="train[:3]")
    PAIRS = [
        (QUERY_PREFIX + ex["query"], DOC_PREFIX + ex["positive"]) for ex in ds
    ]
except Exception:
    # 2) Fallback to manual examples
    PAIRS = [
        (QUERY_PREFIX + "BGPセッションダウン時の確認コマンドは？",
         DOC_PREFIX   + "show ip bgp summary を実行し、Neighbor の状態とメッセージ数を確認する。"),
        (QUERY_PREFIX + "VLAN間ルーティングができない原因の切り分け手順は？",
         DOC_PREFIX   + "L3スイッチのSVIがUpか、ACLでブロックされていないか、ARP解決状況を順に確認する。"),
        (QUERY_PREFIX + "Tokyoの首都は？",
         DOC_PREFIX   + "Tokyo is the capital of Japan.")
    ]

# --- Compute & compare ---
for i, (q, d) in enumerate(PAIRS, 1):
    q_base = embed_texts(model_base, tok_base, [q])[0]
    d_base = embed_texts(model_base, tok_base, [d])[0]
    q_lora = embed_texts(model_lora, tok_lora, [q])[0]
    d_lora = embed_texts(model_lora, tok_lora, [d])[0]

    s_base = cosine(q_base, d_base)
    s_lora = cosine(q_lora, d_lora)
    print(f"[Pair {i}]")
    print(f"  Base cos: {s_base:.4f}")
    print(f"  LoRA cos: {s_lora:.4f}")
    print(f"  Δ (LoRA-Base): {s_lora - s_base:+.4f}\n")

