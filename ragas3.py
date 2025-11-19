from datasets import Dataset
from ragas import EvaluationDataset, evaluate

# --- Embedding (ローカル) ---
from sentence_transformers import SentenceTransformer
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- LLM (vLLM / OpenAI互換) ---
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# --- Ragas metrics ---
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextRecall,
    ContextPrecision,
)

# =============================
# 1. 評価データの準備
# =============================

data = {
    "question": [
        "What is the capital of Japan?",
    ],
    "answer": [
        "The capital of Japan is Tokyo.",
    ],
    "contexts": [
        ["Tokyo is the capital city of Japan."],
    ],
    "ground_truths": [
        ["Tokyo"],
    ],
}

hf_dataset = Dataset.from_dict(data)
eval_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)

# =============================
# 2. ローカル埋め込みモデル
# =============================

embedding_model_path = "/models/embedding-model/"  # 任意のローカルパス

embedding = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={"device": "cpu"},  # GPUなら "cuda"
    )
)

# =============================
# 3. LLM（vLLM の OpenAI互換 API）
# =============================

llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="qwen2.5-7b-instruct",   # vLLM がロードしているモデル名
        temperature=0,
        base_url="http://localhost:8000/v1",  # vLLM のエンドポイント
        api_key="dummy",   # vLLM は認証不要だが、必須扱いのためダミー
    )
)

# =============================
# 4. メトリクス定義
# =============================

metrics = [
    Faithfulness(llm=llm),
    ResponseRelevancy(llm=llm),

    # Embedding ベース（LLM不要）
    ContextRecall(embeddings=embedding),
    ContextPrecision(embeddings=embedding),
]

# =============================
# 5. 評価実行
# =============================

result = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    raise_exceptions=False
)

print(result.to_pandas())