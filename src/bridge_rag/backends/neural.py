"""Local embedding and cross-encoder retrieval.

Default tests do not construct this object. Loading it downloads
sentence-transformers/all-MiniLM-L6-v2 and BAAI/bge-reranker-v2-m3.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from bridge_rag.backends.text import TextUnit

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def gpu_status() -> dict:
    """Read VRAM without creating a CUDA context in this process."""
    try:
        memory = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"free_mb": 0, "compute_pids": []}
    free_lines = [line.strip() for line in memory.stdout.splitlines() if line.strip()]
    free_mb = int(free_lines[0]) if free_lines else 0
    pids = []
    for line in apps.stdout.splitlines():
        text = line.strip()
        if text.isdigit():
            pids.append(int(text))
    return {"free_mb": free_mb, "compute_pids": pids}


def resolve_device(requested: str | None = None, *, min_free_mb: int = 8192) -> str:
    if requested and requested != "auto":
        return requested
    status = gpu_status()
    if status["compute_pids"] or status["free_mb"] < min_free_mb:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)


def _logit(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    if logits.ndim == 2 and logits.size(-1) == 2:
        return logits[:, 1]
    return logits


@dataclass
class RankedUnit:
    unit: TextUnit
    coarse_score: float
    rerank_score: float


class NeuralRetriever:
    def __init__(
        self,
        *,
        device: str | None = None,
        embedding_model: str = EMBED_MODEL,
        reranker_model: str = RERANK_MODEL,
        coarse_k: int = 20,
        rerank_batch: int = 8,
        rerank_max_length: int = 512,
        embed_max_length: int = 128,
        fp16: bool | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.coarse_k = coarse_k
        self.rerank_batch = rerank_batch
        self.rerank_max_length = rerank_max_length
        self.embed_max_length = embed_max_length
        self.fp16 = (self.device == "cuda") if fp16 is None else fp16
        self._embed_tokenizer = None
        self._embed_model = None
        self._rerank_tokenizer = None
        self._rerank_model = None
        self._vector_cache: dict[str, torch.Tensor] = {}

    def load(self) -> "NeuralRetriever":
        if self._embed_model is not None:
            return self
        self._embed_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self._embed_model = AutoModel.from_pretrained(self.embedding_model).to(self.device)
        self._embed_model.eval()
        self._rerank_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model)
        dtype = torch.float16 if self.fp16 else None
        self._rerank_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model, torch_dtype=dtype)
        self._rerank_model.to(self.device)
        self._rerank_model.eval()
        return self

    def encode(self, texts: list[str]) -> torch.Tensor:
        self.load()
        missing = [text for text in texts if text not in self._vector_cache]
        if missing:
            tokenizer = self._embed_tokenizer
            model = self._embed_model
            assert tokenizer is not None and model is not None
            for start in range(0, len(missing), 32):
                batch = missing[start : start + 32]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.embed_max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    hidden = model(**encoded).last_hidden_state
                    pooled = functional.normalize(_mean_pool(hidden, encoded["attention_mask"]), p=2, dim=-1)
                for text, vector in zip(batch, pooled):
                    self._vector_cache[text] = vector.detach().to("cpu")
        return torch.stack([self._vector_cache[text] for text in texts], dim=0)

    def rerank_scores(self, query: str, passages: list[str]) -> list[float]:
        self.load()
        tokenizer = self._rerank_tokenizer
        model = self._rerank_model
        assert tokenizer is not None and model is not None
        scores: list[float] = []
        for start in range(0, len(passages), self.rerank_batch):
            batch = passages[start : start + self.rerank_batch]
            encoded = tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.rerank_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = _logit(model(**encoded).logits).float().detach().cpu()
            scores.extend(float(value) for value in logits.tolist())
        return scores

    def rank_units(self, query: str, units: list[TextUnit], top_k: int) -> list[RankedUnit]:
        if not units or top_k <= 0:
            return []
        self.load()
        passages = [unit.raw_text for unit in units]
        vectors = self.encode([query, *passages])
        query_vector = vectors[0]
        coarse = torch.matmul(vectors[1:], query_vector)
        order = torch.argsort(coarse, descending=True).tolist()
        chosen = order[: min(self.coarse_k, len(order))]
        chosen_units = [units[index] for index in chosen]
        scores = self.rerank_scores(query, [unit.raw_text for unit in chosen_units])
        ranked = [
            RankedUnit(unit=unit, coarse_score=float(coarse[index]), rerank_score=score)
            for unit, index, score in zip(chosen_units, chosen, scores)
        ]
        ranked.sort(key=lambda item: (-item.rerank_score, item.unit.doc_index, item.unit.sent_index, item.unit.passage_id))
        return ranked[:top_k]

    def memory_mb(self) -> dict[str, float | None]:
        if not torch.cuda.is_available() or self.device != "cuda":
            return {"allocated_mb": None, "peak_mb": None}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "peak_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        }
