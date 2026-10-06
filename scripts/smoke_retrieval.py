#!/usr/bin/env python3
"""Rank a few sentences with MiniLM and bge-reranker.

Does not call an LLM API. If another process already holds the GPU, this
exits without loading weights so it does not take VRAM from that job.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bridge_rag.backends.neural import NeuralRetriever, gpu_status
from bridge_rag.backends.text import TextUnit


def main() -> int:
    status = gpu_status()
    if status["compute_pids"]:
        print(json.dumps({
            "skipped": True,
            "reason": "another compute process is using the GPU",
            "compute_pids": status["compute_pids"],
            "free_mb": status["free_mb"],
        }, ensure_ascii=False, indent=2))
        return 0
    retriever = NeuralRetriever(device="auto", rerank_batch=8, fp16=True)
    started = time.perf_counter()
    retriever.load()
    load_s = time.perf_counter() - started
    units = [
        TextUnit("grad", "q", "v", 0, 0, "林岚", "林岚毕业于明川大学，后来在海川大学任教。"),
        TextUnit("loc", "q", "v", 1, 0, "明川大学", "明川大学位于临江市。"),
        TextUnit("neg", "q", "v", 2, 0, "林岚", "林岚没有毕业于明川大学。"),
        TextUnit("co", "q", "v", 3, 0, "名单", "名单上有林岚、明川大学。"),
        TextUnit("other", "q", "v", 4, 0, "其他", "这家餐厅周末供应早餐。"),
        TextUnit("teach", "q", "v", 5, 0, "林岚", "林岚在海川大学任教。"),
    ]
    query = "林岚 毕业于 哪所大学"
    ranked = retriever.rank_units(query, units, top_k=4)
    elapsed = time.perf_counter() - started
    report = {
        "device": retriever.device,
        "gpu": None if retriever.device != "cuda" else __import__("torch").cuda.get_device_name(0),
        "embedding_model": retriever.embedding_model,
        "reranker_model": retriever.reranker_model,
        "fp16": retriever.fp16,
        "load_seconds": round(load_s, 3),
        "total_seconds": round(elapsed, 3),
        "memory_mb": {key: None if value is None else round(value, 1) for key, value in retriever.memory_mb().items()},
        "query": query,
        "ranked": [
            {"passage_id": item.unit.passage_id, "rerank_score": round(item.rerank_score, 4), "coarse_score": round(item.coarse_score, 4)}
            for item in ranked
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not ranked or ranked[0].unit.passage_id != "grad":
        print("expected the graduation sentence to rank first", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
