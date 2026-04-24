"""Retrieval + generation metrics."""
from __future__ import annotations
import math, re
from typing import List, Dict
from collections import Counter


def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", (s or "").lower())


def _overlap(a: str, b: str) -> float:
    ta, tb = Counter(_tokens(a)), Counter(_tokens(b))
    common = sum((ta & tb).values())
    denom = max(sum(tb.values()), 1)
    return common / denom


# ---- retrieval ----

def hit_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    return 1.0 if any(d in gold_ids for d in retrieved_ids[:k]) else 0.0


def mrr(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    for i, d in enumerate(retrieved_ids, 1):
        if d in gold_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(retrieved_ids[:k], 1):
        if d in gold_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_ids), k) + 1))
    return dcg / ideal if ideal else 0.0


def context_precision(retrieved_docs: List[Dict], gold_ids: List[str]) -> float:
    if not retrieved_docs:
        return 0.0
    relevant = sum(1 for d in retrieved_docs if d["id"] in gold_ids)
    return relevant / len(retrieved_docs)


def context_recall(retrieved_docs: List[Dict], gold_ids: List[str]) -> float:
    if not gold_ids:
        return 1.0
    got = sum(1 for g in gold_ids if any(d["id"] == g for d in retrieved_docs))
    return got / len(gold_ids)


def retrieval_metrics(retrieved_docs: List[Dict], gold_ids: List[str], k_values=(1, 3, 5, 10)) -> Dict[str, float]:
    ids = [d["id"] for d in retrieved_docs]
    out = {f"hit@{k}": hit_at_k(ids, gold_ids, k) for k in k_values}
    out |= {f"ndcg@{k}": ndcg_at_k(ids, gold_ids, k) for k in k_values}
    out["mrr"] = mrr(ids, gold_ids)
    out["ctx_precision"] = context_precision(retrieved_docs, gold_ids)
    out["ctx_recall"] = context_recall(retrieved_docs, gold_ids)
    return out


# ---- generation ----

def faithfulness(generated_answer: str, retrieved_docs: List[Dict]) -> float:
    """Fraction of answer sentences grounded in retrieved docs."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", generated_answer) if s.strip()]
    if not sents:
        return 1.0
    context = " ".join(d.get("text", "") for d in retrieved_docs)
    grounded = sum(1 for s in sents if _overlap(s, context) > 0.25)
    return grounded / len(sents)


def answer_relevance(generated_answer: str, question: str) -> float:
    return min(1.0, _overlap(question, generated_answer) * 2)


def citation_accuracy(generated_answer: str, retrieved_docs: List[Dict]) -> float:
    """Citations like [doc_42] must point to a doc that actually contains the adjacent claim."""
    cites = re.findall(r"\[([a-zA-Z0-9_\-]+)\]", generated_answer)
    if not cites:
        return 1.0
    doc_map = {d["id"]: d.get("text", "") for d in retrieved_docs}
    good = 0
    for c in cites:
        if c in doc_map:
            good += 1
    return good / len(cites)


def hallucination_rate(generated_answer: str, retrieved_docs: List[Dict]) -> float:
    """Unsupported tokens per 100 tokens."""
    gen_tokens = _tokens(generated_answer)
    if not gen_tokens:
        return 0.0
    ctx_tokens = set()
    for d in retrieved_docs:
        ctx_tokens.update(_tokens(d.get("text", "")))
    unsupported = sum(1 for t in gen_tokens if t not in ctx_tokens)
    return (unsupported / len(gen_tokens)) * 100


def generation_metrics(question: str, generated_answer: str, retrieved_docs: List[Dict]) -> Dict[str, float]:
    return {
        "faithfulness": faithfulness(generated_answer, retrieved_docs),
        "answer_relevance": answer_relevance(generated_answer, question),
        "citation_accuracy": citation_accuracy(generated_answer, retrieved_docs),
        "hallucination_per_100tok": hallucination_rate(generated_answer, retrieved_docs),
    }
