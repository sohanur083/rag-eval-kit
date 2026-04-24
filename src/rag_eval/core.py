"""High-level evaluator."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import json
from pathlib import Path

from .metrics import retrieval_metrics, generation_metrics


@dataclass
class EvalConfig:
    k_values: tuple = (1, 3, 5, 10)
    faithfulness_threshold: float = 0.7
    use_llm_judge: bool = False
    llm_judge: Optional[Callable] = None
    n_workers: int = 1


@dataclass
class EvalResult:
    rows: List[Dict] = field(default_factory=list)
    config: EvalConfig = field(default_factory=EvalConfig)

    def aggregate(self) -> Dict[str, float]:
        if not self.rows:
            return {}
        keys = self.rows[0].keys()
        return {
            k: sum(r.get(k, 0) for r in self.rows) / len(self.rows)
            for k in keys if isinstance(self.rows[0].get(k), (int, float))
        }

    def composite_score(self) -> float:
        agg = self.aggregate()
        weights = {"hit@5": 0.2, "mrr": 0.15, "faithfulness": 0.3, "answer_relevance": 0.2, "citation_accuracy": 0.15}
        num = sum(agg.get(k, 0) * w for k, w in weights.items())
        denom = sum(w for k, w in weights.items() if k in agg)
        return num / denom if denom else 0.0

    def summary(self) -> str:
        agg = self.aggregate()
        lines = [
            "=" * 44,
            f"         RAG EVAL REPORT (n={len(self.rows)})",
            "=" * 44,
            "Retrieval",
        ]
        for k in self.config.k_values:
            key = f"hit@{k}"
            if key in agg:
                lines.append(f"  Hit@{k:<4} ..................... {agg[key]:.2f}")
        for k in ("mrr", "ndcg@5", "ctx_precision", "ctx_recall"):
            if k in agg:
                lines.append(f"  {k:<24} .. {agg[k]:.2f}")
        lines.append("")
        lines.append("Generation")
        for k in ("faithfulness", "answer_relevance", "citation_accuracy", "hallucination_per_100tok"):
            if k in agg:
                lines.append(f"  {k:<24} .. {agg[k]:.2f}")
        lines.append("")
        lines.append(f"Composite RAG-Score ........ {self.composite_score():.2f}")
        lines.append("=" * 44)
        return "\n".join(lines)

    def to_json(self, path: str):
        Path(path).write_text(json.dumps({
            "aggregate": self.aggregate(),
            "composite_score": self.composite_score(),
            "rows": self.rows,
        }, indent=2), encoding="utf-8")

    def to_html(self, path: str):
        agg = self.aggregate()
        bars = "".join(
            f'<div class="row"><span>{k}</span>'
            f'<div class="bar"><div style="width:{min(100, v*100):.0f}%"></div></div>'
            f'<span>{v:.2f}</span></div>'
            for k, v in agg.items() if 0 <= v <= 1
        )
        html = f"""<!doctype html><meta charset="utf-8"><title>RAG Eval Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:780px;margin:40px auto;padding:20px;background:#0b0b12;color:#e8e9ee}}
h1{{margin:0 0 24px}}
.row{{display:grid;grid-template-columns:260px 1fr 60px;gap:14px;align-items:center;margin:8px 0;font-family:ui-monospace,monospace;font-size:13px}}
.bar{{background:#1a1a22;height:18px;border-radius:3px;overflow:hidden}}
.bar>div{{background:linear-gradient(90deg,#7c5cff,#00d4ff);height:100%}}
.score{{font-size:32px;font-weight:700;color:#00d4ff;margin:20px 0}}
</style>
<h1>RAG Eval Report — n = {len(self.rows)}</h1>
<div class="score">Composite: {self.composite_score():.2f}</div>
{bars}
"""
        Path(path).write_text(html, encoding="utf-8")


class RagEvaluator:
    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()

    def evaluate(self, dataset: List[Dict]) -> EvalResult:
        rows = []
        for row in dataset:
            r = {}
            r.update(retrieval_metrics(row["retrieved_docs"], row.get("gold_doc_ids", []), self.config.k_values))
            r.update(generation_metrics(row["question"], row["generated_answer"], row["retrieved_docs"]))
            rows.append(r)
        return EvalResult(rows=rows, config=self.config)
