"""rag-eval-kit — retrieval + generation evaluation for RAG systems."""
from .core import RagEvaluator, EvalConfig, EvalResult
from .metrics import retrieval_metrics, generation_metrics

__version__ = "0.1.0"
__all__ = ["RagEvaluator", "EvalConfig", "EvalResult", "retrieval_metrics", "generation_metrics"]
