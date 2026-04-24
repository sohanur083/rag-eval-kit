"""CLI entrypoint."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .core import RagEvaluator, EvalConfig


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag-eval")
    ap.add_argument("--dataset", required=True, help="JSONL file with RAG outputs")
    ap.add_argument("--output", default="rag_report.html", help="HTML report path")
    ap.add_argument("--json-out", default=None, help="Also dump metrics JSON")
    args = ap.parse_args(argv)

    dataset = [json.loads(l) for l in Path(args.dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    result = RagEvaluator(EvalConfig()).evaluate(dataset)
    print(result.summary())
    result.to_html(args.output)
    if args.json_out:
        result.to_json(args.json_out)
    print(f"\nHTML report: {args.output}")


if __name__ == "__main__":
    main()
