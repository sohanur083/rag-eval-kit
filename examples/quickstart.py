from rag_eval import RagEvaluator

dataset = [
    {
        "question": "What is the boiling point of water?",
        "gold_doc_ids": ["doc_1"],
        "retrieved_docs": [
            {"id": "doc_1", "text": "Water boils at 100 degrees Celsius at standard pressure."},
            {"id": "doc_2", "text": "Mercury has a boiling point of 356 degrees."},
        ],
        "generated_answer": "Water boils at 100°C at standard atmospheric pressure. [doc_1]",
    },
    {
        "question": "Capital of Australia?",
        "gold_doc_ids": ["doc_5"],
        "retrieved_docs": [
            {"id": "doc_3", "text": "Sydney is the largest city in Australia."},
            {"id": "doc_5", "text": "Canberra is the capital of Australia."},
        ],
        "generated_answer": "The capital of Australia is Sydney.",
    },
]

result = RagEvaluator().evaluate(dataset)
print(result.summary())
result.to_html("rag_report.html")
print("\nHTML report saved → rag_report.html")
