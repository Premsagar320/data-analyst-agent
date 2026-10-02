import re
from agent import app

TESTS = [
    ("How many orders have quantity greater than 3?", "3"),
    ("Which region has the lowest total revenue? Revenue is quantity times price.", "East"),
    ("What is the average revenue per order in the South region? Revenue is quantity times price.", "1633"),
    ("What is the total revenue from Phone orders in the North and South regions together? Revenue is quantity times price.", "2400"),
]

results = []
for question, expected in TESTS:
    r = app.invoke({"csv_path": "data/sales.csv", "question": question})
    ok = re.search(rf"\b{expected}\b", r["output"]) is not None
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] retries={r['retries']} | {question}")
    if not ok:
        print("   expected:", expected)
        print("   got:", r["output"][:150] or r["error"][:150])

print(f"\nScore: {sum(results)}/{len(results)}")
