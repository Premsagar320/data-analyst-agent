import re
from agent import app

CSV = "data/sales.csv"

# (question, expected value that must appear in the printed output)
TESTS = [
    ("How many orders are there in total?", "8"),
    ("Which region has the highest total revenue? Revenue is quantity times price.", "South"),
    ("What is the total revenue across all orders? Revenue is quantity times price.", "10300"),
    ("Which product has the highest total revenue? Revenue is quantity times price.", "Laptop"),
    ("What is the average quantity per order?", "3.25"),
    ("How many orders came from the East region?", "2"),
    ("What is the total revenue for the North region? Revenue is quantity times price.", "4000"),
    ("What is the total revenue in March 2026? Revenue is quantity times price.", "3100"),
    ("What is the highest price in the data?", "800"),
]


def passed(output, expected):
    return re.search(rf"\b{re.escape(expected)}\b", output) is not None


results = []
for question, expected in TESTS:
    r = app.invoke({"csv_path": CSV, "question": question})
    ok = passed(r["output"], expected)
    results.append(ok)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] retries={r['retries']} | {question}")
    if not ok:
        print(f"   expected: {expected}")
        print(f"   got: {r['output'][:150] or r['error'][:150]}")

print(f"\nScore: {sum(results)}/{len(results)}")
