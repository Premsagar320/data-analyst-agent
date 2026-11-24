import re

import pandas as pd

from agent import app

CSV = "data/titanic.csv"

# ── compute expected answers from the data, not hardcoded ──────────────────
df = pd.read_csv(CSV)

EXPECTED = [
    ("How many passengers survived?",
     df["Survived"].sum()),
    ("What is the average age of the passengers?",
     df["Age"].mean()),
    ("How many passengers have a missing Age value?",
     df["Age"].isna().sum()),
    ("What percentage of female passengers survived?",
     df[df["Sex"] == "female"]["Survived"].mean() * 100),
    ("What percentage of male passengers survived?",
     df[df["Sex"] == "male"]["Survived"].mean() * 100),
    ("How many passengers are in third class?",
     (df["Pclass"] == 3).sum()),
    ("How many passengers embarked from Southampton (S)?",
     (df["Embarked"] == "S").sum()),
    ("How many passengers have a missing Cabin value?",
     df["Cabin"].isna().sum()),
    ("What is the age of the oldest passenger?",
     df["Age"].max()),
    ("What is the average fare paid by first class passengers?",
     df[df["Pclass"] == 1]["Fare"].mean()),
]


def _any_number_within(text, expected, tol=0.005):
    """Return True if any number found in *text* is within *tol* of *expected*."""
    expected = float(expected)
    for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", text):
        if abs(float(m.group()) - expected) <= abs(expected) * tol:
            return True
    return False


# ── run the agent and score ────────────────────────────────────────────────
results = []
for question, expected in EXPECTED:
    r = app.invoke({"csv_path": CSV, "question": question})
    output = r["output"] or r["error"]
    ok = _any_number_within(output, expected)
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] retries={r['retries']} | {question}")
    if not ok:
        print(f"   expected: {expected:.4f}")
        print(f"   got:      {output[:150]}")

print(f"\nScore: {sum(results)}/{len(results)}")
