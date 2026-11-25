"""
LangChain tools for data inspection.

Two tools let the LLM examine the dataset *before* committing to code,
giving it real signal about column names, types, nulls, and actual values.

  • get_dataframe_summary  – column names, dtypes, null counts, row/col shape
  • get_sample_rows        – first N rows as a plain-text table

set_context(csv_path) must be called once per run (done by inspect_node in
agent.py) so the tools know which file to load.
"""

import pandas as pd
from langchain.tools import tool

# ---------------------------------------------------------------------------
# Shared context – set by agent.py before each LLM call
# ---------------------------------------------------------------------------
_context: dict = {"csv_path": ""}


def set_context(csv_path: str) -> None:
    """Point the tools at the CSV file for the current run."""
    _context["csv_path"] = csv_path


def _load_df() -> pd.DataFrame:
    return pd.read_csv(_context["csv_path"])


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_dataframe_summary() -> str:
    """Return shape, column names, dtypes, and missing-value counts for the dataset.

    Call this first to understand what columns are available and whether
    any have null values worth handling before writing pandas code.
    """
    df = _load_df()
    lines = [f"Rows: {len(df)}, Columns: {len(df.columns)}", "Columns and types:"]
    for col in df.columns:
        lines.append(f"  - {col}: {df[col].dtype}, missing={df[col].isna().sum()}")
    return "\n".join(lines)


@tool
def get_sample_rows(n: int = 5) -> str:
    """Return the first N rows of the dataset as a plain-text table.

    Use this to inspect actual values, spot mixed types, or verify column
    contents before writing analysis code.

    Args:
        n: Number of rows to return (default 5, capped at 20).
    """
    n = min(int(n), 20)
    df = _load_df()
    return df.head(n).to_string(index=False)


# ---------------------------------------------------------------------------
# Convenience list – imported by agent.py
# ---------------------------------------------------------------------------
ANALYST_TOOLS = [get_dataframe_summary, get_sample_rows]
