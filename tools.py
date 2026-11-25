"""
LangChain tools exposed to the LLM.

Three tools let the LLM inspect the dataset and run code explicitly,
rather than blindly emitting pandas code with no feedback loop.

  • get_dataframe_summary  – column names, dtypes, null counts, row/col shape
  • get_sample_rows        – first N rows of the DataFrame
  • run_python_code        – execute pandas code and return stdout or the error

A module-level _context dict is populated by agent.py before each LLM call
so the tools know which CSV to load.
"""

import os
import re
import subprocess
import sys
import tempfile

import pandas as pd
from langchain.tools import tool

# ---------------------------------------------------------------------------
# Shared context – set by agent.py before invoking the LLM
# ---------------------------------------------------------------------------
_context: dict = {"csv_path": ""}


def set_context(csv_path: str) -> None:
    """Point the tools at the CSV file for the current run."""
    _context["csv_path"] = csv_path


def _load_df() -> pd.DataFrame:
    return pd.read_csv(_context["csv_path"])


# ---------------------------------------------------------------------------
# Blocked patterns (mirrors runner.py – tools enforce the same safety rules)
# ---------------------------------------------------------------------------
_BLOCKED_IMPORTS = r"^\s*(import|from)\s+(os|sys|subprocess|shutil|socket|requests|urllib|pathlib)\b"
_BLOCKED_CALLS   = r"\b(open|eval|exec|compile|__import__)\s*\("
_BLOCKED_WRITES  = r"\.(to_csv|to_excel|to_pickle|to_json|to_parquet)\s*\("


def _check_code(code: str) -> str:
    if (re.search(_BLOCKED_IMPORTS, code, re.MULTILINE)
            or re.search(_BLOCKED_CALLS, code)
            or re.search(_BLOCKED_WRITES, code)):
        return "Blocked: code uses a disallowed import or function. Use only pandas operations on df."
    return ""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_dataframe_summary() -> str:
    """Return shape, column names, dtypes, and missing-value counts for the dataset.

    Use this before writing code to understand what columns are available
    and whether any have null values worth handling.
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


@tool
def run_python_code(code: str) -> str:
    """Execute Python/pandas code against the loaded DataFrame and return the output.

    A DataFrame named `df` is already loaded from the CSV — do not reload it.
    Use print() to surface the final result.
    Returns stdout on success, or the error message on failure.

    Args:
        code: Python code to execute. Must use only pandas operations on df.
    """
    problem = _check_code(code)
    if problem:
        return problem

    script = (
        f"import pandas as pd\n"
        f"df = pd.read_csv({_context['csv_path']!r})\n"
        + code
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name

    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            out = result.stdout.strip()
            return out if out else "Code ran but printed nothing. Use print() to show the result."
        return result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "Timeout: code took too long to run."
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# Convenience list – import this in agent.py
# ---------------------------------------------------------------------------
ANALYST_TOOLS = [get_dataframe_summary, get_sample_rows, run_python_code]
