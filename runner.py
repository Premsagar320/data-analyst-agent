import os, re, subprocess, sys, tempfile

BLOCKED_IMPORTS = r"^\s*(import|from)\s+(os|sys|subprocess|shutil|socket|requests|urllib|pathlib)\b"
BLOCKED_CALLS = r"\b(open|eval|exec|compile|__import__)\s*\("
BLOCKED_WRITES = r"\.(to_csv|to_excel|to_pickle|to_json|to_parquet)\s*\("


def check_code(code):
    if (re.search(BLOCKED_IMPORTS, code, re.MULTILINE)
            or re.search(BLOCKED_CALLS, code)
            or re.search(BLOCKED_WRITES, code)):
        return "Blocked: code uses a disallowed import or function. Use only pandas operations on df."
    return ""


def check_output(out):
    text = out.strip()
    if not text:
        return "Code ran but printed nothing. Use print() to show the result."
    low = text.lower()
    if low in ("nan", "none", "[]", "{}") or "empty dataframe" in low or "empty series" in low:
        return "Result is empty or NaN. Check column names and filters."
    return ""


def execute_node(state):
    problem = check_code(state["code"])
    if problem:
        return {"output": "", "error": problem}

    script = f"import pandas as pd\ndf = pd.read_csv({state['csv_path']!r})\n" + state["code"]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=15
        )
        out = result.stdout.strip()
        err = result.stderr.strip() if result.returncode != 0 else ""
    except subprocess.TimeoutExpired:
        out, err = "", "Timeout: code took too long"
    finally:
        os.remove(path)

    if not err:
        err = check_output(out)
    return {"output": out, "error": err}
