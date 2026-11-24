import os, re, subprocess, sys, tempfile, time
from typing import TypedDict

import pandas as pd
from langgraph.graph import StateGraph, START, END

from llm import ask_llm, MODEL
from summary import summarize_df
from runner import execute_node
from logger import log_run

MAX_RETRIES = 3


class AgentState(TypedDict):
    csv_path: str
    question: str
    summary: str
    code: str
    output: str
    error: str
    retries: int
    answer: str
    history: list  # each entry: {"code": str, "error": str}


def clean_code(text):
    """Remove markdown fences the LLM adds around code."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def inspect_node(state):
    df = pd.read_csv(state["csv_path"])
    return {"summary": summarize_df(df), "retries": 0, "error": "", "output": "", "history": []}


def generate_node(state):
    prompt = f"""Dataset info:
{state['summary']}

Question: {state['question']}

Write pandas code that answers the question. A DataFrame named df is already loaded.
Use print() to show the final result. Return only code."""
    return {"code": clean_code(ask_llm(prompt))}


def fix_node(state):
    # Record the current failed attempt before trying again.
    updated_history = state["history"] + [{"code": state["code"], "error": state["error"]}]

    # Build a section listing every prior failure so the LLM avoids them.
    prior_failures = ""
    if updated_history:
        lines = ["The following fixes have already been tried and MUST NOT be repeated:\n"]
        for i, entry in enumerate(updated_history, 1):
            lines.append(f"Attempt {i}:")
            lines.append(f"Code:\n{entry['code']}")
            lines.append(f"Error:\n{entry['error']}\n")
        prior_failures = "\n".join(lines)

    prompt = f"""Question: {state['question']}

Dataset info:
{state['summary']}

{prior_failures}
Your latest code:
{state['code']}

It failed with this error:
{state['error']}

Fix the code. A DataFrame named df is already loaded. Return only code."""
    return {
        "code": clean_code(ask_llm(prompt)),
        "retries": state["retries"] + 1,
        "history": updated_history,
    }


def explain_node(state):
    if state["error"]:
        return {"answer": f"Could not complete after {state['retries']} retries. Last error: {state['error']}"}
    prompt = f"""Question: {state['question']}

Result of the analysis:
{state['output']}

Answer the question in 1-2 clear sentences using this result."""
    return {"answer": ask_llm(prompt)}


def _last_line(err):
    """Return the last non-empty line of an error string (the exception message)."""
    lines = [l for l in err.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def route(state):
    if not state["error"]:
        return "explain"
    if state["retries"] >= MAX_RETRIES:
        return "explain"
    # If the exception message is identical to the previous attempt's, we are
    # stuck in a loop – stop retrying and surface the problem.
    history = state.get("history", [])
    if history and _last_line(history[-1]["error"]) == _last_line(state["error"]):
        return "explain"
    return "fix"


graph = StateGraph(AgentState)
graph.add_node("inspect", inspect_node)
graph.add_node("generate", generate_node)
graph.add_node("execute", execute_node)
graph.add_node("fix", fix_node)
graph.add_node("explain", explain_node)

graph.add_edge(START, "inspect")
graph.add_edge("inspect", "generate")
graph.add_edge("generate", "execute")
graph.add_conditional_edges("execute", route, {"fix": "fix", "explain": "explain"})
graph.add_edge("fix", "execute")
graph.add_edge("explain", END)

_compiled_app = graph.compile()


class _LoggedApp:
    """Thin wrapper around the compiled LangGraph app that logs every run."""

    def invoke(self, inputs, **kwargs):
        t0 = time.perf_counter()
        result = _compiled_app.invoke(inputs, **kwargs)
        elapsed = time.perf_counter() - t0
        log_run(
            model=MODEL,
            question=inputs.get("question", ""),
            retries=result.get("retries", 0),
            seconds=elapsed,
            had_error=bool(result.get("error", "")),
        )
        return result


app = _LoggedApp()


if __name__ == "__main__":
    result = app.invoke({
        "csv_path": "data/sales.csv",
        "question": "Which region has the highest total revenue? Revenue is quantity times price.",
    })
    print("CODE:\n", result["code"])
    print("\nOUTPUT:\n", result["output"])
    print("\nANSWER:\n", result["answer"])
    print("\nRETRIES USED:", result["retries"])
