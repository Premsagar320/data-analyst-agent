import os, re, subprocess, sys, tempfile, time
from typing import TypedDict

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from llm import ask_llm, get_chat_llm, MODEL
from summary import summarize_df
from runner import execute_node
from logger import log_run
from tools import ANALYST_TOOLS, set_context

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


# Tool name → callable map for the ReAct loop
_TOOL_MAP = {t.name: t for t in ANALYST_TOOLS}


def _run_tool_loop(llm_with_tools, messages: list, max_rounds: int = 5) -> str:
    """Drive a bind_tools ReAct loop until the model stops calling tools.

    The LLM can call get_dataframe_summary or get_sample_rows any number of
    times. Each tool result is appended as a ToolMessage and fed back so the
    model can refine its understanding before writing the final code.

    Returns the text content of the last AIMessage (the final code).
    """
    for _ in range(max_rounds):
        response = llm_with_tools.invoke(messages)
        messages = messages + [response]

        # No tool calls → model is done; return its text output
        if not response.tool_calls:
            return response.content or ""

        # Execute every tool the model requested and collect results
        for tc in response.tool_calls:
            fn = _TOOL_MAP.get(tc["name"])
            try:
                result = fn.invoke(tc["args"]) if fn else f"Unknown tool: {tc['name']}"
            except Exception as exc:
                result = f"Tool error: {exc}"
            messages = messages + [
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            ]

    # Safety: return whatever the model last said if rounds exhausted
    return response.content or ""


def inspect_node(state):
    df = pd.read_csv(state["csv_path"])
    # Point the LangChain tools at the current CSV for this run.
    set_context(state["csv_path"])
    return {"summary": summarize_df(df), "retries": 0, "error": "", "output": "", "history": []}


def generate_node(state):
    llm_with_tools = get_chat_llm().bind_tools(ANALYST_TOOLS)
    messages = [
        SystemMessage("You are a helpful data analyst. "
                      "Use the available tools to explore the dataset before writing code."),
        HumanMessage(
            f"Dataset info:\n{state['summary']}\n\n"
            f"Question: {state['question']}\n\n"
            f"Use get_dataframe_summary or get_sample_rows if you need to inspect "
            f"column names or values. Then write pandas code that answers the question. "
            f"A DataFrame named df is already loaded. Use print() for the result. "
            f"Return only the final code, no explanation."
        ),
    ]
    raw = _run_tool_loop(llm_with_tools, messages)
    return {"code": clean_code(raw)}


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

    llm_with_tools = get_chat_llm().bind_tools(ANALYST_TOOLS)
    messages = [
        SystemMessage("You are a helpful data analyst. "
                      "Use the available tools to inspect the dataset before fixing code."),
        HumanMessage(
            f"Question: {state['question']}\n\n"
            f"Dataset info:\n{state['summary']}\n\n"
            f"{prior_failures}\n"
            f"Your latest code:\n{state['code']}\n\n"
            f"It failed with this error:\n{state['error']}\n\n"
            f"Use get_dataframe_summary or get_sample_rows if you need to verify column "
            f"names or values. Then fix the code. A DataFrame named df is already loaded. "
            f"Return only the corrected code, no explanation."
        ),
    ]
    raw = _run_tool_loop(llm_with_tools, messages)
    return {
        "code": clean_code(raw),
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
