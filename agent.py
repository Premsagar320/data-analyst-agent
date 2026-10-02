import os, re, subprocess, sys, tempfile
from typing import TypedDict

import pandas as pd
from langgraph.graph import StateGraph, START, END

from llm import ask_llm
from summary import summarize_df
from runner import execute_node

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


def clean_code(text):
    """Remove markdown fences the LLM adds around code."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def inspect_node(state):
    df = pd.read_csv(state["csv_path"])
    return {"summary": summarize_df(df), "retries": 0, "error": "", "output": ""}


def generate_node(state):
    prompt = f"""Dataset info:
{state['summary']}

Question: {state['question']}

Write pandas code that answers the question. A DataFrame named df is already loaded.
Use print() to show the final result. Return only code."""
    return {"code": clean_code(ask_llm(prompt))}


def fix_node(state):
    prompt = f"""Question: {state['question']}

Dataset info:
{state['summary']}

Your code:
{state['code']}

It failed with this error:
{state['error']}

Fix the code. A DataFrame named df is already loaded. Return only code."""
    return {"code": clean_code(ask_llm(prompt)), "retries": state["retries"] + 1}


def explain_node(state):
    if state["error"]:
        return {"answer": f"Could not complete after {state['retries']} retries. Last error: {state['error']}"}
    prompt = f"""Question: {state['question']}

Result of the analysis:
{state['output']}

Answer the question in 1-2 clear sentences using this result."""
    return {"answer": ask_llm(prompt)}


def route(state):
    if not state["error"]:
        return "explain"
    if state["retries"] >= MAX_RETRIES:
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

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({
        "csv_path": "data/sales.csv",
        "question": "Which region has the highest total revenue? Revenue is quantity times price.",
    })
    print("CODE:\n", result["code"])
    print("\nOUTPUT:\n", result["output"])
    print("\nANSWER:\n", result["answer"])
    print("\nRETRIES USED:", result["retries"])
