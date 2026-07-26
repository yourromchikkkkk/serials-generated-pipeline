"""Base pipeline graph. Later stages (script enhancer, tarantino eval, ...) are added as nodes here."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pipeline.graph.state import Run


def _load_script(run: Run) -> dict:
    if not run.script_text or not run.script_text.strip():
        raise ValueError("script_text must not be empty")
    return {"status": "script_loaded"}


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(Run)
    graph.add_node("load_script", _load_script)
    graph.add_edge(START, "load_script")
    graph.add_edge("load_script", END)
    return graph.compile()
