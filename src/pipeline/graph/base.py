"""Base pipeline graph. Wires together the node modules under `nodes/` — add a new stage by
creating a module in `nodes/` and registering it here, keeping this file to graph structure only."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pipeline.graph.nodes import load_script, script_enhancer
from pipeline.graph.state import Run


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(Run)
    graph.add_node("load_script", load_script.load_script)
    graph.add_node(script_enhancer.PREPARE, script_enhancer.prepare)
    graph.add_node(script_enhancer.AWAIT_ANSWERS, script_enhancer.await_answers)

    graph.add_edge(START, "load_script")
    graph.add_edge("load_script", script_enhancer.PREPARE)
    graph.add_conditional_edges(
        script_enhancer.PREPARE,
        script_enhancer.route_after_prepare,
        {script_enhancer.AWAIT_ANSWERS: script_enhancer.AWAIT_ANSWERS, END: END},
    )
    graph.add_edge(script_enhancer.AWAIT_ANSWERS, END)

    return graph.compile(checkpointer=MemorySaver())
