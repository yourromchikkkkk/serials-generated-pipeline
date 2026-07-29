"""Base pipeline graph. Wires together the node modules under `nodes/` — add a new stage by
creating a module in `nodes/` and registering it here, keeping this file to graph structure only."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pipeline.graph.nodes import (
    lipsync,
    load_script,
    per_shot_generation,
    script_enhancer,
    shot_list_generation,
    shot_review,
    shot_validation,
    tarantino_evaluation,
)
from pipeline.graph.state import Run


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(Run)
    graph.add_node("load_script", load_script.load_script)
    graph.add_node(script_enhancer.PREPARE, script_enhancer.prepare)
    graph.add_node(script_enhancer.AWAIT_ANSWERS, script_enhancer.await_answers)
    graph.add_node(tarantino_evaluation.EVALUATE, tarantino_evaluation.evaluate)
    graph.add_node(tarantino_evaluation.REWRITE, tarantino_evaluation.rewrite)
    graph.add_node(tarantino_evaluation.AWAIT_USER_DECISION, tarantino_evaluation.await_user_decision)
    graph.add_node(shot_list_generation.GENERATE, shot_list_generation.generate)
    graph.add_node(shot_validation.AWAIT_APPROVAL, shot_validation.await_approval)
    graph.add_node(per_shot_generation.GENERATE, per_shot_generation.generate)
    graph.add_node(shot_review.AWAIT_REVIEW, shot_review.await_review)
    graph.add_node(lipsync.GATE, lipsync.generate)

    graph.add_edge(START, "load_script")
    graph.add_edge("load_script", script_enhancer.PREPARE)
    graph.add_conditional_edges(
        script_enhancer.PREPARE,
        script_enhancer.route_after_prepare,
        {script_enhancer.AWAIT_ANSWERS: script_enhancer.AWAIT_ANSWERS, END: tarantino_evaluation.EVALUATE},
    )
    graph.add_edge(script_enhancer.AWAIT_ANSWERS, tarantino_evaluation.EVALUATE)

    graph.add_conditional_edges(
        tarantino_evaluation.EVALUATE,
        tarantino_evaluation.route_after_evaluate,
        {
            tarantino_evaluation.REWRITE: tarantino_evaluation.REWRITE,
            tarantino_evaluation.AWAIT_USER_DECISION: tarantino_evaluation.AWAIT_USER_DECISION,
            END: shot_list_generation.GENERATE,
        },
    )
    graph.add_edge(tarantino_evaluation.REWRITE, tarantino_evaluation.EVALUATE)
    graph.add_conditional_edges(
        tarantino_evaluation.AWAIT_USER_DECISION,
        tarantino_evaluation.route_after_user_decision,
        {tarantino_evaluation.EVALUATE: tarantino_evaluation.EVALUATE, END: shot_list_generation.GENERATE},
    )
    graph.add_edge(shot_list_generation.GENERATE, shot_validation.AWAIT_APPROVAL)
    graph.add_conditional_edges(
        shot_validation.AWAIT_APPROVAL,
        shot_validation.route_after_await_approval,
        {shot_validation.AWAIT_APPROVAL: shot_validation.AWAIT_APPROVAL, END: per_shot_generation.GENERATE},
    )
    graph.add_edge(per_shot_generation.GENERATE, shot_review.AWAIT_REVIEW)
    graph.add_conditional_edges(
        shot_review.AWAIT_REVIEW,
        shot_review.route_after_await_review,
        {shot_review.AWAIT_REVIEW: shot_review.AWAIT_REVIEW, END: lipsync.GATE},
    )
    graph.add_edge(lipsync.GATE, END)

    return graph.compile(checkpointer=MemorySaver())
