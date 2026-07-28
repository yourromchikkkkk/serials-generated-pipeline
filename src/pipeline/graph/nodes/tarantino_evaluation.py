"""Tarantino evaluation stage — automated quality gate on the script itself. Rejects weak
material before any paid generation begins, the cheapest place to catch a bad idea."""

from langgraph.graph import END
from langgraph.types import interrupt
from langsmith import traceable

from pipeline.clients.litellm_client import chat_completion
from pipeline.graph.state import Run, TarantinoEvaluation

# TODO: replace with anthropic/claude-haiku
MODEL = "openai/gpt-4o-mini"

EVALUATE = "tarantino_evaluate"
REWRITE = "tarantino_rewrite"
AWAIT_USER_DECISION = "tarantino_await_user_decision"

DEFAULT_SCORE_THRESHOLD = 0.7

_RUBRIC_PROMPT = (
    "You are a script quality gate for short-form vertical video. Score the script on "
    "originality, clarity of conflict, and suitability for short-form vertical video. Reply "
    "with exactly two lines, no extra commentary:\n"
    "SCORE: <float between 0 and 1>\n"
    "FEEDBACK: <one or two sentence critique>"
)


@traceable(run_type="llm", name="tarantino_evaluation.score")
def _score_script(script_text: str) -> tuple[float, str]:
    response = chat_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": _RUBRIC_PROMPT},
            {"role": "user", "content": script_text},
        ],
    )
    content = response.choices[0].message.content or ""
    score = 0.0
    feedback = ""
    for line in content.splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                score = 0.0
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()
    return score, feedback


@traceable(run_type="llm", name="tarantino_evaluation.rewrite")
def _auto_rewrite(script_text: str, feedback: str) -> str:
    response = chat_completion(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the script idea to address the critique below, keeping it tight "
                    "enough for a short vertical video. Reply with only the revised script text."
                ),
            },
            {"role": "user", "content": f"Script:\n{script_text}\n\nCritique:\n{feedback}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def evaluate(run: Run) -> dict:
    """Scores the current script and decides pass / routed_to_rewriter / routed_to_user, bounded
    by `tarantino_retry_count_limit`. Runs again after `rewrite` or a user-requested revision, so
    the retry count is checked before deciding to loop rather than only once up front.

    Gated by `enable_tarantino_evaluation` (default on) — unlike the script enhancer this stage
    is on by default, since it's the cheapest place to catch a bad idea before paid generation.

    If `tarantino_retry_count_limit` is exhausted and the script still hasn't passed, the run
    does not ship anyway — it stops for a mandatory human decision, same as a first-attempt
    failure with auto-rewriter disabled."""
    if not run.parameters.get("enable_tarantino_evaluation", True):
        return {"status": "tarantino_skipped"}

    threshold = run.parameters.get("tarantino_score_threshold", DEFAULT_SCORE_THRESHOLD)
    score, feedback = _score_script(run.script_text)

    if score >= threshold:
        decision = "pass"
        status = "tarantino_passed"
    else:
        retry_limit = run.parameters.get("tarantino_retry_count_limit")
        if retry_limit is not None and run.tarantino_retry_count >= retry_limit:
            decision = "routed_to_user"
            status = "tarantino_retry_limit_reached"
        elif run.parameters.get("enable_auto_rewriter", False):
            decision = "routed_to_rewriter"
            status = "tarantino_routed_to_rewriter"
        else:
            decision = "routed_to_user"
            status = "tarantino_routed_to_user"

    evaluation = TarantinoEvaluation(
        run_id=run.run_id,
        script_revision_id=str(len(run.tarantino_evaluations)),
        score=score,
        feedback_text=feedback,
        threshold_used=threshold,
        decision=decision,
        tarantino_retry_count=run.tarantino_retry_count,
    )
    return {
        "status": status,
        "tarantino_evaluations": [*run.tarantino_evaluations, evaluation],
    }


def rewrite(run: Run) -> dict:
    evaluation = run.tarantino_evaluations[-1]
    revised_text = _auto_rewrite(run.script_text, evaluation.feedback_text)
    return {
        "script_text": revised_text,
        "tarantino_retry_count": run.tarantino_retry_count + 1,
        "status": "tarantino_rewritten",
    }


def await_user_decision(run: Run) -> dict:
    evaluation = run.tarantino_evaluations[-1]
    payload = {"score": evaluation.score, "feedback": evaluation.feedback_text}
    if run.status == "tarantino_retry_limit_reached":
        payload["retry_limit_reached"] = True
    decision = interrupt(payload)
    if not isinstance(decision, dict) or decision.get("decision") not in ("approve", "revise"):
        raise ValueError("tarantino evaluation expected {'decision': 'approve' | 'revise', ...}")

    if decision["decision"] == "approve":
        return {"status": "tarantino_user_approved"}

    revised_text = decision.get("revised_text", run.script_text)
    return {
        "script_text": revised_text,
        "tarantino_retry_count": run.tarantino_retry_count + 1,
        "status": "tarantino_user_revised",
    }


def route_after_evaluate(run: Run) -> str:
    if run.status == "tarantino_routed_to_rewriter":
        return REWRITE
    if run.status in ("tarantino_routed_to_user", "tarantino_retry_limit_reached"):
        return AWAIT_USER_DECISION
    return END


def route_after_user_decision(run: Run) -> str:
    if run.status == "tarantino_user_revised":
        return EVALUATE
    return END
