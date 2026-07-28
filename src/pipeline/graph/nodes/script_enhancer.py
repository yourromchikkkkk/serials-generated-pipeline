"""Script enhancer stage — optional HITL clarifying-questions round."""

from langgraph.graph import END
from langgraph.types import interrupt
from langsmith import traceable

from pipeline.clients.litellm_client import chat_completion
from pipeline.config import get_settings
from pipeline.graph.state import Run, ScriptRevision

PREPARE = "script_enhancer_prepare"
AWAIT_ANSWERS = "script_enhancer_await_answers"


@traceable(run_type="llm", name="script_enhancer.generate_clarifying_questions")
def _generate_clarifying_questions(script_text: str) -> list[str]:
    response = chat_completion(
        model=get_settings().script_enhancer_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You help sharpen a short video script idea before production. Given a "
                    "1-3 sentence idea, ask up to 3 concise clarifying questions about setting, "
                    "tone, and key beats. Reply with one question per line, no numbering or "
                    "extra commentary."
                ),
            },
            {"role": "user", "content": script_text},
        ],
    )
    content = response.choices[0].message.content or ""
    return [line.strip("- ").strip() for line in content.splitlines() if line.strip()]


@traceable(run_type="llm", name="script_enhancer.merge_answers")
def _merge_answers(script_text: str, questions: list[str], answers: list[str]) -> str:
    qa_pairs = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions, answers))
    response = chat_completion(
        model=get_settings().script_enhancer_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the script idea into an expanded version that incorporates the "
                    "answers to the clarifying questions. Keep it tight enough for a short "
                    "vertical video. Reply with only the revised script text."
                ),
            },
            {
                "role": "user",
                "content": f"Original idea:\n{script_text}\n\nClarifying Q&A:\n{qa_pairs}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def prepare(run: Run) -> dict:
    """Gated by the `enable_script_enhancer` parameter supplied with the run
    skipped entirely unless the user opted in.

    Generates the clarifying questions and persists them to state *before* the interrupt in
    `await_answers`. Code before an `interrupt()` call re-runs on resume, so the LLM call has to
    live in a separate, earlier node — otherwise resuming would re-generate a fresh (possibly
    different) set of questions instead of reusing the ones the user answered."""
    if not run.parameters.get("enable_script_enhancer", False):
        return {"status": "script_enhancer_skipped"}

    revision_limit = run.parameters.get("revision_count_limit")
    if revision_limit is not None and run.revision_count >= revision_limit:
        return {"status": "script_enhancer_limit_reached"}

    questions = _generate_clarifying_questions(run.script_text)
    if not questions:
        return {"status": "script_enhancer_no_questions"}

    return {"clarifying_questions": questions, "status": "script_enhancer_awaiting_answers"}


def await_answers(run: Run) -> dict:
    questions = run.clarifying_questions or []
    answers = interrupt({"clarifying_questions": questions})
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise ValueError("script enhancer expected one answer per clarifying question")

    revised_text = _merge_answers(run.script_text, questions, answers)
    revision_number = run.revision_count + 1
    revision = ScriptRevision(
        run_id=run.run_id,
        revision_number=revision_number,
        previous_text=run.script_text,
        clarifying_questions=questions,
        user_answers=answers,
        revised_text=revised_text,
        revision_count=revision_number,
    )
    return {
        "script_text": revised_text,
        "revision_count": revision_number,
        "script_revisions": [*run.script_revisions, revision],
        "clarifying_questions": None,
        "status": "script_enhanced",
    }


def route_after_prepare(run: Run) -> str:
    if run.status == "script_enhancer_awaiting_answers":
        return AWAIT_ANSWERS
    return END
