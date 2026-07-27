"""Input script stage"""

from pipeline.graph.state import Run


def load_script(run: Run) -> dict:
    if not run.script_text or not run.script_text.strip():
        raise ValueError("script_text must not be empty")
    return {"status": "script_loaded"}
