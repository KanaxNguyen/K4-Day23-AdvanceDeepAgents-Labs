"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import (
    FINALIZER_PATH,
    REPORT_PATH,
    SOURCES_PATH,
    VALIDATOR_PATH,
    WORKDIR,
    build_lead_agent,
)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    cleaned = re.sub(r"[^\w]+", "-", str(topic).strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        return "topic"
    return cleaned[:60].rstrip("-")


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Conduct a comprehensive, multi-source deep research survey on the topic: '{topic}'.\n\n"
        "Follow your execution workflow strictly:\n"
        "1. Plan with write_todos to deconstruct the topic into at least 3-5 sub-questions.\n"
        "2. Delegate each sub-question to the 'researcher' subagent via the 'task' tool.\n"
        "3. Collect and consolidate all notes into /tmp/work/research/sources.json, ensuring at least 3 distinct source families (arxiv, hf-daily, hf-search, web) are present.\n"
        "4. Draft the comprehensive survey report in /tmp/work/report/report.md following REPORT_TEMPLATE.md with inline citations [n]. Do NOT write the ## References section.\n"
        "5. Execute /tmp/work/research/finalize_citations.py to generate references and renumber citations.\n"
        "6. Execute /tmp/work/research/check_citations.py and fix any issues until it outputs OK.\n"
        "7. Have the 'citation-checker' subagent spot-check 3-4 claims.\n"
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_calls_counter = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        # Tool calls
        calls = getattr(msg, "tool_calls", None) or []
        if not calls and hasattr(msg, "additional_kwargs"):
            calls = msg.additional_kwargs.get("tool_calls", [])
        for call in calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                tool_calls_counter[name] += 1

        # Token usage
        usage = getattr(msg, "usage_metadata", None) or {}
        if usage:
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
        elif hasattr(msg, "response_metadata"):
            token_usage = msg.response_metadata.get("token_usage", {})
            input_tokens += token_usage.get("prompt_tokens", 0)
            output_tokens += token_usage.get("completion_tokens", 0)

    subagent_calls = tool_calls_counter.get("task", 0)
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls_counter),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("Report is missing or empty in sandbox")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError("sources.json is missing or empty in sandbox")

    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources_data, list) or not sources_data:
            raise ValueError("sources.json must be a non-empty JSON list")
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}") from exc

    slug = slugify(topic)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_info = summarize(messages, elapsed, model_name)
    source_families = sorted(list({s.get("source") for s in sources_data if s.get("source")}))

    meta = {
        "topic": topic,
        **summary_info,
        "n_sources": len(sources_data),
        "source_families": source_families,
    }

    report_file = reports_dir / f"{slug}.md"
    sources_file = reports_dir / f"{slug}.sources.json"
    meta_file = reports_dir / f"{slug}.meta.json"

    sources_file.write_text(json.dumps(sources_data, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_file.write_bytes(report_bytes)

    return report_file


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = str(topic).strip()
    if not topic:
        print("Usage: python research.py \"<topic>\"", file=sys.stderr)
        return 2

    model = make_model()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "model")
    start = time.monotonic()

    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(
            backend,
            {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            },
        )
        agent = build_lead_agent(backend, model)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": build_prompt(topic)}]},
            config={"recursion_limit": 1000},
        )
        elapsed = time.monotonic() - start
        messages = result.get("messages", []) if isinstance(result, dict) else []

        try:
            report_path = save_outputs(backend, topic, messages, elapsed, model_name)
            print(f"Report saved to: {report_path}")
            return 0
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
