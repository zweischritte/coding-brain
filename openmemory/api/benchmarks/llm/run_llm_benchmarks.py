#!/usr/bin/env python
"""Run local LLM benchmarks via Ollama API.

Usage:
  python openmemory/api/benchmarks/llm/run_llm_benchmarks.py \
    --pull-missing \
    --output docs/BENCHMARK-LLM-RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


DEFAULT_MODELS = [
    "qwen3:8b",
    "qwen2.5:14b",
    "mistral:7b",
    "mixtral:8x7b",
    "phi3:mini",
]

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_CASES_PATH = Path(__file__).parent / "data" / "cases.json"

ALLOWED_CATEGORIES = {
    "decision",
    "convention",
    "architecture",
    "dependency",
    "workflow",
    "testing",
    "security",
    "performance",
    "runbook",
    "glossary",
}

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "categorization": {
        "type": "object",
        "properties": {
            "categories": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["categories"],
        "additionalProperties": False,
    },
    "entity_relations": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False,
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "relationship": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["source", "relationship", "destination"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["entities", "relations"],
        "additionalProperties": False,
    },
    "concepts": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string"},
                        "type": {"type": "string"},
                        "importance": {"type": "number"},
                        "context": {"type": "string"},
                        "mention_count": {"type": "integer"},
                    },
                    "required": [
                        "entity",
                        "type",
                        "importance",
                        "context",
                        "mention_count",
                    ],
                    "additionalProperties": False,
                },
            },
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string"},
                        "type": {"type": "string"},
                        "confidence": {"type": "number"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "entities": {"type": "array", "items": {"type": "string"}},
                        "source_type": {"type": "string"},
                    },
                    "required": [
                        "concept",
                        "type",
                        "confidence",
                        "evidence",
                        "entities",
                        "source_type",
                    ],
                    "additionalProperties": False,
                },
            },
            "summary": {"type": "string"},
            "language": {"type": "string"},
        },
        "required": ["entities", "concepts", "summary", "language"],
        "additionalProperties": False,
    },
}


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    rank = int(round((pct / 100) * (len(ordered) - 1)))
    return ordered[rank]


def _load_cases(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    cases = data.get("cases", [])
    if limit is not None:
        cases = cases[:limit]
    return cases


def _ollama_request(
    base_url: str,
    endpoint: str,
    payload: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    timeout: int = 600,
) -> requests.Response:
    url = f"{base_url.rstrip('/')}{endpoint}"
    return requests.post(url, json=payload, stream=stream, timeout=timeout)


def _ollama_get(base_url: str, endpoint: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _list_models(base_url: str) -> List[str]:
    data = _ollama_get(base_url, "/api/tags")
    return [model["name"] for model in data.get("models", [])]


def _pull_model(base_url: str, model: str) -> None:
    response = _ollama_request(base_url, "/api/pull", {"name": model}, stream=True)
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = payload.get("status")
        if status:
            print(f"[pull] {model}: {status}")
        if payload.get("error"):
            raise RuntimeError(payload["error"])


def _build_prompt(case: Dict[str, Any]) -> str:
    task = case["task"]
    text = case["text"]
    if task == "categorization":
        categories = ", ".join(sorted(ALLOWED_CATEGORIES))
        return (
            "Classify the memory into one or more categories from this list: "
            f"{categories}. Return JSON only that matches the schema.\n"
            f"Memory: {text}"
        )
    if task == "entity_relations":
        return (
            "Extract entities and relations from the text. Return JSON only that "
            "matches the schema. Use short relation labels.\n"
            f"Text: {text}"
        )
    if task == "concepts":
        return (
            "Extract business entities and concepts from the text. Preserve the "
            "original language and do not translate. Return JSON only that matches "
            "the schema.\n"
            f"Text: {text}"
        )
    raise ValueError(f"Unknown task: {task}")


def _validate_categorization(obj: Any) -> Tuple[bool, bool]:
    if not isinstance(obj, dict):
        return False, False
    categories = obj.get("categories")
    if not isinstance(categories, list):
        return False, False
    if not all(isinstance(item, str) for item in categories):
        return False, False
    normalized = [item.strip().lower() for item in categories]
    allowed_ok = all(item in ALLOWED_CATEGORIES for item in normalized)
    return True, allowed_ok


def _validate_entity_relations(obj: Any) -> Tuple[bool, bool]:
    if not isinstance(obj, dict):
        return False, False
    entities = obj.get("entities")
    relations = obj.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return False, False
    for entity in entities:
        if not isinstance(entity, dict):
            return False, False
        if not isinstance(entity.get("name"), str) or not isinstance(entity.get("type"), str):
            return False, False
    for relation in relations:
        if not isinstance(relation, dict):
            return False, False
        if not all(
            isinstance(relation.get(key), str)
            for key in ("source", "relationship", "destination")
        ):
            return False, False
    return True, True


def _validate_concepts(obj: Any) -> Tuple[bool, bool]:
    if not isinstance(obj, dict):
        return False, False
    entities = obj.get("entities")
    concepts = obj.get("concepts")
    summary = obj.get("summary")
    language = obj.get("language")
    if not isinstance(entities, list) or not isinstance(concepts, list):
        return False, False
    if not isinstance(summary, str) or not isinstance(language, str):
        return False, False
    for entity in entities:
        if not isinstance(entity, dict):
            return False, False
        required_keys = ["entity", "type", "importance", "context", "mention_count"]
        if any(key not in entity for key in required_keys):
            return False, False
    for concept in concepts:
        if not isinstance(concept, dict):
            return False, False
        required_keys = ["concept", "type", "confidence", "evidence", "entities", "source_type"]
        if any(key not in concept for key in required_keys):
            return False, False
    return True, True


def _validate_output(task: str, obj: Any) -> Tuple[bool, bool]:
    if task == "categorization":
        return _validate_categorization(obj)
    if task == "entity_relations":
        return _validate_entity_relations(obj)
    if task == "concepts":
        return _validate_concepts(obj)
    return False, False


def _run_case(
    base_url: str,
    model: str,
    case: Dict[str, Any],
    request_timeout: int,
    num_predict: Optional[int],
) -> Dict[str, Any]:
    prompt = _build_prompt(case)
    task = case["task"]
    schema = SCHEMAS[task]

    payload = {
        "model": model,
        "prompt": prompt,
        "format": schema,
        "stream": False,
        "options": {"temperature": 0},
    }
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict

    start = time.perf_counter()
    try:
        response = _ollama_request(
            base_url,
            "/api/generate",
            payload,
            timeout=request_timeout,
        )
        duration_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        return {
            "case_id": case["id"],
            "task": task,
            "language": case.get("language"),
            "latency_ms": duration_ms,
            "parse_ok": False,
            "schema_ok": False,
            "allowed_ok": False,
            "error": f"request_error: {exc}",
            "prompt_eval_count": None,
            "eval_count": None,
            "total_duration_ns": None,
            "tokens_per_sec": None,
        }

    raw_text = data.get("response", "")
    parse_ok = False
    schema_ok = False
    allowed_ok = False
    error = None

    try:
        parsed = json.loads(raw_text)
        parse_ok = True
        schema_ok, allowed_ok = _validate_output(task, parsed)
    except json.JSONDecodeError as exc:
        error = f"json_decode_error: {exc}"

    total_duration_ns = data.get("total_duration")
    prompt_eval_count = data.get("prompt_eval_count")
    eval_count = data.get("eval_count")
    tokens_per_sec = None
    if total_duration_ns and prompt_eval_count is not None and eval_count is not None:
        total_tokens = prompt_eval_count + eval_count
        total_seconds = total_duration_ns / 1_000_000_000
        if total_seconds > 0 and total_tokens > 0:
            tokens_per_sec = total_tokens / total_seconds

    return {
        "case_id": case["id"],
        "task": task,
        "language": case.get("language"),
        "latency_ms": duration_ms,
        "parse_ok": parse_ok,
        "schema_ok": schema_ok,
        "allowed_ok": allowed_ok,
        "error": error,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "total_duration_ns": total_duration_ns,
        "tokens_per_sec": tokens_per_sec,
    }


def _summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [item["latency_ms"] for item in results]
    parse_ok = [item for item in results if item["parse_ok"]]
    schema_ok = [item for item in results if item["schema_ok"]]
    allowed_ok = [item for item in results if item["allowed_ok"]]
    tokens_per_sec = [item["tokens_per_sec"] for item in results if item["tokens_per_sec"]]

    return {
        "requests": len(results),
        "parse_rate": len(parse_ok) / len(results) if results else 0.0,
        "schema_rate": len(schema_ok) / len(results) if results else 0.0,
        "allowed_rate": len(allowed_ok) / len(results) if results else 0.0,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_avg_ms": statistics.mean(latencies) if latencies else None,
        "tokens_per_sec_avg": statistics.mean(tokens_per_sec) if tokens_per_sec else None,
    }


def _summarize_by_task(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        by_task.setdefault(item["task"], []).append(item)
    return {task: _summarize_results(items) for task, items in by_task.items()}


def _render_markdown(
    metadata: Dict[str, Any],
    summaries: Dict[str, Dict[str, Any]],
    per_task: Dict[str, Dict[str, Dict[str, Any]]],
) -> str:
    lines = []
    lines.append("# Local LLM Benchmark Results")
    lines.append("")
    lines.append(f"Date: {metadata['timestamp']}")
    lines.append(f"Base URL: {metadata['base_url']}")
    lines.append(f"Ollama Version: {metadata.get('ollama_version', 'unknown')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Requests | Parse % | Schema % | Allowed % | P50 ms | P95 ms | Avg tok/s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for model, summary in summaries.items():
        lines.append(
            "| {model} | {requests} | {parse:.2%} | {schema:.2%} | {allowed:.2%} | {p50:.1f} | {p95:.1f} | {tps} |".format(
                model=model,
                requests=summary["requests"],
                parse=summary["parse_rate"],
                schema=summary["schema_rate"],
                allowed=summary["allowed_rate"],
                p50=summary["latency_p50_ms"] or 0.0,
                p95=summary["latency_p95_ms"] or 0.0,
                tps=f"{summary['tokens_per_sec_avg']:.1f}" if summary["tokens_per_sec_avg"] else "n/a",
            )
        )
    lines.append("")

    lines.append("## Per-task Breakdown")
    lines.append("")
    for model, tasks in per_task.items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Task | Requests | Parse % | Schema % | Allowed % | P50 ms | P95 ms |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for task, summary in tasks.items():
            lines.append(
                "| {task} | {requests} | {parse:.2%} | {schema:.2%} | {allowed:.2%} | {p50:.1f} | {p95:.1f} |".format(
                    task=task,
                    requests=summary["requests"],
                    parse=summary["parse_rate"],
                    schema=summary["schema_rate"],
                    allowed=summary["allowed_rate"],
                    p50=summary["latency_p50_ms"] or 0.0,
                    p95=summary["latency_p95_ms"] or 0.0,
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_outputs(
    output_path: Optional[Path],
    json_output_path: Optional[Path],
    metadata: Dict[str, Any],
    summaries: Dict[str, Dict[str, Any]],
    per_task_summary: Dict[str, Dict[str, Dict[str, Any]]],
    raw_results: Dict[str, List[Dict[str, Any]]],
) -> None:
    if json_output_path:
        payload = {
            "metadata": metadata,
            "summaries": summaries,
            "per_task": per_task_summary,
            "results": raw_results,
        }
        json_output_path.write_text(json.dumps(payload, indent=2))

    report = _render_markdown(metadata, summaries, per_task_summary)
    if output_path:
        output_path.write_text(report)
    else:
        print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local LLM benchmarks via Ollama.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pull-missing", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--num-predict", type=int, default=256)
    args = parser.parse_args()

    base_url = args.base_url
    cases = _load_cases(args.cases, args.limit)
    available = set(_list_models(base_url))
    ollama_version = _ollama_get(base_url, "/api/version").get("version")

    for model in args.models:
        if model not in available:
            if not args.pull_missing:
                print(f"[skip] {model} (not pulled)")
                continue
            print(f"[pull] {model}")
            _pull_model(base_url, model)
            available.add(model)

    summaries: Dict[str, Dict[str, Any]] = {}
    per_task_summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    raw_results: Dict[str, List[Dict[str, Any]]] = {}
    metadata_base = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "base_url": base_url,
        "ollama_version": ollama_version,
        "cases": len(cases),
    }

    for model in args.models:
        if model not in available:
            continue
        print(f"[run] {model}")
        model_results = []
        for case in cases:
            result = _run_case(
                base_url,
                model,
                case,
                args.request_timeout,
                args.num_predict if args.num_predict > 0 else None,
            )
            model_results.append(result)
        raw_results[model] = model_results
        summaries[model] = _summarize_results(model_results)
        per_task_summary[model] = _summarize_by_task(model_results)
        metadata = dict(metadata_base, models=list(summaries.keys()))
        _write_outputs(
            args.output,
            args.json_output,
            metadata,
            summaries,
            per_task_summary,
            raw_results,
        )

    metadata = dict(metadata_base, models=list(summaries.keys()))
    _write_outputs(
        args.output,
        args.json_output,
        metadata,
        summaries,
        per_task_summary,
        raw_results,
    )


if __name__ == "__main__":
    main()
