"""Retrieval evaluation against a labeled question set (hit@k, MRR, answer recall)."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from aiven_rag_mcp.models import SearchHit, SearchMode
from aiven_rag_mcp.service import RagService

EVAL_DEPTH = 10


class EvalQuestion(BaseModel):
    id: str
    question: str
    expected: list[str]
    answer_contains: Optional[str] = None


class QuestionResult(BaseModel):
    id: str
    rank: Optional[int]
    answer_in_top5: Optional[bool]
    top_source: Optional[str]
    latency_ms: float


class ModeSummary(BaseModel):
    mode: SearchMode
    rerank: bool
    questions: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr_at_10: float
    answer_in_top5: Optional[float]
    latency_p50_ms: float
    latency_p95_ms: float
    results: list[QuestionResult]


def load_questions(path: Path) -> list[EvalQuestion]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion.model_validate(q) for q in data["questions"]]


def _first_rank(hits: list[SearchHit], expected: list[str]) -> Optional[int]:
    for hit in hits:
        if any(hit.source_uri.endswith(suffix) for suffix in expected):
            return hit.rank
    return None


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct * (len(ordered) - 1))))
    return round(ordered[index], 1)


def evaluate(
    service: RagService,
    questions: list[EvalQuestion],
    mode: SearchMode,
    *,
    rerank: bool = False,
    collection: Optional[str] = None,
) -> ModeSummary:
    results: list[QuestionResult] = []
    for question in questions:
        started = time.perf_counter()
        response = service.search(
            question.question, k=EVAL_DEPTH, mode=mode, collection=collection, rerank=rerank
        )
        latency = (time.perf_counter() - started) * 1000
        hits = response.results
        answer_found: Optional[bool] = None
        if question.answer_contains:
            needle = question.answer_contains.lower()
            answer_found = any(needle in (h.heading + h.content).lower() for h in hits[:5])
        results.append(
            QuestionResult(
                id=question.id,
                rank=_first_rank(hits, question.expected),
                answer_in_top5=answer_found,
                top_source=hits[0].source_uri if hits else None,
                latency_ms=round(latency, 1),
            )
        )

    def hit_at(k: int) -> float:
        return round(
            sum(1 for r in results if r.rank is not None and r.rank <= k) / len(results), 3
        )

    answers = [r.answer_in_top5 for r in results if r.answer_in_top5 is not None]
    latencies = [r.latency_ms for r in results]
    return ModeSummary(
        mode=mode,
        rerank=rerank,
        questions=len(results),
        hit_at_1=hit_at(1),
        hit_at_3=hit_at(3),
        hit_at_5=hit_at(5),
        mrr_at_10=round(statistics.fmean(1 / r.rank if r.rank else 0.0 for r in results), 3),
        answer_in_top5=round(sum(answers) / len(answers), 3) if answers else None,
        latency_p50_ms=_percentile(latencies, 0.5),
        latency_p95_ms=_percentile(latencies, 0.95),
        results=results,
    )


def format_table(summaries: list[ModeSummary]) -> str:
    lines = [
        "| mode | hit@1 | hit@3 | hit@5 | MRR@10 | answer in top 5 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        answer = "-" if s.answer_in_top5 is None else f"{s.answer_in_top5:.2f}"
        label = s.mode + (" + rerank" if s.rerank else "")
        lines.append(
            f"| {label} | {s.hit_at_1:.2f} | {s.hit_at_3:.2f} | {s.hit_at_5:.2f} | "
            f"{s.mrr_at_10:.3f} | {answer} | {s.latency_p50_ms} | {s.latency_p95_ms} |"
        )
    return "\n".join(lines)
