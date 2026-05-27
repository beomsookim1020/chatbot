"""Generation context pruning experiment.

This experiment keeps the existing RFP generation pipeline in place and adds a
thin context-selection layer on top of ``src.generation.rfp_generation``.  It is
designed to compare the current context builder against pruned, typed, and
hierarchical context variants without overwriting previous generation outputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.rfp_generation import (  # noqa: E402
    build_context_package,
    build_prompt,
    create_generation_summary,
    create_timestamped_output_dir,
    enrich_generation_record,
    load_chunk_index,
    load_generation_input_rows,
    load_source_store_index,
    postprocess_answer,
    prepare_generation_items,
    read_csv_records,
    truncate_text,
    truncate_text_preserve_lines,
    write_json,
    write_jsonl,
)


DEFAULT_PREDICTIONS = "outputs/predictions/best_variant_predictions.jsonl"
DEFAULT_EVAL_CSV = "data/eval/representative_wrong_30_eval_batch_format.csv"
DEFAULT_CHUNKS = "indexes/chroma_kure_v1_soyeon_690_260520_chunks_v2_690/chunks.jsonl"
DEFAULT_SOURCE_STORE = "data/source_store_v2_690.jsonl"
DEFAULT_OUTPUT_ROOT = "outputs/context_experiments"
DEFAULT_EXPERIMENT_ID = "context_pruning_690"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


ALREADY_IMPLEMENTED = [
    {
        "feature": "source_store 주입",
        "status": "이미 있음",
        "location": "src/generation/rfp_generation.py::build_context_package, _build_evidence_blocks",
        "note": "source_store_id로 원문을 lookup하고 source_full_text를 evidence block에 넣는 구조가 있음.",
    },
    {
        "feature": "enhanced_fields",
        "status": "이미 있음",
        "location": "notebooks/02_generation_legacy_variants.ipynb",
        "note": "레거시 실험 variant에서 field-aware/enhanced field 후보 추출을 실험함.",
    },
    {
        "feature": "typed_context",
        "status": "이미 있음",
        "location": "notebooks/02_generation_legacy_variants.ipynb, src/generation/rfp_generation.py",
        "note": "question type, intent plan, fact_type 기반 context 구성이 이미 있음.",
    },
    {
        "feature": "strict_guardrail",
        "status": "이미 있음",
        "location": "notebooks/02_generation_legacy_variants.ipynb, src/generation/rfp_generation.py::postprocess_answer",
        "note": "citation/numeric/policy validation과 failure tag 후처리가 있음.",
    },
    {
        "feature": "fact_candidates",
        "status": "이미 있음",
        "location": "src/generation/rfp_generation.py::QUESTION_TYPE_TO_FACT_TYPE, _expand_same_source_fact_blocks",
        "note": "fact_candidates chunk와 fact_type별 근거 확장 로직이 있음.",
    },
    {
        "feature": "computed_values",
        "status": "이미 있음",
        "location": "src/generation/rfp_generation.py::_compute_deterministic_values",
        "note": "예산 차액/합계/비율 등 deterministic 계산값을 context에 넣는 로직이 있음.",
    },
    {
        "feature": "citation validation",
        "status": "이미 있음",
        "location": "src/generation/rfp_generation.py::_attach_deterministic_citations, _validate_citations",
        "note": "LLM이 citation을 직접 만들지 않고 후처리에서 근거 block 기반으로 붙이고 검증함.",
    },
    {
        "feature": "question type 분류",
        "status": "이미 있음",
        "location": "src/generation/rfp_generation.py::classify_question",
        "note": "budget, duration, bid_deadline, submission_documents, eligibility 등을 분류함.",
    },
    {
        "feature": "context_max_chars",
        "status": "이미 있음",
        "location": "src/generation/rfp_generation.py::DEFAULT_GENERATION_CONFIG, _format_context_text",
        "note": "fact/synthesis context 최대 길이 설정이 있음.",
    },
    {
        "feature": "source_max_chars",
        "status": "이미 있음",
        "location": "src/generation/rfp_generation.py::DEFAULT_GENERATION_CONFIG['source_store_text_chars']",
        "note": "source_store 원문 주입 길이 제한이 있음.",
    },
]


QUESTION_GROUP_FACT_TYPES = {
    "budget": {
        "budget",
        "project_budget",
        "estimated_price",
        "base_amount",
        "total_allocation",
    },
    "date_or_period": {
        "date",
        "period",
        "duration",
        "project_duration",
        "bid_deadline",
        "submission_deadline",
        "submission_period",
        "maintenance_period",
        "warranty_period",
        "deadline_term",
        "other_deadline",
    },
    "qualification": {
        "qualification",
        "eligibility",
        "requirement",
        "requirements",
        "threshold_budget",
    },
    "submission_documents": {
        "submission_documents",
        "submission_logistics",
        "document",
        "proposal",
    },
    "general": {
        "document_summary",
        "business_type",
        "requirements",
        "evaluation",
    },
}


QUESTION_GROUP_KEYWORDS = {
    "budget": [
        "예산",
        "사업비",
        "기초금액",
        "금액",
        "가격",
        "추정가격",
        "산출내역",
        "budget",
        "amount",
        "price",
        "cost",
    ],
    "date_or_period": [
        "기간",
        "일정",
        "마감",
        "계약일",
        "제출마감",
        "입찰마감",
        "착수",
        "deadline",
        "period",
        "date",
    ],
    "qualification": [
        "참가자격",
        "자격",
        "제한요건",
        "실적",
        "입찰자격",
        "qualification",
        "requirement",
    ],
    "submission_documents": [
        "제출서류",
        "구비서류",
        "제안서",
        "서류",
        "별지",
        "서식",
        "submission",
        "document",
        "proposal",
    ],
    "general": [
        "요약",
        "목적",
        "배경",
        "요구사항",
        "평가",
        "summary",
        "requirement",
    ],
}


NOISY_FACT_TYPES = {"document_identity"}
NOISY_ANSWER_POLICIES = {"route_only_not_final_answer"}
BUDGET_BLOCKED_FACT_TYPES = {"threshold_budget", "payment_terms"}


@dataclass(frozen=True)
class ContextVariant:
    name: str
    description: str
    use_source_store: bool
    strategy: str
    rank_evidence: bool
    max_direct: int
    max_supporting: int
    max_reference: int
    max_chunks_per_doc: int
    max_source_items: int
    source_max_chars: int
    target_doc_source_only: bool
    context_max_chars_fact: int
    context_max_chars_synthesis: int
    evidence_text_chars: int
    include_reference_context: bool = True


VARIANTS = [
    ContextVariant(
        name="A_current_context_baseline",
        description="기존 rfp_generation context builder를 그대로 사용한다.",
        use_source_store=False,
        strategy="baseline",
        rank_evidence=False,
        max_direct=0,
        max_supporting=0,
        max_reference=0,
        max_chunks_per_doc=999,
        max_source_items=0,
        source_max_chars=0,
        target_doc_source_only=False,
        context_max_chars_fact=8000,
        context_max_chars_synthesis=12000,
        evidence_text_chars=900,
    ),
    ContextVariant(
        name="B_pruned_context_by_question_type",
        description="질문 유형과 fact_type이 맞는 근거만 우선 남기고 source_store는 사용하지 않는다.",
        use_source_store=False,
        strategy="pruned",
        rank_evidence=True,
        max_direct=4,
        max_supporting=3,
        max_reference=1,
        max_chunks_per_doc=2,
        max_source_items=0,
        source_max_chars=0,
        target_doc_source_only=False,
        context_max_chars_fact=5200,
        context_max_chars_synthesis=7600,
        evidence_text_chars=850,
    ),
    ContextVariant(
        name="C_pruned_context_with_limited_source_store",
        description="질문 유형 pruning 뒤 target 문서와 맞는 source_store 원문만 2개까지 확장한다.",
        use_source_store=True,
        strategy="pruned_source",
        rank_evidence=True,
        max_direct=4,
        max_supporting=3,
        max_reference=1,
        max_chunks_per_doc=2,
        max_source_items=2,
        source_max_chars=1000,
        target_doc_source_only=True,
        context_max_chars_fact=6200,
        context_max_chars_synthesis=8600,
        evidence_text_chars=850,
    ),
    ContextVariant(
        name="D_hierarchical_context_with_evidence_ranking",
        description="evidence ranking을 적용하고 direct/supporting/reference 계층으로 context를 재구성한다.",
        use_source_store=True,
        strategy="hierarchical_ranked",
        rank_evidence=True,
        max_direct=5,
        max_supporting=4,
        max_reference=2,
        max_chunks_per_doc=2,
        max_source_items=3,
        source_max_chars=1200,
        target_doc_source_only=True,
        context_max_chars_fact=7000,
        context_max_chars_synthesis=9800,
        evidence_text_chars=950,
    ),
]


AMOUNT_RE = re.compile(
    r"(?<!\d)(?:\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:조\s*원|억원|억\s*원|백만원|천만원|만원|천원|원|억)"
)
DATE_RE = re.compile(
    r"20\d{2}\s*[.\-/년]\s*\d{1,2}\s*[.\-/월]\s*\d{1,2}\s*(?:일)?"
    r"(?:\s*\d{1,2}\s*:\s*\d{2})?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--eval-csv", default=DEFAULT_EVAL_CSV)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS)
    parser.add_argument("--source-store", default=DEFAULT_SOURCE_STORE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--run-name", default="context_pruning")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all rows from the eval CSV.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Run only selected variant name. Can be repeated.",
    )
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Build and save contexts without loading the HF model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "predictions": project_path(args.predictions),
        "eval_csv": project_path(args.eval_csv),
        "chunks": project_path(args.chunks),
        "source_store": project_path(args.source_store),
        "output_root": project_path(args.output_root),
    }
    selected_variants = select_variants(args.variant)
    require_inputs(paths, selected_variants)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_timestamped_output_dir(
        paths["output_root"],
        args.run_name,
        run_timestamp=run_timestamp,
    )

    print("Output directory:", output_dir)
    print("Selected variants:", ", ".join(variant.name for variant in selected_variants))

    result_rows, context_rows = load_generation_input_rows(
        paths["predictions"],
        paths["predictions"],
        experiment_id=args.experiment_id,
    )
    eval_rows = read_csv_records(paths["eval_csv"])
    eval_id_order = extract_eval_ids(eval_rows)
    if not eval_id_order:
        raise ValueError(f"No question ids found in eval CSV: {paths['eval_csv']}")
    if args.limit and args.limit > 0:
        eval_id_order = eval_id_order[: args.limit]

    attach_eval_metadata(result_rows, eval_rows)
    items = prepare_generation_items(
        result_rows,
        context_rows,
        experiment_id=args.experiment_id,
        sample_size=None,
        review_focus=False,
    )
    items = order_items_by_eval(items, eval_id_order)
    chunk_ids, source_files = collect_chunk_selection(items)
    chunk_index = load_chunk_index(
        paths["chunks"],
        chunk_ids=chunk_ids or None,
        source_files=source_files or None,
    )
    if not chunk_index:
        raise ValueError(f"Chunk index is empty. Check chunks path: {paths['chunks']}")

    source_store_index = load_source_store_index(
        paths["source_store"],
        enabled=any(variant.use_source_store for variant in selected_variants),
    )
    if any(variant.use_source_store for variant in selected_variants) and not source_store_index:
        raise ValueError(f"source_store is required but could not be loaded: {paths['source_store']}")

    generator = None
    if not args.context_only:
        generator_cls = load_huggingface_generator_class()
        generator = generator_cls(
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

    run_config = {
        "predictions": str(paths["predictions"]),
        "eval_csv": str(paths["eval_csv"]),
        "chunks": str(paths["chunks"]),
        "source_store": str(paths["source_store"]),
        "experiment_id": args.experiment_id,
        "model_name": args.model_name,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "limit": args.limit,
        "context_only": args.context_only,
        "run_timestamp": run_timestamp,
        "variants": [variant.__dict__ for variant in selected_variants],
    }
    write_json(output_dir / "run_config.json", run_config)
    write_already_implemented(output_dir / "already_implemented.md")

    all_records: list[dict[str, Any]] = []
    variant_records: dict[str, list[dict[str, Any]]] = {}
    for variant in selected_variants:
        print(f"\n[{variant.name}] running {len(items)} items")
        records = run_variant(
            variant,
            items,
            chunk_index=chunk_index,
            source_store_index=source_store_index,
            generator=generator,
            args=args,
            run_timestamp=run_timestamp,
        )
        variant_records[variant.name] = records
        all_records.extend(records)
        variant_dir = output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=False)
        write_jsonl(variant_dir / "generated_answers.jsonl", records)
        write_rows_csv(variant_dir / "review.csv", build_review_rows(records))

    metrics_rows = build_metrics_rows(variant_records)
    write_jsonl(output_dir / "context_pruning_results.jsonl", all_records)
    write_rows_csv(output_dir / "context_pruning_review.csv", build_review_rows(all_records))
    write_rows_csv(output_dir / "context_pruning_metrics.csv", metrics_rows)
    write_summary_md(output_dir / "context_pruning_summary.md", metrics_rows, variant_records)

    print("\nSaved:")
    print(" -", output_dir / "context_pruning_results.jsonl")
    print(" -", output_dir / "context_pruning_review.csv")
    print(" -", output_dir / "context_pruning_metrics.csv")
    print(" -", output_dir / "context_pruning_summary.md")
    print_summary_to_console(metrics_rows, variant_records)


def run_variant(
    variant: ContextVariant,
    items: list[dict[str, Any]],
    *,
    chunk_index: dict[str, dict[str, Any]],
    source_store_index: dict[str, dict[str, Any]],
    generator: Any | None,
    args: argparse.Namespace,
    run_timestamp: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    active_source_store = source_store_index if variant.use_source_store else {}
    for idx, item in enumerate(items, start=1):
        context_package = build_variant_context_package(
            item,
            variant,
            chunk_index=chunk_index,
            source_store_index=active_source_store,
        )
        if generator is None:
            answer = context_only_answer(context_package)
            generation_ms = 0.0
        else:
            messages = build_prompt(context_package)
            started = time.perf_counter()
            raw_text = generator.generate_prompt(
                messages[1]["content"],
                system_prompt=messages[0]["content"],
            )
            generation_ms = (time.perf_counter() - started) * 1000
            answer = postprocess_answer(raw_text, context_package)
            answer["_raw_text"] = raw_text

        record = enrich_generation_record(
            answer,
            item,
            context_package,
            generation_ms=generation_ms,
            model_name=args.model_name if generator is not None else "context_only",
            experiment_name=variant.name,
            run_timestamp=run_timestamp,
        )
        record = add_required_experiment_fields(record, context_package, variant)
        records.append(record)

        if idx == 1 or idx % 5 == 0 or idx == len(items):
            qid = record.get("question_id", "")
            chars = record.get("context_char_count", 0)
            print(f"[{variant.name}] {idx}/{len(items)} {qid} context_chars={chars}")
    return records


def build_variant_context_package(
    item: dict[str, Any],
    variant: ContextVariant,
    *,
    chunk_index: dict[str, dict[str, Any]],
    source_store_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_config = {
        "max_context_chars_fact": variant.context_max_chars_fact if variant.strategy == "baseline" else 20000,
        "max_context_chars_synthesis": variant.context_max_chars_synthesis if variant.strategy == "baseline" else 24000,
        "max_blocks_fact": 6 if variant.strategy == "baseline" else 14,
        "max_blocks_synthesis": 10 if variant.strategy == "baseline" else 18,
        "evidence_text_chars": variant.evidence_text_chars,
        "source_store_text_chars": max(variant.source_max_chars, 1),
    }
    package = build_context_package(
        str(item.get("question") or ""),
        item.get("retrieved_contexts", []) or [],
        chunk_index=chunk_index,
        source_store_index=source_store_index,
        use_source_store=variant.use_source_store,
        config=base_config,
    )
    package["variant_name"] = variant.name
    package["variant_description"] = variant.description
    if variant.strategy == "baseline":
        return add_baseline_context_report(package)

    pruned = prune_context_package(package, variant)
    max_chars = (
        variant.context_max_chars_synthesis
        if package.get("question_analysis", {}).get("needs_synthesis")
        else variant.context_max_chars_fact
    )
    pruned["context_text"] = format_hierarchical_context(pruned, variant, max_chars=max_chars)
    return pruned


def prune_context_package(package: dict[str, Any], variant: ContextVariant) -> dict[str, Any]:
    analysis = package.get("question_analysis", {}) if isinstance(package.get("question_analysis"), dict) else {}
    group = infer_question_group(analysis, package.get("question", ""))
    blocks = [dict(block) for block in package.get("evidence_blocks", []) if isinstance(block, dict)]
    ranked = rank_blocks(blocks, analysis, group)
    direct, supporting, reference = select_hierarchical_blocks(ranked, analysis, group, variant)
    selected = direct + supporting + reference
    selected = apply_source_store_policy(selected, analysis, variant)

    selected_ids = {identity_key(block) for block in selected}
    dropped = [block for block in ranked if identity_key(block) not in selected_ids]
    core_summary = dict(package.get("core_summary", {}))
    if not should_include_computed_values(analysis, group):
        core_summary["computed_values"] = {}
        analysis = dict(analysis)
        analysis["computed_values"] = {}

    package = dict(package)
    package["question_analysis"] = analysis
    package["core_summary"] = core_summary
    package["evidence_blocks"] = selected
    package["context_pruning"] = {
        "strategy": variant.strategy,
        "question_group": group,
        "direct_count": len(direct),
        "supporting_count": len(supporting),
        "reference_count": len(reference),
        "dropped_count": len(dropped),
        "source_store_used_count": sum(1 for block in selected if block.get("_include_source_store")),
        "noisy_evidence_count": sum(1 for block in selected if is_noisy_block(block, group)),
        "dropped_noisy_evidence_count": sum(1 for block in dropped if is_noisy_block(block, group)),
        "used_source_store_ids": unique_values(
            block.get("source_store_id", "")
            for block in selected
            if block.get("_include_source_store") and block.get("source_store_id")
        ),
        "used_evidence_ids": evidence_ids(selected),
        "target_docs": target_docs(analysis),
    }
    return package


def add_baseline_context_report(package: dict[str, Any]) -> dict[str, Any]:
    package = dict(package)
    analysis = package.get("question_analysis", {}) if isinstance(package.get("question_analysis"), dict) else {}
    blocks = [block for block in package.get("evidence_blocks", []) if isinstance(block, dict)]
    group = infer_question_group(analysis, package.get("question", ""))
    package["context_pruning"] = {
        "strategy": "baseline",
        "note": "기존 context builder 출력 그대로 사용",
        "question_group": group,
        "direct_count": len(blocks),
        "supporting_count": 0,
        "reference_count": 0,
        "dropped_count": 0,
        "source_store_used_count": sum(1 for block in blocks if block.get("source_full_text")),
        "noisy_evidence_count": sum(1 for block in blocks if is_noisy_block(block, group)),
        "dropped_noisy_evidence_count": 0,
        "used_source_store_ids": unique_values(
            block.get("source_store_id", "")
            for block in blocks
            if block.get("source_full_text") and block.get("source_store_id")
        ),
        "used_evidence_ids": evidence_ids(blocks),
        "target_docs": target_docs(analysis),
    }
    return package


def rank_blocks(
    blocks: list[dict[str, Any]],
    analysis: dict[str, Any],
    group: str,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for block in blocks:
        text_blob = normalized_join(
            block.get("source_file", ""),
            block.get("section_path", ""),
            block.get("fact_type", ""),
            block.get("text", ""),
        )
        score = safe_float(block.get("score"), 0.0)
        rank = int(safe_float(block.get("rank"), 9999.0))
        reasons: list[str] = [f"base_score={score:.2f}"]

        if rank < 9999:
            rank_bonus = max(0.0, 6.0 - min(float(rank), 6.0)) * 0.35
            score += rank_bonus
            reasons.append(f"rank_bonus={rank_bonus:.2f}")

        fact_type = str(block.get("fact_type") or "")
        chunk_type = str(block.get("chunk_type") or "")
        if fact_type_matches_group(fact_type, group):
            score += 8.0
            reasons.append("fact_type_match")
        elif chunk_type == "fact_candidates" and group != "general":
            score -= 4.0
            reasons.append("fact_candidate_mismatch")

        if direct_keyword_match(text_blob, group):
            score += 3.0
            reasons.append("direct_keyword")

        if source_matches_target(block, analysis):
            score += 6.0
            reasons.append("target_doc_match")
        elif target_docs(analysis):
            score -= 5.0
            reasons.append("target_doc_mismatch")

        if str(block.get("answer_policy") or "") in NOISY_ANSWER_POLICIES:
            score -= 7.0
            reasons.append("route_only_policy")
        if fact_type in NOISY_FACT_TYPES:
            score -= 5.0
            reasons.append("document_identity")
        if group == "budget" and fact_type in BUDGET_BLOCKED_FACT_TYPES:
            score -= 9.0
            reasons.append("budget_blocked_fact")
        if bool(block.get("is_backfilled")):
            score -= 1.5
            reasons.append("backfilled")

        block = dict(block)
        block["_context_rank_score"] = round(score, 4)
        block["_context_rank_reasons"] = reasons
        block["_question_group"] = group
        ranked.append(block)
    ranked.sort(
        key=lambda block: (
            -safe_float(block.get("_context_rank_score"), 0.0),
            int(safe_float(block.get("rank"), 9999.0)),
            str(block.get("chunk_id") or ""),
        )
    )
    return ranked


def select_hierarchical_blocks(
    ranked: list[dict[str, Any]],
    analysis: dict[str, Any],
    group: str,
    variant: ContextVariant,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[str] = set()
    doc_counts: Counter[str] = Counter()
    direct: list[dict[str, Any]] = []
    supporting: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []

    def maybe_add(block: dict[str, Any], bucket: list[dict[str, Any]], max_items: int, section: str) -> bool:
        if len(bucket) >= max_items:
            return False
        key = identity_key(block)
        doc_key = normalize_doc_key(block.get("source_file", ""))
        if key in seen:
            return False
        if doc_key and doc_counts[doc_key] >= variant.max_chunks_per_doc:
            return False
        block = dict(block)
        block["_context_section"] = section
        bucket.append(block)
        seen.add(key)
        if doc_key:
            doc_counts[doc_key] += 1
        return True

    for block in ranked:
        if is_direct_answer_block(block, analysis, group):
            maybe_add(block, direct, variant.max_direct, "DIRECT_ANSWER_EVIDENCE")

    if should_allow_document_summary_fallback(analysis, group, direct):
        for block in ranked:
            if str(block.get("fact_type") or "") == "document_summary":
                maybe_add(block, direct, variant.max_direct, "DIRECT_ANSWER_EVIDENCE")

    for block in ranked:
        if is_noisy_block(block, group):
            continue
        if source_matches_target(block, analysis) or not target_docs(analysis):
            maybe_add(block, supporting, variant.max_supporting, "SUPPORTING_EVIDENCE")

    if variant.include_reference_context:
        for block in ranked:
            if variant.strategy == "pruned" and is_noisy_block(block, group):
                continue
            maybe_add(block, reference, variant.max_reference, "REFERENCE_CONTEXT")

    if not direct and ranked:
        maybe_add(ranked[0], direct, max(1, variant.max_direct), "DIRECT_ANSWER_EVIDENCE")

    return direct, supporting, reference


def apply_source_store_policy(
    selected: list[dict[str, Any]],
    analysis: dict[str, Any],
    variant: ContextVariant,
) -> list[dict[str, Any]]:
    used = 0
    result: list[dict[str, Any]] = []
    for block in selected:
        block = dict(block)
        include_source = False
        if variant.use_source_store and block.get("source_full_text") and used < variant.max_source_items:
            section = str(block.get("_context_section") or "")
            target_ok = source_matches_target(block, analysis) or not target_docs(analysis)
            if section in {"DIRECT_ANSWER_EVIDENCE", "SUPPORTING_EVIDENCE"}:
                include_source = target_ok if variant.target_doc_source_only else True
        if include_source:
            used += 1
            block["_include_source_store"] = True
            block["source_full_text"] = truncate_text(block.get("source_full_text", ""), variant.source_max_chars)
        else:
            block["_include_source_store"] = False
            block["source_full_text"] = ""
        result.append(block)
    return result


def format_hierarchical_context(
    package: dict[str, Any],
    variant: ContextVariant,
    *,
    max_chars: int,
) -> str:
    analysis = package.get("question_analysis", {})
    core_summary = package.get("core_summary", {})
    pruning = package.get("context_pruning", {})
    blocks = [block for block in package.get("evidence_blocks", []) if isinstance(block, dict)]
    by_section = {
        "DIRECT_ANSWER_EVIDENCE": [
            block for block in blocks if block.get("_context_section") == "DIRECT_ANSWER_EVIDENCE"
        ],
        "SUPPORTING_EVIDENCE": [
            block for block in blocks if block.get("_context_section") == "SUPPORTING_EVIDENCE"
        ],
        "REFERENCE_CONTEXT": [
            block for block in blocks if block.get("_context_section") == "REFERENCE_CONTEXT"
        ],
    }

    lines = [
        "[핵심 추출값 요약]",
        f"variant: {variant.name}",
        f"질문유형: {', '.join(analysis.get('question_types', []) or [])}",
        f"답변유형: {analysis.get('answer_type', 'unknown')}",
        f"context_pruning_group: {pruning.get('question_group', 'general')}",
    ]
    if analysis.get("intent_slots"):
        lines.append(f"의도 슬롯: {', '.join(analysis.get('intent_slots', []) or [])}")
    if analysis.get("period_subtypes"):
        lines.append(f"기간 세부유형: {', '.join(analysis.get('period_subtypes', []) or [])}")
    if core_summary.get("target_slots"):
        lines.append("")
        lines.append("[target slots]")
        for slot in core_summary.get("target_slots", []):
            lines.append(
                f"- target={slot.get('target_label', '')} | "
                f"matched_source_file={slot.get('matched_source_file', '') or '-'} | "
                f"match_score={slot.get('match_score', 0)}"
            )
    if core_summary.get("computed_values") and should_include_computed_values(
        analysis,
        pruning.get("question_group", "general"),
    ):
        lines.append("")
        lines.append("[computed values - 코드 계산 결과]")
        lines.append(json.dumps(core_summary.get("computed_values"), ensure_ascii=False))

    for section_name in ["DIRECT_ANSWER_EVIDENCE", "SUPPORTING_EVIDENCE", "REFERENCE_CONTEXT"]:
        section_blocks = by_section[section_name]
        lines.append("")
        lines.append(f"[{section_name}]")
        if not section_blocks:
            lines.append("- 없음")
            continue
        for idx, block in enumerate(section_blocks, start=1):
            lines.extend(format_evidence_block(block, idx, section_name))

    text = "\n".join(lines)
    return truncate_text_preserve_lines(text, max_chars)


def format_evidence_block(block: dict[str, Any], idx: int, section_name: str) -> list[str]:
    rank_score = block.get("_context_rank_score", "")
    reasons = ", ".join(block.get("_context_rank_reasons", []) or [])
    evidence_id = block.get("evidence_id") or f"E{idx}"
    lines = [
        "",
        (
            f"- {section_name} {idx}: evidence_id={evidence_id} | "
            f"source_file={block.get('source_file', '')} | chunk_id={block.get('chunk_id', '')} | "
            f"rank={block.get('rank', '')} | chunk_type={block.get('chunk_type', '') or '-'} | "
            f"fact_type={block.get('fact_type', '') or '-'} | section={block.get('section_path', '') or '-'} | "
            f"answer_policy={block.get('answer_policy', '') or '-'} | "
            f"context_rank_score={rank_score} | reasons={reasons or '-'}"
        ),
        str(block.get("text") or ""),
    ]
    if block.get("_include_source_store") and block.get("source_full_text"):
        lines.append(
            f"[source_store 확장 원문: source_store_id={block.get('source_store_id', '')}]"
        )
        lines.append(str(block.get("source_full_text") or ""))
    return lines


def add_required_experiment_fields(
    record: dict[str, Any],
    context_package: dict[str, Any],
    variant: ContextVariant,
) -> dict[str, Any]:
    pruning = context_package.get("context_pruning", {})
    analysis = context_package.get("question_analysis", {})
    used_context = context_package.get("context_text", "")
    record["variant_name"] = variant.name
    record["question_type"] = analysis.get("answer_type", "unknown")
    record["generated_answer"] = record.get("answer", "")
    record["gold_answer"] = record.get("ground_truth", "")
    record["used_context"] = used_context
    record["used_evidence_ids"] = pruning.get("used_evidence_ids") or evidence_ids(
        context_package.get("evidence_blocks", [])
    )
    record["used_source_store_ids"] = pruning.get("used_source_store_ids") or unique_values(
        block.get("source_store_id", "")
        for block in context_package.get("evidence_blocks", [])
        if isinstance(block, dict) and block.get("_include_source_store") and block.get("source_store_id")
    )
    record["context_char_count"] = len(used_context)
    record["context_pruning"] = pruning
    record["gold_signal_present"] = gold_signal_present(
        record.get("generated_answer", ""),
        record.get("gold_answer", ""),
    )
    return record


def context_only_answer(context_package: dict[str, Any]) -> dict[str, Any]:
    analysis = context_package.get("question_analysis", {})
    return {
        "answer": "",
        "answer_type": analysis.get("answer_type", "unknown"),
        "confidence": "low",
        "is_answerable": False,
        "final_values": {},
        "documents": [],
        "citations": [],
        "missing_info": ["context_only_run"],
        "warnings": ["generation was skipped by --context-only"],
        "_valid_json": None,
        "_failure_tags": ["context_only_run"],
    }


def build_metrics_rows(variant_records: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = compute_variant_metrics(variant_records.get("A_current_context_baseline", []))
    for variant_name, records in variant_records.items():
        metrics = compute_variant_metrics(records)
        row = {"variant_name": variant_name, **metrics}
        if baseline:
            base_chars = baseline.get("avg_context_chars", math.nan)
            current_chars = metrics.get("avg_context_chars", math.nan)
            row["context_char_reduction_vs_baseline_pct"] = percent_delta_reduction(
                base_chars,
                current_chars,
            )
            row["gold_signal_rate_delta_vs_baseline"] = safe_delta(
                metrics.get("gold_signal_present_rate"),
                baseline.get("gold_signal_present_rate"),
            )
            row["noisy_evidence_delta_vs_baseline"] = safe_delta(
                metrics.get("avg_noisy_evidence_count"),
                baseline.get("avg_noisy_evidence_count"),
            )
        rows.append(row)
    return rows


def compute_variant_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    summary = create_generation_summary(records)
    context_counts = [safe_float(record.get("context_char_count"), 0.0) for record in records]
    noisy_counts = [
        safe_float(record.get("context_pruning", {}).get("noisy_evidence_count"), 0.0)
        for record in records
    ]
    direct_counts = [
        safe_float(record.get("context_pruning", {}).get("direct_count"), 0.0)
        for record in records
    ]
    supporting_counts = [
        safe_float(record.get("context_pruning", {}).get("supporting_count"), 0.0)
        for record in records
    ]
    reference_counts = [
        safe_float(record.get("context_pruning", {}).get("reference_count"), 0.0)
        for record in records
    ]
    source_store_counts = [
        safe_float(record.get("context_pruning", {}).get("source_store_used_count"), 0.0)
        for record in records
    ]
    gold_values = [
        record.get("gold_signal_present")
        for record in records
        if record.get("gold_signal_present") is not None
    ]
    return {
        "total_questions": len(records),
        "valid_json_rate": summary.get("valid_json_rate"),
        "answer_available_rate": summary.get("answer_available_rate"),
        "citation_valid_rate": summary.get("citation_valid_rate"),
        "numeric_grounded_rate": summary.get("numeric_grounded_rate"),
        "answerable_rate": summary.get("answerable_rate"),
        "gold_signal_present_rate": mean_bool(gold_values),
        "avg_context_chars": mean(context_counts),
        "median_context_chars": median(context_counts),
        "avg_noisy_evidence_count": mean(noisy_counts),
        "avg_direct_evidence_count": mean(direct_counts),
        "avg_supporting_evidence_count": mean(supporting_counts),
        "avg_reference_evidence_count": mean(reference_counts),
        "source_store_used_rate": mean_bool(value > 0 for value in source_store_counts),
        "avg_source_store_items": mean(source_store_counts),
        "failure_tag_counts": json.dumps(summary.get("failure_tag_counts", {}), ensure_ascii=False),
    }


def build_review_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "variant_name": record.get("variant_name", ""),
                "question_id": record.get("question_id", ""),
                "question": record.get("question", ""),
                "question_type": record.get("question_type", ""),
                "generated_answer": record.get("generated_answer", ""),
                "gold_answer": record.get("gold_answer", ""),
                "gold_signal_present": record.get("gold_signal_present", ""),
                "retrieved_docs_top5": record.get("retrieved_docs_top5", ""),
                "used_evidence_ids": json.dumps(record.get("used_evidence_ids", []), ensure_ascii=False),
                "used_source_store_ids": json.dumps(record.get("used_source_store_ids", []), ensure_ascii=False),
                "context_char_count": record.get("context_char_count", ""),
                "failure_tags": json.dumps(record.get("_failure_tags", []), ensure_ascii=False),
                "manual_correct": "",
                "failure_type": "",
                "review_note": "",
            }
        )
    return rows


def write_summary_md(
    path: Path,
    metrics_rows: list[dict[str, Any]],
    variant_records: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Context Pruning Experiment Summary",
        "",
        "## Already Implemented",
    ]
    for item in ALREADY_IMPLEMENTED:
        lines.append(
            f"- {item['feature']}: {item['status']} "
            f"({item['location']}) - {item['note']}"
        )
    lines.extend(["", "## Metrics"])
    for row in metrics_rows:
        lines.append(
            "- {variant}: gold_signal_proxy={gold:.3f} | avg_context_chars={chars:.1f} | "
            "source_store_used_rate={source:.3f} | noisy_evidence_avg={noisy:.2f}".format(
                variant=row.get("variant_name", ""),
                gold=safe_float(row.get("gold_signal_present_rate"), math.nan),
                chars=safe_float(row.get("avg_context_chars"), math.nan),
                source=safe_float(row.get("source_store_used_rate"), math.nan),
                noisy=safe_float(row.get("avg_noisy_evidence_count"), math.nan),
            )
        )
    lines.extend(
        [
            "",
            "gold_signal_proxy는 사람 채점용 보조 지표입니다. 최종 정확도는 review CSV의 manual_correct/failure_type으로 판단합니다.",
            "",
            "## Failure Examples",
        ]
    )
    for example in failure_examples(variant_records, max_examples=5):
        lines.append(
            f"- {example['variant_name']} {example['question_id']}: "
            f"{truncate_text(example['question'], 140)} | "
            f"tags={example['failure_tags']} | answer={truncate_text(example['generated_answer'], 180)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary_to_console(
    metrics_rows: list[dict[str, Any]],
    variant_records: dict[str, list[dict[str, Any]]],
) -> None:
    print("\nExperiment summary")
    for row in metrics_rows:
        print(
            "- {variant}: gold_signal_proxy={gold:.3f}, avg_context_chars={chars:.1f}, "
            "char_reduction_vs_A={reduction}, source_store_used_rate={source:.3f}, "
            "noisy_evidence_avg={noisy:.2f}".format(
                variant=row.get("variant_name", ""),
                gold=safe_float(row.get("gold_signal_present_rate"), math.nan),
                chars=safe_float(row.get("avg_context_chars"), math.nan),
                reduction=format_float(row.get("context_char_reduction_vs_baseline_pct")),
                source=safe_float(row.get("source_store_used_rate"), math.nan),
                noisy=safe_float(row.get("avg_noisy_evidence_count"), math.nan),
            )
        )
    print("Failure examples:")
    for example in failure_examples(variant_records, max_examples=5):
        print(
            f"- {example['variant_name']} {example['question_id']}: "
            f"{truncate_text(example['question'], 90)} | tags={example['failure_tags']}"
        )


def failure_examples(
    variant_records: dict[str, list[dict[str, Any]]],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for variant_name, records in variant_records.items():
        for record in records:
            if record.get("gold_signal_present") is False or record.get("_failure_tags"):
                examples.append(
                    {
                        "variant_name": variant_name,
                        "question_id": record.get("question_id", ""),
                        "question": record.get("question", ""),
                        "generated_answer": record.get("generated_answer", ""),
                        "retrieved_docs_top5": record.get("retrieved_docs_top5", ""),
                        "failure_tags": record.get("_failure_tags", []),
                    }
                )
            if len(examples) >= max_examples:
                return examples
    return examples


def select_variants(requested: list[str]) -> list[ContextVariant]:
    if not requested:
        return VARIANTS
    by_name = {variant.name: variant for variant in VARIANTS}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(
            "Unknown variant(s): "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(by_name)
        )
    return [by_name[name] for name in requested]


def require_inputs(paths: dict[str, Path], variants: list[ContextVariant]) -> None:
    required = [
        ("predictions", paths["predictions"]),
        ("eval_csv", paths["eval_csv"]),
        ("chunks", paths["chunks"]),
    ]
    if any(variant.use_source_store for variant in variants):
        required.append(("source_store", paths["source_store"]))
    missing = [(name, path) for name, path in required if not path.exists()]
    if missing:
        details = "\n".join(f"- {name}: {path}" for name, path in missing)
        raise FileNotFoundError("Required input path does not exist:\n" + details)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_huggingface_generator_class() -> Any:
    """Load HuggingFaceGenerator without importing src.generator.__init__.

    The package __init__ imports OpenAIGenerator, but this experiment must not
    require or use OpenAI. Loading the module by file path keeps the dependency
    surface limited to transformers/torch when generation actually runs.
    """
    module_path = PROJECT_ROOT / "src" / "generator" / "huggingface_generator.py"
    spec = importlib.util.spec_from_file_location("local_huggingface_generator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load HuggingFaceGenerator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("local_huggingface_generator", module)
    spec.loader.exec_module(module)
    return module.HuggingFaceGenerator


def extract_eval_ids(eval_rows: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for row in eval_rows:
        qid = row.get("question_id") or row.get("id") or row.get("qid")
        if qid:
            ids.append(str(qid))
    return unique_values(ids)


def attach_eval_metadata(
    result_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> None:
    eval_by_id = {
        str(row.get("question_id") or row.get("id") or row.get("qid") or ""): row
        for row in eval_rows
    }
    for row in result_rows:
        qid = str(row.get("id") or row.get("question_id") or "")
        eval_row = eval_by_id.get(qid)
        if not eval_row:
            continue
        row["question"] = eval_row.get("question") or row.get("question")
        row["ground_truth_answer"] = (
            eval_row.get("ground_truth_answer")
            or eval_row.get("answer")
            or eval_row.get("gold_answer")
            or row.get("ground_truth_answer")
            or ""
        )
        row["ground_truth_docs"] = (
            eval_row.get("ground_truth_docs")
            or eval_row.get("gold_docs")
            or row.get("ground_truth_docs")
            or ""
        )
        for key in ["type", "difficulty", "metadata_filter"]:
            if eval_row.get(key) is not None:
                row[key] = eval_row.get(key)


def order_items_by_eval(
    items: list[dict[str, Any]],
    eval_id_order: list[str],
) -> list[dict[str, Any]]:
    items_by_id = {str(item.get("question_id")): item for item in items}
    missing = [qid for qid in eval_id_order if qid not in items_by_id]
    if missing:
        raise ValueError(
            "Predictions do not contain eval question ids: "
            + ", ".join(missing[:30])
        )
    return [items_by_id[qid] for qid in eval_id_order]


def collect_chunk_selection(items: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    chunk_ids: set[str] = set()
    source_files: set[str] = set()
    for item in items:
        for context in item.get("retrieved_contexts", []) or []:
            if not isinstance(context, dict):
                continue
            chunk_id = str(context.get("chunk_id") or "")
            if chunk_id:
                chunk_ids.add(chunk_id)
            source_file = str(context.get("source_file") or context.get("filename") or "")
            if source_file:
                source_files.add(source_file)
    return chunk_ids, source_files


def infer_question_group(analysis: dict[str, Any], question: str) -> str:
    answer_type = str(analysis.get("answer_type") or "")
    qtypes = {str(value) for value in analysis.get("question_types", []) or []}
    intents = {str(value) for value in analysis.get("intent_slots", []) or []}
    text = normalize_text(question)
    if answer_type == "budget" or "budget" in qtypes or any(intent.startswith("budget") for intent in intents):
        return "budget"
    if answer_type in {"duration", "bid_deadline"} or qtypes & {"duration", "bid_deadline"}:
        return "date_or_period"
    if answer_type == "eligibility" or "eligibility" in qtypes or "eligibility_check" in intents:
        return "qualification"
    if answer_type in {"submission_documents", "submission_logistics"} or qtypes & {
        "submission_documents",
        "submission_logistics",
    }:
        return "submission_documents"
    for group, keywords in QUESTION_GROUP_KEYWORDS.items():
        if any(normalize_text(keyword) in text for keyword in keywords):
            return group
    return "general"


def fact_type_matches_group(fact_type: str, group: str) -> bool:
    if not fact_type:
        return group == "general"
    return fact_type in QUESTION_GROUP_FACT_TYPES.get(group, set())


def direct_keyword_match(text: str, group: str) -> bool:
    return any(normalize_text(keyword) in text for keyword in QUESTION_GROUP_KEYWORDS.get(group, []))


def is_direct_answer_block(block: dict[str, Any], analysis: dict[str, Any], group: str) -> bool:
    fact_type = str(block.get("fact_type") or "")
    if is_noisy_block(block, group):
        return False
    if target_docs(analysis) and not source_matches_target(block, analysis):
        return False
    if fact_type_matches_group(fact_type, group):
        return True
    text_blob = normalized_join(block.get("section_path", ""), block.get("text", ""))
    return direct_keyword_match(text_blob, group) and str(block.get("chunk_type") or "") != "fact_candidates"


def should_allow_document_summary_fallback(
    analysis: dict[str, Any],
    group: str,
    direct: list[dict[str, Any]],
) -> bool:
    answer_type = str(analysis.get("answer_type") or "")
    qtypes = {str(value) for value in analysis.get("question_types", []) or []}
    if not direct:
        return True
    return group == "general" or answer_type in {"summary", "requirements", "business_type"} or bool(
        qtypes & {"summary", "requirements", "business_type"}
    )


def should_include_computed_values(analysis: dict[str, Any], group: str) -> bool:
    intents = {str(value) for value in analysis.get("intent_slots", []) or []}
    return group == "budget" or any(
        intent in {"budget_difference", "budget_sum", "budget_ratio"} for intent in intents
    )


def is_noisy_block(block: dict[str, Any], group: str) -> bool:
    fact_type = str(block.get("fact_type") or "")
    answer_policy = str(block.get("answer_policy") or "")
    if fact_type in NOISY_FACT_TYPES:
        return True
    if answer_policy in NOISY_ANSWER_POLICIES:
        return True
    if group == "budget" and fact_type in BUDGET_BLOCKED_FACT_TYPES:
        return True
    if str(block.get("chunk_type") or "") == "fact_candidates" and group != "general":
        return bool(fact_type and not fact_type_matches_group(fact_type, group))
    return False


def target_docs(analysis: dict[str, Any]) -> list[str]:
    docs: list[str] = []
    for slot in analysis.get("target_slots", []) or []:
        if not isinstance(slot, dict):
            continue
        matched = slot.get("matched_source_file")
        if matched:
            docs.append(str(matched))
    return unique_values(docs)


def source_matches_target(block: dict[str, Any], analysis: dict[str, Any]) -> bool:
    docs = target_docs(analysis)
    if not docs:
        return False
    source = normalize_doc_key(block.get("source_file") or block.get("source_file_nfc") or "")
    if not source:
        return False
    for doc in docs:
        target = normalize_doc_key(doc)
        if target and (source == target or source in target or target in source):
            return True
    return False


def evidence_ids(blocks: Iterable[dict[str, Any]]) -> list[str]:
    values = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        values.append(str(block.get("evidence_id") or block.get("chunk_id") or ""))
    return [value for value in unique_values(values) if value]


def identity_key(block: dict[str, Any]) -> str:
    for key in ["source_store_id", "evidence_id", "chunk_id"]:
        value = str(block.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    return json.dumps(
        {
            "source_file": block.get("source_file", ""),
            "text": str(block.get("text", ""))[:120],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def gold_signal_present(answer: str, gold_answer: str) -> bool | None:
    gold = str(gold_answer or "").strip()
    if not gold:
        return None
    answer_text = str(answer or "")
    answer_norm = normalize_text(answer_text)
    gold_norm = normalize_text(gold)
    if not answer_norm:
        return False

    gold_amounts = AMOUNT_RE.findall(gold)
    if gold_amounts:
        answer_compact = compact_text(answer_text)
        return any(compact_text(amount) in answer_compact for amount in gold_amounts)

    gold_dates = DATE_RE.findall(gold)
    if gold_dates:
        answer_compact = compact_text(answer_text)
        return any(compact_text(date) in answer_compact for date in gold_dates)

    if any(marker in gold_norm for marker in ["없", "미기재", "명시되어 있지", "확인되지"]):
        return any(
            marker in answer_norm
            for marker in ["없", "확인할 수 없", "명시되어 있지", "찾을 수 없", "not_found"]
        )

    tokens = [
        token
        for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", gold_norm)
        if token not in {"사업", "문서", "해당", "관련", "대한", "있는", "합니다", "입니다"}
    ]
    if not tokens:
        return gold_norm[:20] in answer_norm
    hits = sum(1 for token in tokens if token in answer_norm)
    return hits / max(len(tokens), 1) >= 0.2


def write_already_implemented(path: Path) -> None:
    lines = ["# Already Implemented Generation Context Features", ""]
    for item in ALREADY_IMPLEMENTED:
        lines.append(f"- {item['feature']}: {item['status']}")
        lines.append(f"  - 위치: {item['location']}")
        lines.append(f"  - 메모: {item['note']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize("NFC", str(text or ""))
    return re.sub(r"\s+", " ", value).strip().casefold()


def normalized_join(*values: Any) -> str:
    return normalize_text(" ".join(str(value or "") for value in values))


def normalize_doc_key(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = re.sub(r"\.[A-Za-z0-9]+$", "", text)
    text = re.sub(r"[^0-9A-Za-z가-힣]+", "", text)
    return text.casefold()


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    vals = [value for value in values if value is not None and not math.isnan(float(value))]
    return sum(vals) / len(vals) if vals else math.nan


def median(values: Iterable[float]) -> float:
    vals = sorted(value for value in values if value is not None and not math.isnan(float(value)))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def mean_bool(values: Iterable[Any]) -> float:
    vals = [value for value in values if value is not None]
    if not vals:
        return math.nan
    return sum(1.0 if bool(value) else 0.0 for value in vals) / len(vals)


def safe_delta(current: Any, baseline: Any) -> float:
    current_value = safe_float(current, math.nan)
    baseline_value = safe_float(baseline, math.nan)
    if math.isnan(current_value) or math.isnan(baseline_value):
        return math.nan
    return current_value - baseline_value


def percent_delta_reduction(baseline: Any, current: Any) -> float:
    baseline_value = safe_float(baseline, math.nan)
    current_value = safe_float(current, math.nan)
    if math.isnan(baseline_value) or math.isnan(current_value) or baseline_value == 0:
        return math.nan
    return (baseline_value - current_value) / baseline_value * 100


def format_float(value: Any) -> str:
    numeric = safe_float(value, math.nan)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.2f}"


def unique_values(values: Iterable[Any]) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value is None or value == "":
            continue
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


if __name__ == "__main__":
    main()
