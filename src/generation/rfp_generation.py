"""Field-aware generation utilities for the RFP RAG pipeline.

This module intentionally keeps source_store optional. The default team-share
mode uses retrieved chunks and chunk metadata only; source_store is a later
lookup-based evidence expansion path, not an embedding target.
"""

from __future__ import annotations

import csv
import html
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ANSWER_SCHEMA = {
    "answer": "string",
    "answer_type": (
        "budget|duration|bid_deadline|submission_documents|submission_logistics|"
        "eligibility|business_type|requirements|evaluation|summary|"
        "multi_doc_comparison|general|unknown"
    ),
    "confidence": "high|medium|low",
    "is_answerable": True,
    "final_values": {},
    "documents": [],
    "missing_info": [],
    "warnings": [],
}

FINAL_BUDGET_FACT_TYPES = {"budget", "project_budget", "estimated_price", "base_amount"}
BUDGET_BLOCKED_FACT_TYPES = {"threshold_budget", "payment_terms"}
ALLOCATION_BUDGET_FACT_TYPES = {"total_allocation"}
STRONG_PROJECT_BUDGET_CONTEXT_KEYWORDS = [
    "사업예산",
    "사업 예산",
    "예산금액",
    "예산 금액",
    "사업비",
    "총사업비",
    "총 사업비",
    "소요예산",
    "소요 예산",
    "배정예산",
    "배정 예산",
    "사업금액",
    "사업 금액",
    "계약금액",
    "계약 금액",
    "추정가격",
    "추정 가격",
    "추정금액",
    "추정 금액",
    "발주금액",
    "발주 금액",
    "용역금액",
    "용역 금액",
    "용역비",
    "총 용역금액",
    "총 용역 금액",
]
ALLOCATION_BUDGET_CONTEXT_KEYWORDS = [
    "전체 배정액",
    "전체배정액",
    "배정액",
    "배정예산",
    "배정 예산",
    "총 배정",
    "총배정",
    "전체 사업비",
    "전체사업비",
    "총사업비",
    "총 사업비",
    "소요예산",
    "소요 예산",
]
BLOCKED_BUDGET_FALLBACK_CONTEXT_KEYWORDS = [
    "기초금액",
    "기초 금액",
    "입찰보증",
    "계약보증",
    "하자보증",
    "보증금",
    "낙찰하한",
    "예정가격",
    "가격점수",
    "평가점수",
    "선금",
    "선수금",
    "잔금",
    "지급조건",
    "실적",
    "참가자격",
    "입찰참가자격",
    "미기재",
    "비공개",
    "미포함",
    "예산 미포함",
    "사업예산 미포함",
    "없음",
    "해당없음",
    "해당 없음",
]

DEFAULT_GENERATION_CONFIG = {
    "use_source_store": False,
    "max_context_chars_fact": 8000,
    "max_context_chars_synthesis": 12000,
    "max_blocks_fact": 6,
    "max_blocks_synthesis": 10,
    "evidence_text_chars": 900,
    "source_store_text_chars": 1400,
}

ALLOWED_ANSWER_TYPES = {
    "budget",
    "duration",
    "bid_deadline",
    "submission_documents",
    "submission_logistics",
    "eligibility",
    "business_type",
    "requirements",
    "evaluation",
    "summary",
    "multi_doc_comparison",
    "general",
    "unknown",
}

CANONICAL_FIELD_CANDIDATES = {
    "question": ["question", "query", "input.question", "result.question"],
    "question_id": ["question_id", "id", "qid", "input.id", "result.id"],
    "retrieved_contexts": ["retrieved_contexts", "contexts", "context", "items"],
    "source_file": ["source_file", "filename", "metadata.source_file"],
    "chunk_id": ["chunk_id", "metadata.chunk_id"],
    "section_title": ["section_title", "section_path", "metadata.section_path"],
    "text": ["text", "content", "page_content", "document", "evidence_text_short"],
    "table": ["table", "table_text", "metadata.table"],
    "fact": ["fact", "fact_type", "metadata.fact_type"],
    "fact_candidates": ["fact_candidates", "metadata.fact_candidates"],
    "score": ["score", "rerank_score", "rrf_score", "similarity", "distance"],
    "rank": ["rank", "retrieval_rank", "final_rank"],
    "metadata": ["metadata"],
    "source_store_id": ["source_store_id", "source_ref.source_store_id"],
}

QUESTION_KEYWORDS = {
    "budget": [
        "예산",
        "사업비",
        "사업 금액",
        "사업금액",
        "금액",
        "총액",
        "가격",
        "기초금액",
        "추정가격",
        "얼마",
        "얼말",
        "얼말루",
        "얼마입",
        "액수",
        "자금",
        "에산",
        "총규모",
        "발주비",
        "배정예산",
        "배정 예산",
        "전체 배정액",
        "전체배정액",
        "배정액",
        "전체 사업비",
        "전체사업비",
    ],
    "duration": [
        "사업기간",
        "수행기간",
        "계약기간",
        "기간",
        "착수일",
        "계약일",
        "계약 체결",
        "개월",
        "일간",
        "유지보수",
        "무상",
        "하자",
        "담보",
    ],
    "bid_deadline": [
        "입찰마감",
        "입찰 마감",
        "마감일",
        "마감 일시",
        "언제까지",
        "접수마감",
        "제출마감",
        "투찰",
        "개찰",
    ],
    "submission_documents": [
        "제출서류",
        "제출 서류",
        "구비서류",
        "구비 서류",
        "서류",
        "제안서",
        "가격제안서",
        "별지",
        "서식",
        "사업자등록증",
        "확약서",
        "서약서",
    ],
    "submission_logistics": [
        "제출처",
        "제출 방법",
        "제출방법",
        "제출 장소",
        "제출장소",
        "방문",
        "우편",
        "온라인",
        "이메일",
        "장소",
        "어디로",
        "접수처",
        "제출일시",
    ],
    "eligibility": [
        "참가자격",
        "참가 자격",
        "입찰자격",
        "입찰 자격",
        "자격요건",
        "자격 요건",
        "실적",
        "인증",
        "공동수급",
        "공동 수급",
        "하도급",
        "소프트웨어사업자",
        "중소기업",
        "직접생산",
    ],
    "business_type": [
        "사업유형",
        "유형",
        "구축",
        "운영",
        "고도화",
        "유지관리",
        "개발",
        "컨설팅",
        "ismp",
        "isp",
    ],
    "requirements": [
        "요구사항",
        "요구 사항",
        "기능",
        "성능",
        "보안",
        "인터페이스",
        "데이터",
        "과업",
        "범위",
        "산출물",
        "주의",
        "리스크",
        "db",
        "DB",
        "데이터베이스",
        "아키텍처",
        "통계",
        "병목",
        "쿼리",
        "캐싱",
        "캐시",
        "튜닝",
        "최적화",
        "오프라인",
        "터널",
        "재연결",
        "싱크",
        "동기화",
        "예측",
        "예지",
        "빅데이터",
        "이상 로그",
        "고장 조기",
        "조기 경보",
    ],
    "evaluation": [
        "평가",
        "배점",
        "기술평가",
        "가격평가",
        "정량",
        "정성",
        "협상",
        "선정",
        "점수",
    ],
    "multi_doc": [
        "비교",
        "각각",
        "둘 다",
        "두 사업",
        "두 문서",
        "간의",
        "차액",
        "합계",
        "공통",
        "둘",
        "동시에",
        "차이",
        "공통점",
        " vs ",
        "세 사업",
        "세 문서",
        "세 곳",
        "네 사업",
        "네 문서",
        "네 곳",
        "대조",
    ],
}

QUESTION_TYPE_TO_FACT_TYPE = {
    "budget": {"budget", "project_budget", "estimated_price", "base_amount", "total_allocation"},
    "duration": {
        "duration",
        "project_duration",
        "submission_deadline",
        "submission_period",
        "maintenance_period",
        "warranty_period",
        "deadline_term",
        "other_deadline",
    },
    "bid_deadline": {"bid_deadline"},
    "submission_documents": {"submission_documents"},
    "submission_logistics": {"submission_logistics"},
    "eligibility": {"eligibility"},
    "business_type": {"business_type", "document_summary"},
}

INTENT_REQUIRED_FACT_TYPES = {
    "budget_lookup": ["project_budget", "budget", "estimated_price", "base_amount"],
    "budget_difference": ["project_budget", "budget", "estimated_price", "base_amount"],
    "budget_sum": ["project_budget", "budget", "estimated_price", "base_amount"],
    "budget_ratio": ["project_budget", "budget", "estimated_price", "base_amount"],
    "total_allocation_lookup": ["total_allocation", "project_budget", "budget"],
    "duration_lookup": [
        "project_duration",
        "submission_deadline",
        "submission_period",
        "maintenance_period",
        "warranty_period",
        "deadline_term",
    ],
    "submission_documents": ["submission_documents"],
    "submission_logistics": ["submission_logistics"],
    "eligibility_check": ["eligibility", "threshold_budget"],
    "negative_check": ["eligibility", "requirements"],
    "purpose_summary": ["document_summary", "business_type", "requirements"],
    "requirements_summary": ["requirements", "business_type", "document_summary"],
    "requirements_list": ["requirements", "business_type"],
    "technical_requirement_lookup": ["requirements"],
    "technical_purpose_summary": ["requirements", "business_type", "document_summary"],
    "risk_comparison": ["requirements", "business_type", "document_summary"],
    "multi_doc_comparison": ["document_summary", "business_type", "requirements"],
    "general": ["document_summary"],
}

INTENT_PREFERRED_CHUNK_TYPES = {
    "budget_lookup": ["fact_candidates", "text", "table"],
    "budget_difference": ["fact_candidates", "text", "table"],
    "budget_sum": ["fact_candidates", "text", "table"],
    "budget_ratio": ["fact_candidates", "text", "table"],
    "total_allocation_lookup": ["fact_candidates", "text", "table"],
    "duration_lookup": ["fact_candidates", "text"],
    "submission_documents": ["fact_candidates", "table", "text"],
    "submission_logistics": ["fact_candidates", "text"],
    "eligibility_check": ["fact_candidates", "text", "table"],
    "negative_check": ["text", "table", "fact_candidates"],
    "purpose_summary": ["text", "table", "fact_candidates"],
    "requirements_summary": ["text", "table", "fact_candidates"],
    "requirements_list": ["table", "text", "fact_candidates"],
    "technical_requirement_lookup": ["text", "table", "fact_candidates"],
    "technical_purpose_summary": ["text", "table", "fact_candidates"],
    "risk_comparison": ["text", "table", "fact_candidates"],
    "multi_doc_comparison": ["text", "table", "fact_candidates"],
    "general": ["text", "table", "fact_candidates"],
}

INTENT_ANSWER_SECTIONS = {
    "budget_lookup": "예산",
    "budget_difference": "차액",
    "budget_sum": "합계",
    "budget_ratio": "계산",
    "total_allocation_lookup": "전체 배정액",
    "duration_lookup": "기간",
    "submission_documents": "제출서류",
    "submission_logistics": "제출 방법/일정",
    "eligibility_check": "입찰 자격",
    "negative_check": "포함 여부",
    "purpose_summary": "핵심 요약",
    "requirements_summary": "요구사항 요약",
    "requirements_list": "목록",
    "technical_requirement_lookup": "기술 조치",
    "technical_purpose_summary": "기술 전략",
    "risk_comparison": "비교/리스크",
    "multi_doc_comparison": "비교",
    "general": "답변",
}

SYNTHESIS_TYPES = {"requirements", "evaluation", "summary", "general"}
FACT_LOOKUP_TYPES = {
    "document_identity",
    "document_summary",
    "budget",
    "project_budget",
    "estimated_price",
    "base_amount",
    "total_allocation",
    "duration",
    "project_duration",
    "maintenance_period",
    "warranty_period",
    "deadline_term",
    "bid_deadline",
    "submission_documents",
    "submission_logistics",
    "eligibility",
    "business_type",
}

ANSWER_STATUS_VALUES = {
    "answered",
    "not_found_in_context",
    "insufficient_context",
    "ambiguous",
    "retrieval_context_missing",
}
ANSWERABLE_NEGATIVE_STATUSES = {"not_found_in_context"}
TARGET_MATCH_THRESHOLD = 0.34
STRONG_TARGET_MATCH_THRESHOLD = 0.55
PURPOSE_SUMMARY_KEYWORDS = [
    "목표",
    "목적",
    "배경",
    "필요성",
    "핵심",
    "핵심만",
    "짚어서",
    "요약",
    "의미",
    "전략",
    "효용",
    "효과",
    "파장",
    "리스크",
    "추진 내용",
    "추진내용",
    "성과 목표",
]
LIST_QUESTION_KEYWORDS = [
    "열거",
    "나열",
    "목록",
    "모두",
    "전부",
    "3가지",
    "세 가지",
    "네 가지",
    "4가지",
]
NEGATIVE_CHECK_KEYWORDS = [
    "명시되어 있습니까",
    "포함되어 있습니까",
    "기재되어 있습니까",
    "해야 합니까",
    "해야 하나",
    "해야 하는",
    "필수",
    "반드시",
    "기명해야",
    "있습니까",
    "있나요",
    "여부",
    "확인할 수",
    "없는",
    "없나요",
]
BUDGET_DIFFERENCE_KEYWORDS = ["차액", "차이", "편차", "얼마나 차이"]
BUDGET_SUM_KEYWORDS = ["합계", "총합", "더하면", "합산", "총액", "모두 더", "종합", "통합 결산", "결산"]
BUDGET_RATIO_KEYWORDS = ["%", "퍼센트", "비율", "분의", "월급", "단가", "남길", "나머지", "선수금", "잔금"]
TOTAL_ALLOCATION_KEYWORDS = [
    "전체 배정액",
    "전체배정액",
    "배정액",
    "배정예산",
    "배정 예산",
    "총 배정",
    "총배정",
    "전체 사업비",
    "전체사업비",
]
TECHNICAL_REQUIREMENT_KEYWORDS = [
    "db",
    "DB",
    "데이터베이스",
    "아키텍처",
    "통계",
    "병목",
    "쿼리",
    "캐싱",
    "캐시",
    "튜닝",
    "성능",
    "속도",
    "최적화",
    "오프라인",
    "터널",
    "재연결",
    "싱크",
    "동기화",
    "튕김",
    "장애복구",
]
TECHNICAL_PURPOSE_KEYWORDS = [
    "예측",
    "예지",
    "예지 보전",
    "유지보수",
    "빅데이터",
    "이상 로그",
    "고장 조기",
    "조기 경보",
    "패턴 학습",
    "로그 패턴",
]
RISK_COMPARISON_KEYWORDS = [
    "리스크",
    "파장",
    "피해",
    "장애",
    "장애 발생",
    "ddos",
    "DDoS",
    "페일 세이프",
    "fail-safe",
    "셧다운",
    "경제적",
    "물리적",
    "대조",
    "비교",
    "예측 비평",
]
SCENARIO_TARGET_BLOCK_KEYWORDS = [
    "페일 세이프",
    "fail-safe",
    "장애 발생",
    "시스템 통신 지연",
    "DDoS",
    "디도스",
    "셧다운",
    "오프라인",
    "터널",
]
QUESTION_ASPECT_REQUIREMENTS = [
    {
        "aspect": "factory_output",
        "question_markers": ["팩토리", "아웃풋", "생산", "제품"],
        "answer_markers": ["팩토리", "아웃풋", "생산", "제품", "라인", "공장"],
    },
    {
        "aspect": "field_impact",
        "question_markers": ["현장"],
        "answer_markers": ["현장", "실무", "운영", "효율", "소요시간", "검증"],
    },
    {
        "aspect": "technical_mechanism",
        "question_markers": ["통계", "병목", "쿼리", "캐싱", "DB", "데이터베이스", "오프라인", "재연결", "싱크"],
        "answer_markers": ["통계", "병목", "쿼리", "캐시", "캐싱", "DB", "데이터베이스", "최적화", "오프라인", "재연결", "동기화"],
    },
    {
        "aspect": "risk_comparison",
        "question_markers": ["리스크", "피해", "장애", "DDoS", "페일 세이프", "셧다운"],
        "answer_markers": ["리스크", "피해", "장애", "복구", "안전", "비교", "차이", "파장"],
    },
]

PERIOD_SUBTYPE_KEYWORDS = {
    "project_duration": ["사업기간", "수행기간", "계약기간", "계약 체결", "착수일", "계약일"],
    "submission_deadline": ["제출마감", "제출 마감", "제출기한", "제출 기한", "언제까지"],
    "submission_period": ["제출기간", "제출 기간", "접수기간", "접수 기간"],
    "bid_deadline": ["입찰마감", "입찰 마감", "투찰", "개찰", "마감일"],
    "maintenance_period": ["유지보수", "무상유지", "무상 유지", "운영지원"],
    "warranty_period": ["하자", "담보", "보증"],
    "other_deadline": ["통보", "조치", "납부", "완료", "보고"],
}

AMOUNT_RE = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
    r"(?:조\s*원|억원|억\s*원|억|백만원|백만|천만원|천만|만원|천원|원)"
)
NUMERIC_AMOUNT_RE = AMOUNT_RE
PERCENT_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:%|퍼센트)")
FRACTION_RE = re.compile(r"(\d+)\s*분의\s*(\d+)")
DATE_RE = re.compile(
    r"(?:20\d{2}\s*[.\-/년]\s*\d{1,2}\s*[.\-/월]\s*\d{1,2}\s*(?:일)?)"
    r"(?:\s*\d{1,2}\s*:\s*\d{2})?"
)
DURATION_RE = re.compile(
    r"(?:계약|착수|사업|수행|검수|완료|종료|하자|유지보수)[^\n.]{0,35}?"
    r"(?:\d+\s*(?:개월|일|년)|\d{4}\s*년[^\n.]{0,20})"
)


@dataclass
class EvidenceBlock:
    source_file: str
    chunk_id: str
    rank: int
    chunk_type: str
    fact_type: str
    section_path: str
    text: str
    score: float
    source_store_id: str = ""
    source_full_text: str = ""
    source_file_nfc: str = ""
    evidence_id: str = ""
    retrieval_role: str = ""
    answer_policy: str = ""
    answer_risk_level: str = ""
    budget_answer_enabled: bool = False
    eligibility_answer_enabled: bool = False
    payment_answer_enabled: bool = False
    selection_stage: str = ""
    is_backfilled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_file_nfc": self.source_file_nfc,
            "chunk_id": self.chunk_id,
            "evidence_id": self.evidence_id,
            "rank": self.rank,
            "chunk_type": self.chunk_type,
            "fact_type": self.fact_type,
            "section_path": self.section_path,
            "text": self.text,
            "score": self.score,
            "source_store_id": self.source_store_id,
            "source_full_text": self.source_full_text,
            "retrieval_role": self.retrieval_role,
            "answer_policy": self.answer_policy,
            "answer_risk_level": self.answer_risk_level,
            "budget_answer_enabled": self.budget_answer_enabled,
            "eligibility_answer_enabled": self.eligibility_answer_enabled,
            "payment_answer_enabled": self.payment_answer_enabled,
            "selection_stage": self.selection_stage,
            "is_backfilled": self.is_backfilled,
        }


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def has_any(text: str, keywords: Iterable[str]) -> bool:
    return any(normalize_text(keyword) in text for keyword in keywords)


def truncate_text(text: Any, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 20].rstrip() + " ...[truncated]"


def truncate_text_preserve_lines(text: Any, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 20].rstrip() + " ...[truncated]"


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_csv_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def is_generation_predictions_jsonl(path: str | Path) -> bool:
    path = Path(path)
    if path.suffix.lower() != ".jsonl":
        return False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            return isinstance(record.get("retrieved_contexts"), list)
    return False


def load_generation_predictions_jsonl(
    path: str | Path,
    *,
    experiment_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load nested generation_predictions JSONL as result/context rows.

    Retrieval notebooks may save one record per question:
    {"id": ..., "question": ..., "retrieved_contexts": [{rank, chunk_id, text, ...}]}.
    Generation code expects separate result_rows and context_rows, so this
    adapter flattens retrieved_contexts without changing the source file.
    """
    result_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for record in read_jsonl(path):
        question_id = str(record.get("id") or record.get("question_id") or "")
        question = str(record.get("question") or "")
        if not question_id:
            raise ValueError(f"generation prediction record is missing id/question_id: {path}")
        if not question:
            raise ValueError(f"generation prediction record is missing question: {question_id}")

        retrieved_docs_top5 = record.get("retrieved_docs_top5")
        if not retrieved_docs_top5:
            retrieved_docs_top5 = _top_unique_retrieved_docs(record.get("retrieved_contexts", []), top_k=5)

        result_rows.append(
            {
                "id": question_id,
                "question_id": question_id,
                "question": question,
                "answer": record.get("answer", ""),
                "ground_truth_answer": record.get("ground_truth_answer", ""),
                "ground_truth_docs": _jsonish_to_text(record.get("ground_truth_docs", "")),
                "retrieved_docs_top5": _jsonish_to_text(retrieved_docs_top5),
                "latency_ms": record.get("latency_ms", ""),
                "retrieval_ms": record.get("retrieval_ms", ""),
                "rerank_ms": record.get("rerank_ms", ""),
                "model_name": record.get("model_name", ""),
                "embedding_model": record.get("embedding_model", ""),
                "retriever_config": _jsonish_to_text(record.get("retriever_config", "")),
                "experiment_id": experiment_id,
            }
        )

        for context in record.get("retrieved_contexts", []):
            if not isinstance(context, dict):
                continue
            metadata = context.get("metadata") if isinstance(context.get("metadata"), dict) else {}
            context_rows.append(
                {
                    "question_id": question_id,
                    "id": question_id,
                    "question": question,
                    "experiment_id": experiment_id,
                    "rank": context.get("rank", ""),
                    "chunk_id": context.get("chunk_id") or metadata.get("chunk_id") or "",
                    "source_file": (
                        context.get("source_file")
                        or context.get("filename")
                        or metadata.get("source_file")
                        or metadata.get("source_file_nfc")
                        or ""
                    ),
                    "filename": context.get("filename", ""),
                    "doc_id": context.get("doc_id") or metadata.get("doc_id") or "",
                    "text": context.get("text") or context.get("content") or "",
                    "score": context.get("score", ""),
                    "rerank_score": context.get("rerank_score", ""),
                    "metadata": metadata,
                    "source_store_id": context.get("source_store_id", ""),
                    "selection_stage": context.get("selection_stage", ""),
                    "is_backfilled": context.get("is_backfilled", False),
                    "query_variant": context.get("query_variant", ""),
                    "query_variant_count": context.get("query_variant_count", ""),
                }
            )
    return result_rows, context_rows


def load_generation_input_rows(
    results_path: str | Path,
    contexts_path: str | Path,
    *,
    experiment_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contexts_path = Path(contexts_path)
    results_path = Path(results_path)
    if is_generation_predictions_jsonl(contexts_path):
        prediction_results, context_rows = load_generation_predictions_jsonl(
            contexts_path,
            experiment_id=experiment_id,
        )
        if results_path == contexts_path:
            return prediction_results, context_rows
        if results_path.suffix.lower() == ".csv" and results_path.exists():
            return read_csv_records(results_path), context_rows
        return prediction_results, context_rows

    if results_path.suffix.lower() == ".csv":
        result_rows = read_csv_records(results_path)
    elif is_generation_predictions_jsonl(results_path):
        result_rows, _ = load_generation_predictions_jsonl(
            results_path,
            experiment_id=experiment_id,
        )
    else:
        result_rows = read_jsonl(results_path)

    if contexts_path.suffix.lower() == ".csv":
        context_rows = read_csv_records(contexts_path)
    else:
        context_rows = read_jsonl(contexts_path)
    return result_rows, context_rows


def _top_unique_retrieved_docs(retrieved_contexts: Any, top_k: int = 5) -> list[str]:
    if not isinstance(retrieved_contexts, list):
        return []
    sorted_contexts = sorted(
        [context for context in retrieved_contexts if isinstance(context, dict)],
        key=lambda context: _safe_float(context.get("rank"), 9999.0),
    )
    docs: list[str] = []
    for context in sorted_contexts:
        metadata = context.get("metadata") if isinstance(context.get("metadata"), dict) else {}
        doc = (
            context.get("source_file")
            or context.get("filename")
            or metadata.get("source_file")
            or metadata.get("source_file_nfc")
            or context.get("doc_id")
            or metadata.get("doc_id")
            or ""
        )
        doc_text = str(doc).strip()
        if doc_text and doc_text not in docs:
            docs.append(doc_text)
        if len(docs) >= top_k:
            break
    return docs


def _jsonish_to_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def write_json(path: str | Path, data: dict[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_timestamped_output_dir(
    output_base_dir: str | Path,
    experiment_name: str,
    *,
    run_timestamp: str | None = None,
) -> Path:
    timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", experiment_name).strip("_")
    if not safe_name:
        raise ValueError("experiment_name must contain at least one safe character")
    output_dir = Path(output_base_dir) / f"{safe_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def inspect_jsonl_structure(
    path: str | Path,
    *,
    sample_size: int = 10,
    scan_limit: int = 1000,
) -> dict[str, Any]:
    records = _read_diverse_jsonl_sample(path, sample_size=sample_size, scan_limit=scan_limit)
    if not records:
        raise ValueError(f"no JSONL records found: {path}")

    top_keys = sorted({key for record in records for key in record.keys()})
    nested_keys = sorted(
        {
            nested_key
            for record in records
            for nested_key in _flatten_dict_keys(record)
        }
    )
    value_types: dict[str, list[str]] = {}
    missing_ratio: dict[str, float] = {}
    for key in nested_keys:
        values = [_get_by_path(record, key) for record in records]
        present_values = [value for value in values if value is not None]
        value_types[key] = sorted({_infer_value_type(value) for value in present_values})
        missing_ratio[key] = (len(records) - len(present_values)) / len(records)

    mapping = build_canonical_field_mapping(records)
    return {
        "path": str(path),
        "sample_size": len(records),
        "scan_limit": scan_limit,
        "top_level_keys": top_keys,
        "nested_keys": nested_keys,
        "value_types": value_types,
        "missing_ratio": missing_ratio,
        "canonical_field_mapping": mapping,
        "missing_canonical_fields": [
            field for field, info in mapping.items() if not info.get("path")
        ],
    }


def _read_diverse_jsonl_sample(
    path: str | Path,
    *,
    sample_size: int,
    scan_limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_chunk_types: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f, start=1):
            if line_index > scan_limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_type = str(record.get("chunk_type", ""))
            should_add = len(selected) < min(3, sample_size)
            if chunk_type and chunk_type not in seen_chunk_types:
                should_add = True
                seen_chunk_types.add(chunk_type)
            if should_add and len(selected) < sample_size:
                selected.append(record)
            if len(selected) >= sample_size and len(seen_chunk_types) >= 4:
                break
    return selected


def build_canonical_field_mapping(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}

    mapping: dict[str, Any] = {}
    for canonical_field, candidate_paths in CANONICAL_FIELD_CANDIDATES.items():
        selected_path = ""
        present_ratio = 0.0
        for candidate_path in candidate_paths:
            ratio = _path_present_ratio(records, candidate_path)
            if ratio > present_ratio:
                selected_path = candidate_path
                present_ratio = ratio
        mapping[canonical_field] = {
            "path": selected_path if present_ratio > 0 else "",
            "present_ratio": round(present_ratio, 4),
            "candidate_paths": candidate_paths,
        }

    if not mapping["fact_candidates"]["path"] and _has_fact_candidate_records(records):
        mapping["fact_candidates"] = {
            "path": "content",
            "present_ratio": 1.0,
            "candidate_paths": ["chunk_type=fact_candidates -> content"],
            "note": "fact_candidates records were detected by chunk_type.",
        }
    return mapping


def _flatten_dict_keys(record: dict[str, Any], prefix: str = "") -> list[str]:
    keys: list[str] = []
    for key, value in record.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        keys.append(path)
        if isinstance(value, dict):
            keys.extend(_flatten_dict_keys(value, path))
    return keys


def _get_by_path(record: dict[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _path_present_ratio(records: list[dict[str, Any]], path: str) -> float:
    present = sum(_get_by_path(record, path) is not None for record in records)
    return present / len(records) if records else 0.0


def _infer_value_type(value: Any) -> str:
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null"
    return "str"


def _has_fact_candidate_records(records: list[dict[str, Any]]) -> bool:
    return any(str(record.get("chunk_type", "")) == "fact_candidates" for record in records)


def classify_question(question: str) -> dict[str, Any]:
    q = normalize_text(question)
    question_types = [
        qtype for qtype, keywords in QUESTION_KEYWORDS.items() if has_any(q, keywords)
    ]
    is_total_allocation = _looks_like_total_allocation(q)
    is_technical_requirement = _looks_like_technical_requirement(q)
    is_technical_purpose = _looks_like_technical_purpose(q)
    is_risk_comparison = _looks_like_risk_comparison(q)
    if "budget" not in question_types and AMOUNT_RE.search(str(question or "")):
        budget_operation_markers = (
            BUDGET_DIFFERENCE_KEYWORDS
            + BUDGET_SUM_KEYWORDS
            + BUDGET_RATIO_KEYWORDS
            + ["계산", "산술", "합산", "액수", "금액", "단가", "월급", "예산"]
        )
        if has_any(q, budget_operation_markers):
            question_types.append("budget")
    if is_total_allocation and "budget" not in question_types:
        question_types.append("budget")

    if _looks_like_purpose_summary(q) and "requirements" not in question_types:
        question_types.append("requirements")
    if (is_technical_requirement or is_technical_purpose or is_risk_comparison) and "requirements" not in question_types:
        question_types.append("requirements")
    if _looks_like_negative_check(q) and not is_technical_requirement and "requirements" not in question_types:
        question_types.append("requirements")

    if "duration" in question_types:
        if is_technical_purpose and not _looks_like_explicit_period_question(q):
            question_types = [qtype for qtype in question_types if qtype != "duration"]
            period_subtypes = []
        else:
            period_subtypes = [
                subtype
                for subtype, keywords in PERIOD_SUBTYPE_KEYWORDS.items()
                if has_any(q, keywords)
            ]
            if not period_subtypes:
                period_subtypes = ["project_duration", "submission_deadline", "maintenance_period"]
    else:
        period_subtypes = []

    if "multi_doc" not in question_types and (_looks_like_multi_doc(question) or is_risk_comparison):
        question_types.append("multi_doc")

    if not question_types:
        question_types = ["general"]

    target_slots = _extract_target_slots(question)
    scenario_terms = _extract_scenario_terms(question)
    intent_slots = _infer_intent_slots(question, question_types)
    intent_plan = _build_intent_plan(
        question,
        question_types,
        intent_slots,
        target_slots,
        period_subtypes,
    )
    answer_type = _infer_answer_type_from_intents(question_types, intent_slots)
    needs_synthesis = bool(set(question_types) & SYNTHESIS_TYPES) or any(
        intent in intent_slots
        for intent in {
            "purpose_summary",
            "requirements_summary",
            "technical_requirement_lookup",
            "technical_purpose_summary",
            "risk_comparison",
            "multi_doc_comparison",
        }
    )
    return {
        "question_types": question_types,
        "period_subtypes": period_subtypes,
        "is_multi_doc": "multi_doc" in question_types,
        "is_multi_intent": len(intent_plan) > 1,
        "needs_synthesis": needs_synthesis,
        "answer_type": answer_type,
        "intent_slots": intent_slots,
        "intent_plan": intent_plan,
        "target_slots": target_slots,
        "scenario_terms": scenario_terms,
    }


def _looks_like_multi_doc(question: str) -> bool:
    # Conservative heuristic: explicit comparison markers or connectors between
    # two institution-like entities. Avoid treating "예산과 사업기간" as multi-doc.
    q = normalize_text(question)
    explicit_markers = [
        "비교",
        "각각",
        "둘 다",
        "차이",
        "두 사업",
        "두 문서",
        "간의",
        "차액",
        "합계",
        "공통",
    ]
    if any(marker in q for marker in explicit_markers):
        return True
    has_connector = bool(re.search(r"(와|과|랑|그리고)", q))
    org_markers = ["대학교", "공사", "공단", "재단", "연구원", "협회", "시청", "구청", "기관", "센터"]
    org_count = sum(q.count(marker) for marker in org_markers)
    return has_connector and org_count >= 2


def _infer_answer_type(question_types: list[str]) -> str:
    if "multi_doc" in question_types:
        for concrete_type in [
            "budget",
            "duration",
            "bid_deadline",
            "submission_documents",
            "eligibility",
            "business_type",
            "evaluation",
            "requirements",
        ]:
            if concrete_type in question_types:
                return concrete_type
        return "multi_doc_comparison"
    concrete_types = [qtype for qtype in question_types if qtype != "general"]
    if len(concrete_types) > 1:
        return "summary"
    priority = [
        "budget",
        "duration",
        "bid_deadline",
        "submission_documents",
        "submission_logistics",
        "eligibility",
        "business_type",
        "requirements",
        "evaluation",
        "general",
    ]
    for qtype in priority:
        if qtype in question_types:
            return qtype
    return "unknown"


def _infer_answer_type_from_intents(question_types: list[str], intent_slots: list[str]) -> str:
    has_budget_intent = any(
        intent in intent_slots
        for intent in {"budget_lookup", "budget_difference", "budget_sum", "budget_ratio", "total_allocation_lookup"}
    )
    has_summary_intent = bool(
        {"purpose_summary", "requirements_summary", "technical_requirement_lookup", "technical_purpose_summary", "risk_comparison"}
        & set(intent_slots)
    )
    if has_budget_intent and not has_summary_intent:
        return "budget"
    if "purpose_summary" in intent_slots or "requirements_summary" in intent_slots:
        return "summary"
    if "technical_requirement_lookup" in intent_slots:
        return "requirements"
    if "technical_purpose_summary" in intent_slots or "risk_comparison" in intent_slots:
        return "summary"
    if "multi_doc_comparison" in intent_slots and "budget" not in question_types:
        return "multi_doc_comparison"
    return _infer_answer_type(question_types)


def _looks_like_purpose_summary(q: str) -> bool:
    return has_any(q, PURPOSE_SUMMARY_KEYWORDS)


def _looks_like_negative_check(q: str) -> bool:
    return has_any(q, NEGATIVE_CHECK_KEYWORDS)


def _looks_like_total_allocation(q: str) -> bool:
    return has_any(q, TOTAL_ALLOCATION_KEYWORDS)


def _looks_like_technical_requirement(q: str) -> bool:
    return has_any(q, TECHNICAL_REQUIREMENT_KEYWORDS)


def _looks_like_technical_purpose(q: str) -> bool:
    return has_any(q, TECHNICAL_PURPOSE_KEYWORDS)


def _looks_like_risk_comparison(q: str) -> bool:
    return has_any(q, RISK_COMPARISON_KEYWORDS)


def _looks_like_explicit_period_question(q: str) -> bool:
    return has_any(
        q,
        [
            "사업기간",
            "수행기간",
            "계약기간",
            "제출기한",
            "제출 기간",
            "입찰마감",
            "마감일",
            "언제까지",
            "몇 개월",
            "몇개월",
            "몇 일",
            "몇일",
        ],
    )


def _infer_intent_slots(question: str, question_types: list[str]) -> list[str]:
    q = normalize_text(question)
    intents: list[str] = []
    if "budget" in question_types:
        if _looks_like_total_allocation(q):
            intents.append("total_allocation_lookup")
        elif has_any(q, BUDGET_SUM_KEYWORDS):
            intents.append("budget_sum")
        elif has_any(q, BUDGET_DIFFERENCE_KEYWORDS):
            intents.append("budget_difference")
        elif has_any(q, BUDGET_RATIO_KEYWORDS):
            intents.append("budget_ratio")
        else:
            intents.append("budget_lookup")
    if "duration" in question_types:
        intents.append("duration_lookup")
    if "submission_documents" in question_types:
        intents.append("submission_documents")
    if "submission_logistics" in question_types:
        intents.append("submission_logistics")
    technical_requirement = _looks_like_technical_requirement(q)
    technical_purpose = _looks_like_technical_purpose(q)
    risk_comparison = _looks_like_risk_comparison(q)
    if ("eligibility" in question_types or _looks_like_negative_check(q)) and not technical_requirement and not technical_purpose:
        intents.append("negative_check" if _looks_like_negative_check(q) else "eligibility_check")
    has_derived_budget_intent = any(
        intent in intents for intent in {"budget_difference", "budget_sum", "budget_ratio"}
    )
    if technical_requirement:
        intents.append("technical_requirement_lookup")
    elif technical_purpose:
        intents.append("technical_purpose_summary")
    elif _looks_like_purpose_summary(q):
        intents.append("purpose_summary")
    elif "requirements" in question_types and "budget" not in question_types and not has_derived_budget_intent and "negative_check" not in intents:
        intents.append("requirements_summary")
    if risk_comparison:
        intents.append("risk_comparison")
    if "requirements" in question_types and has_any(q, LIST_QUESTION_KEYWORDS):
        intents.append("requirements_list")
    if "multi_doc" in question_types:
        intents.append("multi_doc_comparison")
    return _unique_preserve_order(intents or ["general"])


def _build_intent_plan(
    question: str,
    question_types: list[str],
    intent_slots: list[str],
    target_slots: list[dict[str, Any]],
    period_subtypes: list[str],
) -> list[dict[str, Any]]:
    q = normalize_text(question)
    target_labels = [slot.get("target_label", "") for slot in target_slots if slot.get("target_label")]
    plans: list[dict[str, Any]] = []
    for index, intent in enumerate(intent_slots, start=1):
        required_fact_types = _required_fact_types_for_intent(intent, period_subtypes)
        plan = {
            "intent_id": f"I{index:02d}",
            "intent": intent,
            "answer_section": INTENT_ANSWER_SECTIONS.get(intent, intent),
            "targets": target_labels,
            "target_policy": _intent_target_policy(intent, question_types, target_labels),
            "required_fact_types": required_fact_types,
            "preferred_chunk_types": INTENT_PREFERRED_CHUNK_TYPES.get(intent, ["text", "table", "fact_candidates"]),
            "requires_computation": intent in {"budget_difference", "budget_sum", "budget_ratio"},
            "requires_all_targets": intent == "multi_doc_comparison" or ("multi_doc" in question_types and bool(target_labels)),
            "classification_signals": _intent_classification_signals(q, intent),
        }
        plans.append(plan)
    return plans or [
        {
            "intent_id": "I01",
            "intent": "general",
            "answer_section": INTENT_ANSWER_SECTIONS["general"],
            "targets": target_labels,
            "target_policy": _intent_target_policy("general", question_types, target_labels),
            "required_fact_types": INTENT_REQUIRED_FACT_TYPES["general"],
            "preferred_chunk_types": INTENT_PREFERRED_CHUNK_TYPES["general"],
            "requires_computation": False,
            "requires_all_targets": False,
            "classification_signals": [],
        }
    ]


def _required_fact_types_for_intent(intent: str, period_subtypes: list[str]) -> list[str]:
    if intent == "duration_lookup" and period_subtypes:
        return _unique_preserve_order(period_subtypes)
    return list(INTENT_REQUIRED_FACT_TYPES.get(intent, []))


def _intent_target_policy(intent: str, question_types: list[str], target_labels: list[str]) -> str:
    if intent in {"budget_difference", "budget_sum", "multi_doc_comparison", "risk_comparison"}:
        return "per_target_required"
    if "multi_doc" in question_types and target_labels:
        return "per_target_preferred"
    if target_labels:
        return "single_target_preferred"
    return "context_scope"


def _intent_classification_signals(q: str, intent: str) -> list[str]:
    keyword_map = {
        "budget_difference": BUDGET_DIFFERENCE_KEYWORDS,
        "budget_sum": BUDGET_SUM_KEYWORDS,
        "budget_ratio": BUDGET_RATIO_KEYWORDS,
        "purpose_summary": PURPOSE_SUMMARY_KEYWORDS,
        "requirements_summary": QUESTION_KEYWORDS["requirements"],
        "requirements_list": LIST_QUESTION_KEYWORDS,
        "negative_check": NEGATIVE_CHECK_KEYWORDS,
        "multi_doc_comparison": QUESTION_KEYWORDS["multi_doc"],
        "submission_documents": QUESTION_KEYWORDS["submission_documents"],
        "submission_logistics": QUESTION_KEYWORDS["submission_logistics"],
        "eligibility_check": QUESTION_KEYWORDS["eligibility"],
        "duration_lookup": QUESTION_KEYWORDS["duration"],
        "budget_lookup": QUESTION_KEYWORDS["budget"],
        "total_allocation_lookup": TOTAL_ALLOCATION_KEYWORDS,
        "technical_requirement_lookup": TECHNICAL_REQUIREMENT_KEYWORDS,
        "technical_purpose_summary": TECHNICAL_PURPOSE_KEYWORDS,
        "risk_comparison": RISK_COMPARISON_KEYWORDS,
    }
    return _matched_keywords(q, keyword_map.get(intent, []))


def _matched_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    normalized = normalize_text(text)
    return [keyword for keyword in keywords if normalize_text(keyword) in normalized]


def _extract_target_slots(question: str) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()
    quote_patterns = [
        r"([^\n]{0,45}?)의\s*'([^']{3,120})'",
        r'([^\n]{0,45}?)의\s*"([^"]{3,120})"',
        r"([^\n]{0,45}?)의\s*「([^」]{3,120})」",
        r"'([^']{3,120})'",
        r'"([^"]{3,120})"',
        r"「([^」]{3,120})」",
    ]
    for pattern in quote_patterns:
        for match in re.finditer(pattern, question):
            if len(match.groups()) == 2:
                issuer = _clean_issuer_hint(match.group(1))
                project = _clean_target_label(match.group(2))
                label = f"{issuer} {project}".strip()
            else:
                issuer = ""
                project = _clean_target_label(match.group(1))
                label = project
            if _is_scenario_target_label(label):
                continue
            key = _normalize_doc_key(label)
            if len(key) < 3 or key in seen:
                continue
            seen.add(key)
            slots.append(
                {
                    "target_label": label,
                    "issuer_hint": issuer,
                    "project_hint": project,
                    "target_tokens": _target_tokens(label),
                    "matched_source_file": "",
                    "match_score": 0.0,
                    "required_fields": [],
                    "missing_fields": [],
                }
            )
    for label in _extract_unquoted_project_targets(question):
        if _is_scenario_target_label(label):
            continue
        key = _normalize_doc_key(label)
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        slots.append(
            {
                "target_label": label,
                "issuer_hint": "",
                "project_hint": label,
                "target_tokens": _target_tokens(label),
                "matched_source_file": "",
                "match_score": 0.0,
                "required_fields": [],
                "missing_fields": [],
            }
        )
    parenthesized_amount_pattern = re.compile(
        r"([가-힣A-Za-z0-9㈜㈔&()·\s]{3,90}?)\s*\(\s*(?:"
        + AMOUNT_RE.pattern
        + r")\s*\)"
    )
    for match in parenthesized_amount_pattern.finditer(question):
        label = _clean_unquoted_budget_target_label(match.group(1))
        if not _is_plausible_unquoted_target_label(label):
            continue
        if _is_scenario_target_label(label):
            continue
        key = _normalize_doc_key(label)
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        slots.append(
            {
                "target_label": label,
                "issuer_hint": "",
                "project_hint": label,
                "target_tokens": _target_tokens(label),
                "matched_source_file": "",
                "match_score": 0.0,
                "required_fields": [],
                "missing_fields": [],
            }
        )

    budget_target_pattern = re.compile(
        r"([가-힣A-Za-z0-9㈜㈔&()·\s]{3,90}?)(?:의\s*)?(?:사업\s*)?(?:예산|에산|사업비|발주비|투입\s*금액|자금\s*예산|자금\s*에산)"
    )
    for match in budget_target_pattern.finditer(question):
        label = _clean_unquoted_budget_target_label(match.group(1))
        if not _is_plausible_unquoted_target_label(label):
            continue
        if _is_scenario_target_label(label):
            continue
        key = _normalize_doc_key(label)
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        slots.append(
            {
                "target_label": label,
                "issuer_hint": "",
                "project_hint": label,
                "target_tokens": _target_tokens(label),
                "matched_source_file": "",
                "match_score": 0.0,
                "required_fields": [],
                "missing_fields": [],
            }
        )
    return _dedupe_target_slots(slots)


def _extract_unquoted_project_targets(question: str) -> list[str]:
    patterns = [
        r"([가-힣A-Za-z0-9()·\-\s]{2,45}?)의\s+([가-힣A-Za-z0-9()·\-\s]{3,90}?(?:사업|시스템|고도화|개량|구축|용역|관개망|플랫폼|교육망))(?=\s*(?:에서|에|쪽|관련|관해|의|은|는|,|\.|\?))",
        r"((?:한국|부산|대구|대전|광주|서울|인천|코이카|KOICA|그랜드|고려|철도|수자원|보건|연구|관광|대검찰청|검찰청)[가-힣A-Za-z0-9()·\-\s]{3,100}?(?:사업|시스템|고도화|개량|구축|용역|관개망|플랫폼|교육망))(?=\s*(?:에서|에|쪽|관련|관해|의|은|는|,|\.|\?))",
    ]
    labels: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, question):
            if len(match.groups()) == 2:
                issuer = _clean_issuer_hint(match.group(1))
                project = _clean_target_label(match.group(2))
                label = f"{issuer} {project}".strip()
            else:
                label = _clean_target_label(match.group(1))
            if _is_plausible_unquoted_target_label(label):
                labels.append(label)
    return _unique_preserve_order(labels)


def _dedupe_target_slots(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    norm_labels: list[str] = []
    for slot in slots:
        label = str(slot.get("target_label", "")).strip()
        norm = _normalize_doc_key(label)
        loose_norm = norm.replace("의", "").replace("코레일", "")
        if not norm:
            continue
        if any(norm in existing or existing in norm or loose_norm in existing or existing in loose_norm for existing in norm_labels):
            continue
        deduped.append(slot)
        norm_labels.append(loose_norm or norm)
    return deduped


def _is_scenario_target_label(label: str) -> bool:
    normalized = normalize_text(label)
    if not normalized:
        return False
    if "/" in str(label or "") and has_any(normalized, ["콘텐츠", "성격", "관리하", "대면형"]):
        return True
    if has_any(normalized, SCENARIO_TARGET_BLOCK_KEYWORDS):
        return True
    if has_any(normalized, TECHNICAL_REQUIREMENT_KEYWORDS + TECHNICAL_PURPOSE_KEYWORDS):
        org_markers = [
            "대학교", "공사", "공단", "재단", "연구원", "진흥원", "협회", "시청", "구청",
            "센터", "코이카", "koica", "관광공사",
        ]
        if not has_any(normalized, org_markers):
            return True
    return False


def _extract_scenario_terms(question: str) -> list[str]:
    q = normalize_text(question)
    terms = [
        keyword
        for keyword in RISK_COMPARISON_KEYWORDS + TECHNICAL_REQUIREMENT_KEYWORDS
        if normalize_text(keyword) in q
    ]
    quoted_terms = []
    for match in re.finditer(r"'([^']{3,120})'|\"([^\"]{3,120})\"", question):
        value = match.group(1) or match.group(2) or ""
        if _is_scenario_target_label(value):
            quoted_terms.append(_clean_target_label(value))
    return _unique_preserve_order(quoted_terms + terms)


def _clean_unquoted_budget_target_label(value: str) -> str:
    label = str(value or "")
    label = re.split(r"[.?!]|(?:그리고)|(?:그러고)|(?:다음)|(?:후)|(?:대하여)", label)[-1]
    label = re.split(r"(?:에서\s+|(?:의\s*)?(?:경우|관련해서|관련하여|구체적인|구체적|정확히|대략|어느 정도|얼마|얼말|잡혀|있는지|기재))", label)[0]
    label = _clean_target_label(label)
    label = re.sub(r"^(?:과|와|및|그리고|또는|혹은)\s*", "", label)
    label = re.sub(r"^(?:이들|해당|두|각)\s*", "", label)
    label = re.sub(r"^(?:그|저|이|해당)\s+", "", label)
    label = re.sub(r"\s+", " ", label)
    return label.strip(" :;,.()[]")


def _is_plausible_unquoted_target_label(label: str) -> bool:
    label = str(label or "").strip()
    norm = normalize_text(label)
    if len(_normalize_doc_key(label)) < 6:
        return False
    if AMOUNT_RE.search(label):
        return False
    generic_blockers = ["전체", "남은", "잔여", "최종", "액수", "자금", "수수료", "월급", "단가"]
    if any(blocker in norm for blocker in generic_blockers):
        return False
    anchor_markers = [
        "공사", "공단", "은행", "광역시", "기술원", "연구원", "대학교", "재단", "협회",
        "시청", "구청", "센터", "기관", "시스템", "플랫폼", "용역", "사업", "조달",
        "코이카", "koica", "관개망", "진흥원", "관광공사", "검찰청", "교육망", "플랫폼",
    ]
    return any(marker in norm for marker in anchor_markers)


def _clean_issuer_hint(value: str) -> str:
    value = AMOUNT_RE.sub("", str(value or ""))
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.split(r"(?:예산|사업비|발주비|투입\s*금액)", value)[-1]
    parts = re.split(r"\s*(?:과|와|및|그리고|또는|혹은)\s+", value)
    value = parts[-1] if parts else value
    return _clean_target_label(value)


def _clean_target_label(value: str) -> str:
    value = re.sub(r"[\s,]*(?:과|와|및|그리고|또는|혹은)$", "", str(value or "").strip())
    value = re.sub(r"^(?:과|와|및|그리고|또는|혹은)\s*", "", value)
    value = re.sub(r"^(?:그렇다면|그러면|그럼)\s*", "", value)
    value = re.sub(r"^(?:그|저|이|해당)\s+", "", value)
    return value.strip(" :;,.()[]")


def _target_tokens(value: str) -> list[str]:
    compact = _normalize_doc_key(value)
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9]+", str(value or ""))
    tokens = [token.casefold() for token in raw_tokens if len(token) >= 2]
    for size in [4, 6, 8, 10]:
        tokens.extend(compact[idx : idx + size] for idx in range(0, max(len(compact) - size + 1, 0), size))
    stopwords = {
        "사업", "용역", "구축", "시스템", "정보", "한국", "공사", "재공고", "긴급",
        "구체적인", "구체적", "경우", "기재", "있는지", "잡혀", "대략", "예산",
        "비용", "규모", "자금", "얼마", "얼말", "관련", "관련해서",
    }
    return _unique_preserve_order(token for token in tokens if token and token not in stopwords)


def _normalize_doc_key(value: Any) -> str:
    value = unicodedata.normalize("NFC", str(value or "")).casefold()
    value = re.sub(r"\.(?:hwp|hwpx|pdf|docx?)$", "", value)
    return re.sub(r"[^0-9a-z가-힣]+", "", value)


def _doc_match_text(row: dict[str, Any] | None = None, chunk: dict[str, Any] | None = None, block: dict[str, Any] | EvidenceBlock | None = None) -> str:
    values: list[str] = []
    for obj in [row or {}, chunk or {}]:
        if isinstance(obj, dict):
            metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
            values.extend(
                str(v)
                for v in [
                    obj.get("source_file"),
                    obj.get("source_file_nfc"),
                    obj.get("doc_key"),
                    obj.get("canonical_doc_key"),
                    obj.get("project_name"),
                    obj.get("issuer"),
                    metadata.get("source_file"),
                    metadata.get("source_file_nfc"),
                    metadata.get("doc_key"),
                    metadata.get("canonical_doc_key"),
                    metadata.get("project_name"),
                    metadata.get("issuer"),
                ]
                if v
            )
    if block is not None:
        getter = block.get if isinstance(block, dict) else lambda key, default="": getattr(block, key, default)
        values.extend(str(v) for v in [getter("source_file"), getter("source_file_nfc"), getter("text")] if v)
    return " ".join(values)


def _best_target_match_score(value: str, target_slots: list[dict[str, Any]]) -> float:
    if not target_slots:
        return 0.0
    norm_value = _normalize_doc_key(value)
    best = 0.0
    for slot in target_slots:
        label_key = _normalize_doc_key(slot.get("target_label", ""))
        tokens = slot.get("target_tokens") or _target_tokens(slot.get("target_label", ""))
        if label_key and (label_key in norm_value or norm_value in label_key):
            best = max(best, 1.0)
            continue
        if not tokens:
            continue
        hits = sum(1 for token in tokens if token and token in norm_value)
        score = hits / max(len(tokens), 1)
        issuer = _normalize_doc_key(slot.get("issuer_hint", ""))
        project = _normalize_doc_key(slot.get("project_hint", ""))
        if issuer and issuer in norm_value:
            score += 0.12
        if project and project in norm_value:
            score += 0.35
        best = max(best, min(score, 1.0))
    return best


def _match_target_slots_to_blocks(
    target_slots: list[dict[str, Any]],
    blocks: list[EvidenceBlock],
    analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    analysis = analysis or {}
    required_fields = _required_fields_from_intents(analysis)
    matched = []
    for slot in target_slots:
        best_block = None
        best_score = 0.0
        slot_blocks: list[EvidenceBlock] = []
        for block in blocks:
            score = _best_target_match_score(_doc_match_text(block=block), [slot])
            if score > best_score:
                best_score = score
                best_block = block
            if score >= TARGET_MATCH_THRESHOLD:
                slot_blocks.append(block)
        next_slot = dict(slot)
        next_slot["match_score"] = round(best_score, 4)
        matched_source_file = best_block.source_file if best_block and best_score >= TARGET_MATCH_THRESHOLD else ""
        next_slot["matched_source_file"] = matched_source_file
        next_slot["required_fields"] = required_fields
        missing_fields = []
        if matched_source_file and "project_budget" in required_fields and not _has_project_budget_operand(slot_blocks):
            missing_fields.append("project_budget")
        if matched_source_file and "total_allocation" in required_fields and not _has_total_allocation_operand(slot_blocks):
            missing_fields.append("total_allocation")
        next_slot["missing_fields"] = missing_fields
        matched.append(next_slot)
    return matched


def _required_fields_from_intents(analysis: dict[str, Any]) -> list[str]:
    intents = set(analysis.get("intent_slots", []))
    required_fields = []
    if "total_allocation_lookup" in intents:
        required_fields.append("total_allocation")
    if any(intent in intents for intent in {"budget_lookup", "budget_difference", "budget_sum", "budget_ratio"}):
        required_fields.append("project_budget")
    if "duration_lookup" in intents:
        required_fields.extend(analysis.get("period_subtypes", []) or ["project_duration"])
    if "submission_documents" in intents:
        required_fields.append("submission_documents")
    if "eligibility_check" in intents:
        required_fields.append("eligibility")
    return _unique_preserve_order(required_fields)


def _fact_types_from_intent_plan(analysis: dict[str, Any]) -> set[str]:
    fact_types: set[str] = set()
    for plan in analysis.get("intent_plan", []) or []:
        for fact_type in plan.get("required_fact_types", []) or []:
            if fact_type not in {"text", "table"}:
                fact_types.add(str(fact_type))
    return fact_types


def load_chunk_index(
    chunks_path: str | Path,
    chunk_ids: set[str] | None = None,
    *,
    source_files: set[str] | None = None,
    fact_types: set[str] | None = None,
    embed_enabled_only: bool = False,
) -> dict[str, dict[str, Any]]:
    chunks_path = Path(chunks_path)
    index: dict[str, dict[str, Any]] = {}
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")

    source_file_keys = {_normalize_doc_key(value) for value in (source_files or set()) if value}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = str(record.get("chunk_id", ""))
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            source_key = _normalize_doc_key(record.get("source_file") or metadata.get("source_file") or "")
            selected_by_chunk = chunk_ids is None or chunk_id in chunk_ids
            selected_by_source = bool(source_file_keys and source_key in source_file_keys)
            if not selected_by_chunk and not selected_by_source:
                continue
            if selected_by_source and not selected_by_chunk and fact_types:
                fact_type = str(record.get("fact_type") or metadata.get("fact_type") or "")
                if fact_type not in fact_types:
                    continue
            if embed_enabled_only and record.get("embed_enabled") is False:
                continue
            index[chunk_id] = record
    return index


def load_source_store_index(
    source_store_path: str | Path,
    source_store_ids: set[str] | None = None,
    *,
    enabled: bool = False,
) -> dict[str, dict[str, Any]]:
    if not enabled:
        return {}
    source_store_path = Path(source_store_path)
    if not source_store_path.exists():
        return {}

    index: dict[str, dict[str, Any]] = {}
    with source_store_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            source_store_id = str(record.get("source_store_id", ""))
            if source_store_ids is not None and source_store_id not in source_store_ids:
                continue
            index[source_store_id] = record
            if source_store_ids is not None and len(index) >= len(source_store_ids):
                break
    return index


def prepare_generation_items(
    result_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    *,
    experiment_id: str,
    sample_size: int | None,
    review_focus: bool = True,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    filtered_results = [
        row for row in result_rows if str(row.get("experiment_id", "")) == experiment_id
    ]
    filtered_contexts = [
        row for row in context_rows if str(row.get("experiment_id", "")) == experiment_id
    ]

    contexts_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered_contexts:
        contexts_by_question[str(row.get("question_id", ""))].append(row)
    for rows in contexts_by_question.values():
        rows.sort(key=lambda item: _safe_float(item.get("rank"), 9999.0))

    if sample_size and sample_size < len(filtered_results):
        filtered_results = _select_review_rows(
            filtered_results,
            sample_size=sample_size,
            review_focus=review_focus,
            random_seed=random_seed,
        )

    items: list[dict[str, Any]] = []
    for row in filtered_results:
        question_id = str(row.get("id", ""))
        items.append(
            {
                "question_id": question_id,
                "question": row.get("question", ""),
                "result": row,
                "retrieved_contexts": contexts_by_question.get(question_id, []),
            }
        )
    return items


def _select_review_rows(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    review_focus: bool,
    random_seed: int,
) -> list[dict[str, Any]]:
    if not review_focus:
        rng = random.Random(random_seed)
        selected = rows[:]
        rng.shuffle(selected)
        return selected[:sample_size]

    def is_focus(row: dict[str, Any]) -> bool:
        return any(
            _safe_float(row.get(col), 0.0) > 0
            for col in [
                "candidate_generation_failed_top10",
                "partial_multi_doc_loss",
                "low_rank_correct",
            ]
        ) or _safe_float(row.get("hit_at_5"), 1.0) == 0

    focused = [row for row in rows if is_focus(row)]
    remainder = [row for row in rows if not is_focus(row)]
    rng = random.Random(random_seed)
    rng.shuffle(remainder)
    selected = (focused + remainder)[:sample_size]
    selected.sort(key=lambda row: str(row.get("id", "")))
    return selected


def build_context_package(
    question: str,
    retrieved_contexts: list[dict[str, Any]],
    *,
    chunk_index: dict[str, dict[str, Any]] | None = None,
    source_store_index: dict[str, dict[str, Any]] | None = None,
    use_source_store: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_GENERATION_CONFIG, **(config or {})}
    analysis = classify_question(question)
    chunk_index = chunk_index or {}
    source_store_index = source_store_index or {}
    evidence = _build_evidence_blocks(
        retrieved_contexts,
        analysis,
        chunk_index=chunk_index,
        source_store_index=source_store_index,
        use_source_store=use_source_store,
        config=cfg,
    )
    evidence = _expand_same_source_fact_blocks(evidence, analysis, chunk_index, config=cfg)
    analysis = dict(analysis)
    analysis["target_slots"] = _match_target_slots_to_blocks(
        analysis.get("target_slots", []),
        evidence,
        analysis,
    )
    analysis["intent_plan"] = _build_intent_plan(
        question,
        analysis.get("question_types", []),
        analysis.get("intent_slots", []),
        analysis.get("target_slots", []),
        analysis.get("period_subtypes", []),
    )
    computed_values = _compute_deterministic_values(question, analysis, evidence)
    analysis["computed_values"] = computed_values

    max_blocks = (
        cfg["max_blocks_synthesis"]
        if analysis["needs_synthesis"]
        else cfg["max_blocks_fact"]
    )
    max_chars = (
        cfg["max_context_chars_synthesis"]
        if analysis["needs_synthesis"]
        else cfg["max_context_chars_fact"]
    )

    selected = _select_evidence_blocks(evidence, analysis, max_blocks=max_blocks)
    core_summary = _build_core_summary(selected, analysis)
    core_summary["target_slots"] = analysis.get("target_slots", [])
    core_summary["intent_slots"] = analysis.get("intent_slots", [])
    core_summary["intent_plan"] = analysis.get("intent_plan", [])
    core_summary["intent_evidence"] = _build_intent_evidence_groups(selected, analysis)
    core_summary["computed_values"] = computed_values
    context_text = _format_context_text(core_summary, selected, analysis, max_chars=max_chars)

    failure_tags = []
    if not retrieved_contexts:
        failure_tags.append("retrieval_missing")
    if not selected:
        failure_tags.append("insufficient_evidence")
    if use_source_store and not source_store_index:
        failure_tags.append("source_store_unavailable")

    return {
        "question": question,
        "question_analysis": analysis,
        "core_summary": core_summary,
        "evidence_blocks": [block.to_dict() for block in selected],
        "context_text": context_text,
        "failure_tags": failure_tags,
        "use_source_store": bool(use_source_store and source_store_index),
    }


def _build_evidence_blocks(
    retrieved_contexts: list[dict[str, Any]],
    analysis: dict[str, Any],
    *,
    chunk_index: dict[str, dict[str, Any]],
    source_store_index: dict[str, dict[str, Any]],
    use_source_store: bool,
    config: dict[str, Any],
) -> list[EvidenceBlock]:
    blocks: list[EvidenceBlock] = []
    for row in retrieved_contexts:
        chunk_id = str(row.get("chunk_id", ""))
        chunk = chunk_index.get(chunk_id, {})
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        source_ref = chunk.get("source_ref") if isinstance(chunk.get("source_ref"), dict) else {}
        source_store_id = str(source_ref.get("source_store_id") or row.get("source_store_id") or "")
        source_record = source_store_index.get(source_store_id, {}) if use_source_store else {}
        source_full_text = ""
        if source_record:
            source_full_text = truncate_text(
                source_record.get("full_text") or source_record.get("text") or "",
                int(config["source_store_text_chars"]),
            )

        text = (
            chunk.get("evidence_text_short")
            or chunk.get("content")
            or row.get("text")
            or ""
        )
        source_file = (
            chunk.get("source_file")
            or metadata.get("source_file")
            or row.get("source_file")
            or row.get("filename")
            or ""
        )
        rank = int(_safe_float(row.get("rank"), 9999.0))
        block = EvidenceBlock(
            source_file=str(source_file),
            chunk_id=chunk_id,
            rank=rank,
            chunk_type=str(chunk.get("chunk_type") or metadata.get("chunk_type") or row.get("chunk_type") or ""),
            fact_type=str(chunk.get("fact_type") or metadata.get("fact_type") or row.get("fact_type") or ""),
            section_path=str(metadata.get("section_path") or chunk.get("section_path") or row.get("section_path") or ""),
            text=truncate_text(text, int(config["evidence_text_chars"])),
            score=_score_evidence(row, chunk, analysis),
            source_store_id=source_store_id,
            source_full_text=source_full_text,
            source_file_nfc=str(chunk.get("source_file_nfc") or metadata.get("source_file_nfc") or source_file),
            evidence_id=str(chunk.get("evidence_id") or metadata.get("evidence_id") or ""),
            retrieval_role=str(chunk.get("retrieval_role") or metadata.get("retrieval_role") or row.get("retrieval_role") or ""),
            answer_policy=str(chunk.get("answer_policy") or metadata.get("answer_policy") or row.get("answer_policy") or ""),
            answer_risk_level=str(chunk.get("answer_risk_level") or metadata.get("answer_risk_level") or row.get("answer_risk_level") or ""),
            budget_answer_enabled=_as_bool(chunk.get("budget_answer_enabled") or metadata.get("budget_answer_enabled") or row.get("budget_answer_enabled")),
            eligibility_answer_enabled=_as_bool(chunk.get("eligibility_answer_enabled") or metadata.get("eligibility_answer_enabled") or row.get("eligibility_answer_enabled")),
            payment_answer_enabled=_as_bool(chunk.get("payment_answer_enabled") or metadata.get("payment_answer_enabled") or row.get("payment_answer_enabled")),
            selection_stage=str(row.get("selection_stage") or ""),
            is_backfilled=_as_bool(row.get("is_backfilled")),
        )
        blocks.append(block)
    return blocks


def _expand_same_source_fact_blocks(
    blocks: list[EvidenceBlock],
    analysis: dict[str, Any],
    chunk_index: dict[str, dict[str, Any]],
    *,
    config: dict[str, Any],
) -> list[EvidenceBlock]:
    if not blocks or not chunk_index:
        return blocks
    source_keys = {_normalize_doc_key(block.source_file) for block in blocks if block.source_file}
    existing_chunk_ids = {block.chunk_id for block in blocks if block.chunk_id}
    qtypes = set(analysis.get("question_types", []))
    target_fact_types: set[str] = set()
    for qtype in qtypes:
        target_fact_types.update(QUESTION_TYPE_TO_FACT_TYPE.get(qtype, set()))
    target_fact_types.update(analysis.get("period_subtypes", []))
    target_fact_types.update(_fact_types_from_intent_plan(analysis))
    target_fact_types.update({"document_identity", "document_summary"})
    expanded = list(blocks)
    for chunk_id, chunk in chunk_index.items():
        if chunk_id in existing_chunk_ids:
            continue
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        source_file = str(chunk.get("source_file") or metadata.get("source_file") or "")
        if _normalize_doc_key(source_file) not in source_keys:
            continue
        chunk_type = str(chunk.get("chunk_type") or metadata.get("chunk_type") or "")
        fact_type = str(chunk.get("fact_type") or metadata.get("fact_type") or "")
        if chunk_type != "fact_candidates" or fact_type not in target_fact_types:
            continue
        row = {
            "chunk_id": chunk_id,
            "source_file": source_file,
            "rank": 999,
            "selection_stage": "same_source_fact_lookup",
        }
        text = chunk.get("evidence_text_short") or chunk.get("content") or ""
        expanded.append(
            EvidenceBlock(
                source_file=source_file,
                source_file_nfc=str(chunk.get("source_file_nfc") or metadata.get("source_file_nfc") or source_file),
                chunk_id=chunk_id,
                rank=999,
                chunk_type=chunk_type,
                fact_type=fact_type,
                section_path=str(metadata.get("section_path") or chunk.get("section_path") or ""),
                text=truncate_text(text, int(config["evidence_text_chars"])),
                score=_score_evidence(row, chunk, analysis) - 8.0,
                source_store_id=str((chunk.get("source_ref") or {}).get("source_store_id", "")) if isinstance(chunk.get("source_ref"), dict) else "",
                evidence_id=str(chunk.get("evidence_id") or metadata.get("evidence_id") or ""),
                retrieval_role=str(chunk.get("retrieval_role") or metadata.get("retrieval_role") or ""),
                answer_policy=str(chunk.get("answer_policy") or metadata.get("answer_policy") or ""),
                answer_risk_level=str(chunk.get("answer_risk_level") or metadata.get("answer_risk_level") or ""),
                budget_answer_enabled=_as_bool(chunk.get("budget_answer_enabled") or metadata.get("budget_answer_enabled")),
                eligibility_answer_enabled=_as_bool(chunk.get("eligibility_answer_enabled") or metadata.get("eligibility_answer_enabled")),
                payment_answer_enabled=_as_bool(chunk.get("payment_answer_enabled") or metadata.get("payment_answer_enabled")),
                selection_stage="same_source_fact_lookup",
                is_backfilled=False,
            )
        )
    return expanded


def _score_evidence(row: dict[str, Any], chunk: dict[str, Any], analysis: dict[str, Any]) -> float:
    question_types = set(analysis.get("question_types", []))
    target_fact_types = set()
    for qtype in question_types:
        target_fact_types.update(QUESTION_TYPE_TO_FACT_TYPE.get(qtype, set()))
    target_fact_types.update(analysis.get("period_subtypes", []))
    target_fact_types.update(_fact_types_from_intent_plan(analysis))

    chunk_type = str(chunk.get("chunk_type") or row.get("chunk_type") or "")
    fact_type = str(chunk.get("fact_type") or row.get("fact_type") or "")
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    text = normalize_text(
        " ".join(
            [
                str(chunk.get("content", "")),
                str(chunk.get("evidence_text_short", "")),
                str(row.get("text", "")),
                str(chunk.get("metadata", {}).get("section_path", ""))
                if isinstance(chunk.get("metadata"), dict)
                else "",
            ]
        )
    )

    score = 100.0 / max(_safe_float(row.get("rank"), 1.0), 1.0)
    if chunk_type == "fact_candidates":
        score += 15.0
    if chunk_type == "table":
        score += 8.0
    if _as_bool(row.get("is_backfilled")):
        score -= 12.0
    if fact_type in target_fact_types:
        score += 60.0
    answer_policy = str(chunk.get("answer_policy") or metadata.get("answer_policy") or row.get("answer_policy") or "")
    if fact_type == "document_identity" or answer_policy == "route_only_not_final_answer":
        score += 8.0
        if not analysis.get("is_multi_doc") and not analysis.get("needs_synthesis"):
            score -= 45.0
    budget_answer_enabled = _as_bool(
        chunk.get("budget_answer_enabled")
        or metadata.get("budget_answer_enabled")
        or row.get("budget_answer_enabled")
    )
    target_match_score = _best_target_match_score(_doc_match_text(row=row, chunk=chunk), analysis.get("target_slots", []))
    if analysis.get("target_slots"):
        if target_match_score >= STRONG_TARGET_MATCH_THRESHOLD:
            score += 45.0
        elif target_match_score >= TARGET_MATCH_THRESHOLD:
            score += 25.0
        elif fact_type in (FINAL_BUDGET_FACT_TYPES | ALLOCATION_BUDGET_FACT_TYPES) or (chunk_type == "fact_candidates" and "budget" in question_types):
            score -= 55.0
    if "budget" in question_types:
        if budget_answer_enabled:
            score += 35.0
        if fact_type in FINAL_BUDGET_FACT_TYPES or ("total_allocation_lookup" in set(analysis.get("intent_slots", [])) and fact_type == "total_allocation"):
            score += 25.0
        if fact_type in BUDGET_BLOCKED_FACT_TYPES:
            score -= 90.0
        if chunk_type == "fact_candidates" and not budget_answer_enabled:
            score -= 45.0
    intents = set(analysis.get("intent_slots", []))
    if "total_allocation_lookup" in intents:
        if fact_type == "total_allocation":
            score += 70.0
        if has_any(text, ALLOCATION_BUDGET_CONTEXT_KEYWORDS):
            score += 45.0
    if "technical_requirement_lookup" in intents:
        if chunk_type in {"text", "table"}:
            score += 24.0
        if fact_type in {"document_summary", "business_type"}:
            score -= 18.0
        if has_any(text, TECHNICAL_REQUIREMENT_KEYWORDS):
            score += 36.0
    if "technical_purpose_summary" in intents:
        if chunk_type in {"text", "table"}:
            score += 18.0
        if has_any(text, TECHNICAL_PURPOSE_KEYWORDS):
            score += 34.0
    if "risk_comparison" in intents:
        if chunk_type in {"text", "table"}:
            score += 18.0
        if has_any(text, RISK_COMPARISON_KEYWORDS):
            score += 32.0
    if "eligibility" in question_types and fact_type in {"threshold_budget", "eligibility"} and _as_bool(chunk.get("eligibility_answer_enabled") or metadata.get("eligibility_answer_enabled") or row.get("eligibility_answer_enabled")):
        score += 20.0
    for qtype in question_types:
        if qtype in QUESTION_KEYWORDS and has_any(text, QUESTION_KEYWORDS[qtype]):
            score += 10.0
    if analysis.get("needs_synthesis") and chunk_type in {"table", "text"}:
        score += 8.0
    return score


def _select_evidence_blocks(
    blocks: list[EvidenceBlock],
    analysis: dict[str, Any],
    *,
    max_blocks: int,
) -> list[EvidenceBlock]:
    candidate_blocks = [
        block
        for block in blocks
        if not _is_target_mismatched_final_value_block(block, analysis)
    ]
    ranked_all = sorted(candidate_blocks, key=lambda item: item.score, reverse=True)
    selected: list[EvidenceBlock] = []
    target_slots = analysis.get("target_slots", [])
    if target_slots:
        for slot in target_slots:
            matched = [
                block
                for block in ranked_all
                if _best_target_match_score(_doc_match_text(block=block), [slot]) >= TARGET_MATCH_THRESHOLD
            ]
            if matched:
                selected.append(matched[0])

    if not analysis.get("is_multi_doc"):
        used_ids = {id(block) for block in selected}
        selected.extend(block for block in ranked_all if id(block) not in used_ids)
        return selected[:max_blocks]

    grouped: dict[str, list[EvidenceBlock]] = defaultdict(list)
    for block in candidate_blocks:
        grouped[block.source_file or "unknown"].append(block)

    for source_file, group in sorted(grouped.items()):
        ranked = sorted(group, key=lambda item: item.score, reverse=True)
        for block in ranked[: max(1, max_blocks // max(len(grouped), 1))]:
            if id(block) not in {id(item) for item in selected}:
                selected.append(block)

    if len(selected) < max_blocks:
        used_ids = {id(block) for block in selected}
        selected.extend(block for block in ranked_all if id(block) not in used_ids)
    return selected[:max_blocks]


def _is_target_mismatched_final_value_block(block: EvidenceBlock, analysis: dict[str, Any]) -> bool:
    if not analysis.get("target_slots"):
        return False
    intents = set(analysis.get("intent_slots", []))
    if not any(intent in intents for intent in {"budget_lookup", "budget_difference", "budget_sum", "budget_ratio", "total_allocation_lookup"}):
        return False
    if block.fact_type not in FINAL_BUDGET_FACT_TYPES:
        return False
    return _best_target_match_score(_doc_match_text(block=block), analysis.get("target_slots", [])) < TARGET_MATCH_THRESHOLD


def _build_core_summary(
    blocks: list[EvidenceBlock],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    docs: dict[str, dict[str, Any]] = {}
    for block in blocks:
        key = block.source_file or "unknown"
        doc = docs.setdefault(
            key,
            {
                "source_file": key,
                "fact_types": [],
                "key_values": defaultdict(list),
                "evidence_count": 0,
            },
        )
        doc["evidence_count"] += 1
        if block.fact_type:
            doc["fact_types"].append(block.fact_type)
            doc["key_values"][block.fact_type].append(_extract_short_value(block.text, block.fact_type))

        inferred_values = _extract_values_by_question_type(block.text, analysis)
        for value_type, values in inferred_values.items():
            doc["key_values"][value_type].extend(values)

    normalized_docs = []
    for doc in docs.values():
        key_values = {
            key: _unique_preserve_order([value for value in values if value])
            for key, values in doc["key_values"].items()
        }
        normalized_docs.append(
            {
                "source_file": doc["source_file"],
                "fact_types": sorted(set(doc["fact_types"])),
                "key_values": key_values,
                "evidence_count": doc["evidence_count"],
            }
        )

    return {
        "answer_type": analysis.get("answer_type", "unknown"),
        "question_types": analysis.get("question_types", []),
        "period_subtypes": analysis.get("period_subtypes", []),
        "intent_slots": analysis.get("intent_slots", []),
        "intent_plan": analysis.get("intent_plan", []),
        "target_slots": analysis.get("target_slots", []),
        "document_count": len(normalized_docs),
        "documents": normalized_docs,
    }


def _build_intent_evidence_groups(
    blocks: list[EvidenceBlock],
    analysis: dict[str, Any],
    *,
    max_blocks_per_intent: int = 4,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    target_slots = analysis.get("target_slots", [])
    for plan in analysis.get("intent_plan", []) or []:
        required_fact_types = {str(value) for value in plan.get("required_fact_types", []) or []}
        preferred_chunk_types = [str(value) for value in plan.get("preferred_chunk_types", []) or []]
        ranked: list[tuple[float, EvidenceBlock]] = []
        for block in blocks:
            score = block.score
            block_text = normalize_text(block.text)
            if block.fact_type in required_fact_types:
                score += 120.0
            if block.chunk_type in preferred_chunk_types:
                score += 20.0
            if target_slots and _best_target_match_score(_doc_match_text(block=block), target_slots) >= TARGET_MATCH_THRESHOLD:
                score += 30.0
            if plan.get("requires_computation") and _is_allowed_budget_operand_block(block):
                score += 40.0
            intent = str(plan.get("intent", ""))
            if intent == "total_allocation_lookup" and has_any(block_text, ALLOCATION_BUDGET_CONTEXT_KEYWORDS):
                score += 45.0
            if intent == "technical_requirement_lookup" and has_any(block_text, TECHNICAL_REQUIREMENT_KEYWORDS):
                score += 40.0
            if intent == "technical_purpose_summary" and has_any(block_text, TECHNICAL_PURPOSE_KEYWORDS):
                score += 35.0
            if intent == "risk_comparison" and has_any(block_text, RISK_COMPARISON_KEYWORDS):
                score += 35.0
            ranked.append((score, block))

        evidence = []
        for _, block in sorted(ranked, key=lambda item: item[0], reverse=True)[:max_blocks_per_intent]:
            evidence.append(
                {
                    "source_file": block.source_file,
                    "chunk_id": block.chunk_id,
                    "evidence_id": block.evidence_id,
                    "chunk_type": block.chunk_type,
                    "fact_type": block.fact_type,
                    "section_path": block.section_path,
                    "value": _extract_short_value(block.text, block.fact_type) if block.fact_type else truncate_text(block.text, 120),
                }
            )
        groups.append(
            {
                "intent_id": plan.get("intent_id", ""),
                "intent": plan.get("intent", ""),
                "answer_section": plan.get("answer_section", ""),
                "required_fact_types": list(required_fact_types),
                "preferred_chunk_types": preferred_chunk_types,
                "evidence": evidence,
            }
        )
    return groups


def _extract_short_value(text: str, fact_type: str) -> str:
    if fact_type in {"budget", "project_budget", "estimated_price", "base_amount", "total_allocation"}:
        values = AMOUNT_RE.findall(text)
        return values[0] if values else truncate_text(text, 120)
    if fact_type == "bid_deadline":
        values = DATE_RE.findall(text)
        return values[0] if values else truncate_text(text, 120)
    if fact_type in {"duration", "project_duration", "maintenance_period", "warranty_period", "deadline_term"}:
        values = DURATION_RE.findall(text)
        return values[0] if values else truncate_text(text, 120)
    return truncate_text(text, 120)


def _extract_values_by_question_type(text: str, analysis: dict[str, Any]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    qtypes = set(analysis.get("question_types", []))
    if "budget" in qtypes:
        values["budget"].extend(AMOUNT_RE.findall(text))
    if {"duration", "bid_deadline"} & qtypes:
        values["date"].extend(DATE_RE.findall(text))
        values["duration"].extend(DURATION_RE.findall(text))
    return values


def _format_context_text(
    core_summary: dict[str, Any],
    blocks: list[EvidenceBlock],
    analysis: dict[str, Any],
    *,
    max_chars: int,
) -> str:
    lines = [
        "[핵심 추출값 요약]",
        f"질문유형: {', '.join(analysis.get('question_types', []))}",
        f"답변유형: {analysis.get('answer_type', 'unknown')}",
    ]
    if analysis.get("period_subtypes"):
        lines.append(f"기간 세부유형: {', '.join(analysis['period_subtypes'])}")
    if analysis.get("intent_slots"):
        lines.append(f"의도 슬롯: {', '.join(analysis.get('intent_slots', []))}")
    if analysis.get("intent_plan"):
        lines.append("")
        lines.append("[intent plan - 질문 안의 하위 요청]")
        for plan in analysis.get("intent_plan", []):
            required = ", ".join(plan.get("required_fact_types", []) or []) or "-"
            chunks = ", ".join(plan.get("preferred_chunk_types", []) or []) or "-"
            targets = " | ".join(plan.get("targets", []) or []) or "-"
            signals = ", ".join(plan.get("classification_signals", []) or []) or "-"
            lines.append(
                f"- {plan.get('intent_id', '')} {plan.get('answer_section', '')}: "
                f"intent={plan.get('intent', '')} | target={targets} | "
                f"required_fact_types={required} | preferred_chunk_types={chunks} | "
                f"requires_computation={plan.get('requires_computation', False)} | "
                f"target_policy={plan.get('target_policy', '')} | signals={signals}"
            )
    if analysis.get("is_multi_doc"):
        lines.append("주의: 여러 문서를 묻는 질문입니다. 문서별로 값을 분리해서 답해야 합니다.")
    if core_summary.get("target_slots"):
        lines.append("")
        lines.append("[target slots]")
        for slot in core_summary.get("target_slots", []):
            lines.append(
                f"- target={slot.get('target_label', '')} | matched_source_file={slot.get('matched_source_file', '') or '-'} | match_score={slot.get('match_score', 0)}"
            )
    if analysis.get("scenario_terms"):
        lines.append("")
        lines.append("[scenario terms - target 문서명이 아니라 비교/위험 조건]")
        lines.append(", ".join(analysis.get("scenario_terms", [])))
    if core_summary.get("computed_values") and core_summary.get("computed_values", {}).get("result") is not None:
        lines.append("")
        lines.append("[computed values - 코드 계산 결과]")
        lines.append(json.dumps(core_summary.get("computed_values"), ensure_ascii=False))

    if core_summary.get("intent_evidence"):
        lines.append("")
        lines.append("[intent별 근거 묶음]")
        for group in core_summary.get("intent_evidence", []):
            required = ", ".join(group.get("required_fact_types", []) or []) or "-"
            lines.append(
                f"- {group.get('intent_id', '')} {group.get('answer_section', '')}: "
                f"intent={group.get('intent', '')} | required_fact_types={required}"
            )
            for evidence in group.get("evidence", [])[:3]:
                lines.append(
                    f"  · source_file={evidence.get('source_file', '')} | "
                    f"chunk_id={evidence.get('chunk_id', '')} | "
                    f"fact_type={evidence.get('fact_type') or '-'} | value={evidence.get('value', '')}"
                )

    for doc in core_summary.get("documents", []):
        lines.append("")
        lines.append(f"- 문서: {doc['source_file']}")
        if doc.get("fact_types"):
            lines.append(f"  fact_type: {', '.join(doc['fact_types'])}")
        for key, values in doc.get("key_values", {}).items():
            if values:
                lines.append(f"  {key}: {' | '.join(values[:3])}")

    lines.append("")
    lines.append("[근거 block]")
    for idx, block in enumerate(blocks, start=1):
        lines.append("")
        lines.append(
            f"근거 {idx}: evidence_id={block.evidence_id or f'E{idx}'} | source_file={block.source_file} | "
            f"chunk_id={block.chunk_id} | chunk_type={block.chunk_type} | "
            f"fact_type={block.fact_type or '-'} | section={block.section_path or '-'} | "
            f"retrieval_role={block.retrieval_role or '-'} | answer_policy={block.answer_policy or '-'} | "
            f"selection_stage={block.selection_stage or '-'} | backfilled={block.is_backfilled}"
        )
        lines.append(block.text)
        if block.source_full_text:
            lines.append("[source_store 확장 원문]")
            lines.append(block.source_full_text)

    text = "\n".join(lines)
    return truncate_text_preserve_lines(text, max_chars)


def build_prompt(context_package: dict[str, Any]) -> list[dict[str, str]]:
    schema = json.dumps(ANSWER_SCHEMA, ensure_ascii=False, indent=2)
    system = (
        "너는 RFP 문서 기반 QA assistant다. 반드시 제공된 Context 안의 정보만 사용한다. "
        "Context에 없으면 추측하지 말고 is_answerable=false로 답한다. "
        "금액, 날짜, 기간, 공고번호는 원문 표현을 우선 보존한다. "
        "사업기간, 제출기한, 입찰마감일, 유지보수기간, 하자담보책임기간을 섞지 않는다. "
        "여러 문서를 묻는 질문은 문서별로 값을 분리한다. "
        "출력은 JSON 객체 하나만 반환한다."
    )
    user = f"""
[질문]
{context_package.get('question', '')}

[Context]
{context_package.get('context_text', '')}

[출력 JSON 스키마]
{schema}

[답변 규칙]
- answer에는 사용자에게 보여줄 최종 답변을 한국어로 작성한다.
- citations는 직접 생성하지 않는다. 근거 citation은 후처리 코드가 Context의 evidence block에서 자동으로 붙인다.
- fact_type=document_identity 또는 answer_policy=route_only_not_final_answer 근거는 문서 식별 신호로만 사용하고, 숫자/날짜/금액의 최종 근거로 사용하지 않는다.
- backfilled=True 근거는 보조 근거로 취급하고, 핵심값은 answer_policy가 허용하는 fact에서 다시 확인한다.
- 예산 질문에서 threshold_budget 또는 payment_terms는 입찰자격/지급조건 신호일 뿐 사업예산의 최종값으로 쓰지 않는다.
- [intent plan - 질문 안의 하위 요청]이 있으면 intent_id별 요청을 빠짐없이 답한다.
- 답변은 가능한 한 intent plan의 answer_section 순서에 맞춰 작성한다. 예: `예산: ...\n핵심 요약: ...\n근거: ...`
- required_fact_types에 해당하는 근거가 없으면 다른 문서나 다른 fact_type 값으로 대체하지 말고 missing_info에 남긴다.
- [computed values - 코드 계산 결과]가 있으면 숫자, formula, steps, 계산 결과를 변경하지 말고 그대로 사용한다.
- computed values에 steps가 있으면 answer에 `계산 과정:`을 포함하고 단계별 계산식과 최종 결론을 함께 작성한다.
- 의도 슬롯이 여러 개이면 모든 의도에 답한다. 예: budget_lookup + purpose_summary이면 예산과 핵심 요약을 모두 포함한다.
- budget_lookup + purpose_summary 질문은 answer를 반드시 `예산: ...\n핵심 요약: ...\n근거: ...` 형식으로 작성한다.
- budget_sum/budget_difference + multi_doc_comparison/risk_comparison 질문은 계산만 답하지 말고 `예산/계산: ...\n비교/리스크: ...\n근거: ...`를 모두 작성한다.
- total_allocation_lookup은 전체 배정액/배정예산/전체 사업비를 묻는 intent다. 추정가격, 기초금액, 지급조건 금액으로 대체하지 않는다.
- technical_requirement_lookup은 DB, 통계, 쿼리, 캐싱, 튜닝, 오프라인, 재연결 같은 기술 조치를 묻는 intent다. 참가자격/포함 여부로 답하지 말고 원문 요구사항 근거를 찾아 답한다.
- technical_purpose_summary는 예측, 예지 보전, 빅데이터, 이상 로그, 고장 조기 경보 같은 기술 목적을 묻는 intent다. 일반 사업기간/유지보수기간으로 답하지 않는다.
- scenario_terms는 질문의 비교 조건이나 위험 시나리오이며 target 문서명이 아니다. target slots의 matched_source_file에서 값을 선택하고, scenario_terms는 비교 관점으로만 사용한다.
- budget_difference/budget_sum/budget_ratio 질문은 직접 계산하지 말고 [computed values - 코드 계산 결과]의 steps와 answer를 최종 답변으로 사용한다.
- target slots가 있으면 matched_source_file이 일치하는 문서의 값만 최종값으로 사용한다. 같은 기관의 다른 사업 예산을 대체값으로 쓰지 않는다.
- 근거가 부족하면 is_answerable=false, answer_status=insufficient_context 또는 not_found_in_context, confidence=low로 둔다.
- 문서에 없다는 답변은 answer_status=not_found_in_context로 표시한다.
- missing_info와 warnings를 적극적으로 사용한다.
- JSON 외의 설명 문장은 출력하지 않는다.
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def postprocess_answer(raw_text: str, context_package: dict[str, Any]) -> dict[str, Any]:
    parsed, valid_json, parse_error_type = _parse_json_answer(raw_text)
    recovered_answer = bool(not valid_json and str(parsed.get("answer", "")).strip())
    if not valid_json and not recovered_answer:
        parsed = {
            "answer": "",
            "answer_type": context_package.get("question_analysis", {}).get("answer_type", "unknown"),
            "confidence": "low",
            "is_answerable": False,
            "final_values": {},
            "documents": [],
            "citations": [],
            "missing_info": ["valid_json"],
            "warnings": ["LLM output was not valid JSON."],
        }
    elif not valid_json:
        parsed.setdefault(
            "answer_type",
            context_package.get("question_analysis", {}).get("answer_type", "unknown"),
        )
        parsed.setdefault("confidence", "low")
        parsed.setdefault("is_answerable", True)
        parsed.setdefault("final_values", {})
        parsed.setdefault("documents", [])
        parsed.setdefault("missing_info", [])
        parsed.setdefault("warnings", [])
        parsed["missing_info"] = _ensure_list(parsed.get("missing_info"))
        parsed["warnings"] = _ensure_list(parsed.get("warnings"))
        parsed["missing_info"].append("valid_json")
        parsed["warnings"].extend(["LLM output was not valid JSON.", "answer_recovered_from_raw"])

    normalized = _normalize_answer_schema(parsed, context_package)
    normalized = _apply_deterministic_postprocess(normalized, context_package)
    normalized, unit_tags = _normalize_answer_amount_units(normalized, context_package)
    normalized["citations"] = _attach_deterministic_citations(normalized, context_package)
    if not normalized.get("documents") and normalized.get("citations"):
        normalized["documents"] = _unique_preserve_order(
            citation.get("source_file", "")
            for citation in normalized["citations"]
            if isinstance(citation, dict) and citation.get("source_file")
        )
    failure_tags = list(context_package.get("failure_tags", []))
    failure_tags.extend(unit_tags)

    citation_report = _validate_citations(normalized, context_package)
    grounding_report = _validate_numeric_grounding(normalized, context_package)
    policy_report = _validate_answer_policy(normalized, context_package)
    if not valid_json:
        failure_tags.append("llm_invalid_json")
    if not citation_report["citation_valid"]:
        failure_tags.append("insufficient_evidence")
    if not grounding_report["numeric_grounded"] and normalized.get("answer_status") != "not_found_in_context":
        failure_tags.append("llm_hallucination_risk")
    if not grounding_report.get("source_numeric_grounded", True) and normalized.get("answer_status") != "not_found_in_context":
        failure_tags.append("source_numeric_missing")
    if (
        _has_target_required_field_missing(context_package, "project_budget")
        and normalized.get("answer_status") != "not_found_in_context"
        and not _has_grounded_budget_fallback_for_missing_target(normalized, context_package)
    ):
        failure_tags.append("source_numeric_missing")
    if _has_target_required_field_missing(context_package, "total_allocation") and normalized.get("answer_status") != "not_found_in_context":
        failure_tags.append("source_numeric_missing")
    if "total_allocation_missing" in set(normalized.get("missing_info", [])):
        failure_tags.append("source_numeric_missing")
    if not grounding_report.get("derived_numeric_valid", True):
        failure_tags.append("derived_numeric_mismatch")
    if not policy_report["policy_valid"]:
        failure_tags.append("wrong_field_selection")
        normalized["warnings"].extend(policy_report["policy_warnings"])
    if _is_incomplete_multi_doc(normalized, context_package):
        failure_tags.append("incomplete_multi_doc")
    if _has_wrong_target_citation(normalized, context_package):
        failure_tags.append("citation_wrong_target")
    if _has_wrong_target_field_selection(normalized, context_package):
        failure_tags.append("wrong_target_field_selection")
    missing_intents = _missing_intents(normalized, context_package)
    if missing_intents:
        normalized["missing_info"] = _unique_preserve_order(
            list(normalized.get("missing_info", []))
            + [f"missing_intent:{intent}" for intent in missing_intents]
        )
    if missing_intents or _is_multi_intent_incomplete(normalized, context_package):
        failure_tags.append("multi_intent_incomplete")
    missing_aspects = _missing_question_aspects(normalized, context_package)
    if missing_aspects:
        normalized["missing_info"] = _unique_preserve_order(
            list(normalized.get("missing_info", []))
            + [f"missing_question_aspect:{aspect}" for aspect in missing_aspects]
        )
        failure_tags.append("question_aspect_incomplete")
    if normalized.get("answer_status") == "not_found_in_context" and not normalized.get("citations"):
        failure_tags.append("negative_answer_no_checked_evidence")
    if _has_target_doc_coverage_missing(context_package):
        failure_tags.append("target_doc_coverage_missing")

    normalized["_raw_text"] = raw_text
    normalized["_valid_json"] = valid_json
    normalized["_recovered_answer"] = recovered_answer
    normalized["_parse_error_type"] = parse_error_type
    normalized["_citation_valid"] = citation_report["citation_valid"]
    normalized["_numeric_grounded"] = grounding_report["numeric_grounded"]
    normalized["_source_numeric_grounded"] = grounding_report.get("source_numeric_grounded")
    normalized["_derived_numeric_valid"] = grounding_report.get("derived_numeric_valid")
    normalized["_ungrounded_values"] = grounding_report["ungrounded_values"]
    normalized["_derived_numeric_values"] = grounding_report.get("derived_values", [])
    normalized["_answer_policy_valid"] = policy_report["policy_valid"]
    normalized["_answer_policy_violations"] = policy_report["policy_violations"]
    normalized["_missing_intents"] = missing_intents
    failure_tags = _unique_preserve_order(failure_tags)
    normalized = _downgrade_confidence_for_failure_tags(normalized, failure_tags)
    normalized["_failure_tags"] = failure_tags
    normalized["_question_analysis"] = context_package.get("question_analysis", {})
    return normalized


SEVERE_FAILURE_TAGS = {
    "llm_hallucination_risk",
    "source_numeric_missing",
    "derived_numeric_mismatch",
    "wrong_field_selection",
    "wrong_target_field_selection",
    "citation_wrong_target",
    "insufficient_evidence",
    "gt_numeric_mismatch",
    "gt_expected_answer_but_model_not_found",
}
MEDIUM_FAILURE_TAGS = {
    "multi_intent_incomplete",
    "incomplete_multi_doc",
    "gt_semantic_overlap_low",
    "question_aspect_incomplete",
    "target_doc_coverage_missing",
}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _downgrade_confidence_for_failure_tags(answer: dict[str, Any], failure_tags: Iterable[str]) -> dict[str, Any]:
    tags = set(failure_tags or [])
    current = str(answer.get("confidence") or "low")
    target = current if current in CONFIDENCE_RANK else "low"
    if tags & SEVERE_FAILURE_TAGS:
        target = "low"
    elif target == "high" and tags & MEDIUM_FAILURE_TAGS:
        target = "medium"
    if CONFIDENCE_RANK.get(target, 0) < CONFIDENCE_RANK.get(current, 0):
        answer["confidence"] = target
        answer.setdefault("warnings", [])
        answer["warnings"] = _unique_preserve_order(
            list(answer.get("warnings", [])) + [f"confidence_downgraded_by_guard:{current}->{target}"]
        )
    return answer


def _parse_json_answer(raw_text: str) -> tuple[dict[str, Any], bool, str]:
    raw_text = str(raw_text or "").strip()
    if not raw_text:
        return {}, False, "empty_output"
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else {}, isinstance(parsed, dict), ""
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, flags=re.S)
    if not match:
        error_type = "truncated_json" if raw_text.lstrip().startswith("{") else "no_json_object"
        return _recover_partial_answer_fields(raw_text), False, error_type
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}, isinstance(parsed, dict), ""
    except json.JSONDecodeError:
        error_type = "truncated_json" if raw_text.lstrip().startswith("{") and not raw_text.rstrip().endswith("}") else "json_decode_error"
        return _recover_partial_answer_fields(raw_text), False, error_type


def _recover_partial_answer_fields(raw_text: str) -> dict[str, Any]:
    recovered: dict[str, Any] = {}
    for key in ["answer", "answer_type", "confidence"]:
        value = _extract_json_string_field(raw_text, key)
        if value:
            recovered[key] = value
    bool_value = _extract_json_bool_field(raw_text, "is_answerable")
    if bool_value is not None:
        recovered["is_answerable"] = bool_value
    if "answer" in recovered:
        recovered.setdefault("final_values", {})
        recovered.setdefault("documents", [])
        recovered.setdefault("missing_info", [])
        recovered.setdefault("warnings", [])
    return recovered


def _extract_json_string_field(raw_text: str, key: str) -> str:
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
        raw_text,
        flags=re.S,
    )
    if not match:
        return ""
    value = match.group(1)
    try:
        return str(json.loads(f'"{value}"'))
    except json.JSONDecodeError:
        return value.replace('\\"', '"').replace("\\n", "\n")


def _extract_json_bool_field(raw_text: str, key: str) -> bool | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false)', raw_text, flags=re.I)
    if not match:
        return None
    return match.group(1).casefold() == "true"


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_answer_schema(
    parsed: dict[str, Any],
    context_package: dict[str, Any],
) -> dict[str, Any]:
    analysis = context_package.get("question_analysis", {})
    normalized = {
        "answer": str(parsed.get("answer", "")),
        "answer_type": str(parsed.get("answer_type") or analysis.get("answer_type", "unknown")),
        "confidence": str(parsed.get("confidence") or "low"),
        "is_answerable": bool(parsed.get("is_answerable", False)),
        "final_values": parsed.get("final_values") if isinstance(parsed.get("final_values"), dict) else {},
        "documents": parsed.get("documents") if isinstance(parsed.get("documents"), list) else [],
        "citations": parsed.get("citations") if isinstance(parsed.get("citations"), list) else [],
        "missing_info": parsed.get("missing_info") if isinstance(parsed.get("missing_info"), list) else [],
        "warnings": parsed.get("warnings") if isinstance(parsed.get("warnings"), list) else [],
    }
    status = str(parsed.get("answer_status") or "").strip()
    if status not in ANSWER_STATUS_VALUES:
        if normalized["is_answerable"]:
            status = "answered"
        elif normalized["answer"]:
            status = "not_found_in_context"
        else:
            status = "insufficient_context"
    normalized["answer_status"] = status
    if normalized["answer_type"] not in ALLOWED_ANSWER_TYPES:
        normalized["warnings"].append(f"unsupported_answer_type:{normalized['answer_type']}")
        normalized["answer_type"] = analysis.get("answer_type", "unknown")
        if normalized["answer_type"] not in ALLOWED_ANSWER_TYPES:
            normalized["answer_type"] = "unknown"
    if analysis.get("is_multi_intent") and analysis.get("answer_type") in ALLOWED_ANSWER_TYPES:
        if normalized["answer_type"] != analysis.get("answer_type"):
            normalized["warnings"].append(
                f"answer_type_overridden_for_multi_intent:{normalized['answer_type']}->{analysis.get('answer_type')}"
            )
            normalized["answer_type"] = analysis.get("answer_type", normalized["answer_type"])
    if normalized["confidence"] not in {"high", "medium", "low"}:
        normalized["warnings"].append(f"unsupported_confidence:{normalized['confidence']}")
        normalized["confidence"] = "low"
    return normalized


def _normalize_answer_amount_units(answer: dict[str, Any], context_package: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    tags: list[str] = []
    text = str(answer.get("answer", ""))
    if not text:
        return answer, tags

    def replace_parenthetical(match: re.Match[str]) -> str:
        nonlocal tags
        left = match.group(1)
        right = match.group(2)
        left_won = _amount_to_won(left)
        right_won = _amount_to_won(right)
        if left_won is None or right_won is None or left_won == right_won:
            return match.group(0)
        tags.append("unit_conversion_mismatch")
        tags.append("unit_conversion_corrected")
        return _format_won(left_won)

    amount_pattern = AMOUNT_RE.pattern
    text = re.sub(
        rf"({amount_pattern})\s*\(\s*({amount_pattern})\s*\)",
        replace_parenthetical,
        text,
    )

    for amount in _extract_amount_values(text):
        raw = str(amount.get("raw", ""))
        won = amount.get("won")
        if won is None:
            continue
        formatted = _format_won(won)
        if any(unit in raw for unit in ["천원", "백만원", "천만원", "만원", "억원", "억 원"]) and formatted not in text:
            text = text.replace(raw, formatted)
            tags.append("unit_conversion_corrected")

    if text != answer.get("answer"):
        answer["answer"] = text
        answer["warnings"] = _unique_preserve_order(list(answer.get("warnings", [])) + ["unit_conversion_corrected"])

    normalized_final_values = {}
    changed = False
    for key, value in (answer.get("final_values") or {}).items():
        if isinstance(value, str):
            won = _amount_to_won(value)
            if won is not None and any(unit in value for unit in ["천원", "백만원", "천만원", "만원", "억원", "억 원"]):
                normalized_final_values[key] = won
                changed = True
                continue
        normalized_final_values[key] = value
    if changed:
        answer["final_values"] = normalized_final_values
        tags.append("unit_conversion_corrected")
    return answer, _unique_preserve_order(tags)


def _apply_deterministic_postprocess(answer: dict[str, Any], context_package: dict[str, Any]) -> dict[str, Any]:
    computed = context_package.get("core_summary", {}).get("computed_values") or context_package.get("question_analysis", {}).get("computed_values") or {}
    if computed and computed.get("result") is not None:
        answer.setdefault("final_values", {})
        answer["final_values"]["computed_values"] = computed
        computed_answer = computed.get("answer")
        if computed_answer:
            analysis = context_package.get("question_analysis", {})
            is_budget_answer = answer.get("answer_type") == "budget" or analysis.get("answer_type") == "budget"
            summary_intents = {
                "purpose_summary",
                "requirements_summary",
                "technical_requirement_lookup",
                "technical_purpose_summary",
                "risk_comparison",
                "multi_doc_comparison",
            }
            has_summary_intent = bool(set(analysis.get("intent_slots", [])) & summary_intents)
            rendered_computed_answer = _render_computed_answer(computed, context_package)
            if is_budget_answer and not has_summary_intent:
                answer["answer"] = rendered_computed_answer or computed_answer
                answer["is_answerable"] = True
                answer["answer_status"] = "answered"
                answer["confidence"] = "high"
            elif (rendered_computed_answer or computed_answer) not in str(answer.get("answer", "")):
                answer["answer"] = _merge_computed_with_existing_answer(
                    rendered_computed_answer or computed_answer,
                    str(answer.get("answer", "")),
                    analysis,
                )
    elif _has_target_required_field_missing(context_package, "project_budget"):
        analysis = context_package.get("question_analysis", {})
        has_budget_intent = any(
            intent in set(analysis.get("intent_slots", []))
            for intent in {"budget_lookup", "budget_difference", "budget_sum", "budget_ratio"}
        )
        if has_budget_intent and not _has_grounded_budget_fallback_for_missing_target(answer, context_package):
            missing_targets = _dedupe_overlapping_target_labels([
                slot.get("target_label", "")
                for slot in analysis.get("target_slots", [])
                if "project_budget" in (slot.get("missing_fields") or [])
            ])
            target_text = ", ".join([target for target in missing_targets if target]) or "대상 문서"
            answer["answer"] = f"{target_text}의 사업예산 근거를 context에서 확인할 수 없어 계산할 수 없습니다."
            answer["is_answerable"] = False
            answer["answer_status"] = "insufficient_context"
            answer["confidence"] = "low"
            answer["final_values"] = {}
            answer.setdefault("missing_info", [])
            answer["missing_info"] = _unique_preserve_order(
                list(answer.get("missing_info", [])) + ["target_project_budget_missing"]
            )
    analysis = context_package.get("question_analysis", {})
    if (
        "total_allocation_lookup" in set(analysis.get("intent_slots", []))
        and (
            _has_target_required_field_missing(context_package, "total_allocation")
            or not _has_total_allocation_context(context_package)
        )
    ):
        answer["answer"] = "전체 배정액 근거를 context에서 확인할 수 없습니다. 추정가격, 기초금액, 지급조건 금액으로 대체하지 않았습니다."
        answer["is_answerable"] = False
        answer["answer_status"] = "insufficient_context"
        answer["confidence"] = "low"
        answer["final_values"] = {}
        answer.setdefault("missing_info", [])
        answer["missing_info"] = _unique_preserve_order(
            list(answer.get("missing_info", [])) + ["total_allocation_missing"]
        )
    if not answer.get("answer_status"):
        answer["answer_status"] = "answered" if answer.get("is_answerable") else "not_found_in_context"
    return answer


def _render_computed_answer(computed: dict[str, Any], context_package: dict[str, Any]) -> str:
    result = computed.get("result")
    if result is None:
        return str(computed.get("answer", "") or "")
    operation = str(computed.get("operation", ""))
    operands = computed.get("operand_sources") or computed.get("operands") or []
    operand_text = "와 ".join(
        _format_won(item.get("won"))
        for item in operands
        if isinstance(item, dict) and item.get("won") is not None
    )
    question = normalize_text(context_package.get("question", ""))
    wants_steps = has_any(question, ["계산 과정", "어떻게 계산", "산출 과정", "풀이"])
    if wants_steps and computed.get("answer"):
        return str(computed.get("answer", ""))
    if operation == "difference":
        suffix = f" 계산 기준은 {operand_text}입니다." if operand_text else ""
        return f"두 금액의 차액은 {_format_won(result)}입니다.{suffix}"
    if operation == "sum":
        count = len([item for item in operands if isinstance(item, dict) and item.get("won") is not None])
        subject = f"{count}개 금액" if count >= 3 else "두 금액"
        suffix = f" 계산 기준은 {operand_text}입니다." if operand_text else ""
        return f"{subject}을 합산하면 총 {_format_won(result)}입니다.{suffix}"
    if operation == "monthly_unit_after_deduction":
        return f"조건을 반영한 1인당 월 급여 환산액은 {_format_won(result)}입니다."
    if operation == "unit_price_after_deduction_and_fraction_spend":
        return f"조건을 반영한 1개당 배분 금액은 {_format_won(result)}입니다."
    return f"계산 결과는 {_format_won(result)}입니다."


def _merge_computed_with_existing_answer(computed_answer: str, existing_answer: str, analysis: dict[str, Any]) -> str:
    existing_answer = str(existing_answer or "").strip()
    if not existing_answer:
        return computed_answer
    sections = []
    if any(intent in set(analysis.get("intent_slots", [])) for intent in {"budget_sum", "budget_difference", "budget_ratio"}):
        sections.append(f"예산/계산: {computed_answer}")
    else:
        sections.append(computed_answer)
    if any(intent in set(analysis.get("intent_slots", [])) for intent in {"purpose_summary", "requirements_summary", "technical_requirement_lookup", "technical_purpose_summary"}):
        sections.append(f"핵심 요약: {existing_answer}")
    elif any(intent in set(analysis.get("intent_slots", [])) for intent in {"multi_doc_comparison", "risk_comparison"}):
        sections.append(f"비교/리스크: {existing_answer}")
    else:
        sections.append(existing_answer)
    return "\n".join(sections)


def _has_grounded_budget_fallback_for_missing_target(answer: dict[str, Any], context_package: dict[str, Any]) -> bool:
    analysis = context_package.get("question_analysis", {})
    targets = [
        slot
        for slot in analysis.get("target_slots", [])
        if "project_budget" in (slot.get("missing_fields") or []) and slot.get("matched_source_file")
    ]
    if not targets:
        return False
    answer_values = {value.get("won") for value in _extract_amount_values(answer.get("answer", "")) if value.get("won") is not None}
    if not answer_values:
        return False
    for target in targets:
        matched_source = _normalize_doc_key(target.get("matched_source_file", ""))
        for block_data in context_package.get("evidence_blocks", []):
            if _normalize_doc_key(block_data.get("source_file", "")) != matched_source:
                continue
            block = EvidenceBlock(
                source_file=str(block_data.get("source_file", "")),
                source_file_nfc=str(block_data.get("source_file_nfc", "")),
                chunk_id=str(block_data.get("chunk_id", "")),
                evidence_id=str(block_data.get("evidence_id", "")),
                rank=int(_safe_float(block_data.get("rank"), 999)),
                chunk_type=str(block_data.get("chunk_type", "")),
                fact_type=str(block_data.get("fact_type", "")),
                section_path=str(block_data.get("section_path", "")),
                text=str(block_data.get("text", "")),
                score=_safe_float(block_data.get("score"), 0.0),
                retrieval_role=str(block_data.get("retrieval_role", "")),
                answer_policy=str(block_data.get("answer_policy", "")),
                answer_risk_level=str(block_data.get("answer_risk_level", "")),
                budget_answer_enabled=_as_bool(block_data.get("budget_answer_enabled")),
                eligibility_answer_enabled=_as_bool(block_data.get("eligibility_answer_enabled")),
                payment_answer_enabled=_as_bool(block_data.get("payment_answer_enabled")),
                selection_stage=str(block_data.get("selection_stage", "")),
                is_backfilled=_as_bool(block_data.get("is_backfilled")),
            )
            amount = _extract_project_budget_amount_from_block(block, allow_fallback=True)
            if amount and amount.get("won") in answer_values:
                return True
    return False


def _has_total_allocation_context(context_package: dict[str, Any]) -> bool:
    for block_data in context_package.get("evidence_blocks", []):
        block = EvidenceBlock(
            source_file=str(block_data.get("source_file", "")),
            source_file_nfc=str(block_data.get("source_file_nfc", "")),
            chunk_id=str(block_data.get("chunk_id", "")),
            evidence_id=str(block_data.get("evidence_id", "")),
            rank=int(_safe_float(block_data.get("rank"), 999)),
            chunk_type=str(block_data.get("chunk_type", "")),
            fact_type=str(block_data.get("fact_type", "")),
            section_path=str(block_data.get("section_path", "")),
            text=str(block_data.get("text", "")),
            score=_safe_float(block_data.get("score"), 0.0),
            retrieval_role=str(block_data.get("retrieval_role", "")),
            answer_policy=str(block_data.get("answer_policy", "")),
            answer_risk_level=str(block_data.get("answer_risk_level", "")),
            budget_answer_enabled=_as_bool(block_data.get("budget_answer_enabled")),
            eligibility_answer_enabled=_as_bool(block_data.get("eligibility_answer_enabled")),
            payment_answer_enabled=_as_bool(block_data.get("payment_answer_enabled")),
            selection_stage=str(block_data.get("selection_stage", "")),
            is_backfilled=_as_bool(block_data.get("is_backfilled")),
        )
        if _extract_total_allocation_amount_from_block(block):
            return True
    return False


def _dedupe_overlapping_target_labels(labels: Iterable[Any]) -> list[str]:
    deduped: list[str] = []
    norm_labels: list[str] = []
    for label in labels:
        label_text = str(label or "").strip()
        norm = _normalize_doc_key(label_text)
        if not label_text or not norm:
            continue
        if any(norm in existing or existing in norm for existing in norm_labels):
            continue
        deduped.append(label_text)
        norm_labels.append(norm)
    return deduped


def _attach_deterministic_citations(
    answer: dict[str, Any],
    context_package: dict[str, Any],
    *,
    max_citations: int = 3,
) -> list[dict[str, str]]:
    if not answer.get("answer"):
        return []
    blocks = context_package.get("evidence_blocks", [])
    if not blocks:
        return []
    analysis = context_package.get("question_analysis", {})
    ranked = sorted(
        blocks,
        key=lambda block: _citation_priority(block, analysis, answer),
        reverse=True,
    )
    if answer.get("answer_status") == "not_found_in_context":
        matched_sources = {
            _normalize_doc_key(slot.get("matched_source_file", ""))
            for slot in analysis.get("target_slots", [])
            if slot.get("matched_source_file")
        }
        if matched_sources:
            ranked = [
                block
                for block in ranked
                if _normalize_doc_key(block.get("source_file", "")) in matched_sources
            ] or ranked
    citations = []
    used_chunks = set()
    used_sources = set()
    for block in ranked:
        chunk_id = str(block.get("chunk_id", ""))
        source_file = str(block.get("source_file", ""))
        text = str(block.get("text", "")).strip()
        if not text or chunk_id in used_chunks:
            continue
        if analysis.get("is_multi_doc") and source_file in used_sources and len(used_sources) < 2:
            continue
        citations.append(
            {
                "evidence_id": str(block.get("evidence_id") or chunk_id or f"E{len(citations) + 1}"),
                "source_file": source_file,
                "chunk_id": chunk_id,
                "evidence_text": truncate_text(text, 320),
            }
        )
        used_chunks.add(chunk_id)
        if source_file:
            used_sources.add(source_file)
        if len(citations) >= max_citations:
            break

    if len(citations) < min(max_citations, len(ranked)):
        for block in ranked:
            chunk_id = str(block.get("chunk_id", ""))
            text = str(block.get("text", "")).strip()
            if not text or chunk_id in used_chunks:
                continue
            citations.append(
                {
                    "evidence_id": str(block.get("evidence_id") or chunk_id or f"E{len(citations) + 1}"),
                    "source_file": str(block.get("source_file", "")),
                    "chunk_id": chunk_id,
                    "evidence_text": truncate_text(text, 320),
                }
            )
            used_chunks.add(chunk_id)
            if len(citations) >= max_citations:
                break
    return citations


def _citation_priority(
    block: dict[str, Any],
    analysis: dict[str, Any],
    answer: dict[str, Any],
) -> float:
    score = _safe_float(block.get("score"), 0.0)
    fact_type = str(block.get("fact_type", ""))
    answer_policy = str(block.get("answer_policy", ""))
    question_types = set(analysis.get("question_types", []))
    if block.get("is_backfilled"):
        score -= 20.0
    if fact_type == "document_identity" or answer_policy == "route_only_not_final_answer":
        score -= 80.0
    if "budget" in question_types or answer.get("answer_type") == "budget":
        if block.get("budget_answer_enabled"):
            score += 80.0
        if fact_type in FINAL_BUDGET_FACT_TYPES or ("total_allocation_lookup" in set(analysis.get("intent_slots", [])) and fact_type == "total_allocation"):
            score += 60.0
        if fact_type in BUDGET_BLOCKED_FACT_TYPES:
            score -= 120.0
    if "eligibility" in question_types and block.get("eligibility_answer_enabled"):
        score += 35.0
    if "submission_documents" in question_types and fact_type == "submission_documents":
        score += 35.0
    return score


def _validate_answer_policy(
    answer: dict[str, Any],
    context_package: dict[str, Any],
) -> dict[str, Any]:
    analysis = context_package.get("question_analysis", {})
    question_types = set(analysis.get("question_types", []))
    if "budget" not in question_types and answer.get("answer_type") != "budget":
        return {"policy_valid": True, "policy_violations": [], "policy_warnings": []}
    if "eligibility" in question_types:
        return {"policy_valid": True, "policy_violations": [], "policy_warnings": []}

    answer_values = {
        _normalize_value_for_grounding(value)
        for value in _extract_grounding_values(
            json.dumps(
                {
                    "answer": answer.get("answer", ""),
                    "final_values": answer.get("final_values", {}),
                },
                ensure_ascii=False,
            )
        )
    }
    if not answer_values:
        return {"policy_valid": True, "policy_violations": [], "policy_warnings": []}

    allowed_values: set[str] = set()
    blocked_values: set[str] = set()
    total_allocation_mode = "total_allocation_lookup" in set(analysis.get("intent_slots", []))
    for block in context_package.get("evidence_blocks", []):
        block_values = {
            _normalize_value_for_grounding(value)
            for value in _extract_grounding_values(block.get("text", ""))
        }
        fact_type = str(block.get("fact_type", ""))
        if total_allocation_mode:
            if fact_type == "total_allocation" or has_any(normalize_text(block.get("text", "")), ALLOCATION_BUDGET_CONTEXT_KEYWORDS):
                allowed_values.update(block_values)
            elif fact_type in FINAL_BUDGET_FACT_TYPES | BUDGET_BLOCKED_FACT_TYPES:
                blocked_values.update(block_values)
            continue
        if block.get("budget_answer_enabled") or fact_type in FINAL_BUDGET_FACT_TYPES:
            allowed_values.update(block_values)
        if fact_type in BUDGET_BLOCKED_FACT_TYPES:
            blocked_values.update(block_values)

    violations = sorted(value for value in answer_values if value in blocked_values and value not in allowed_values)
    return {
        "policy_valid": not violations,
        "policy_violations": violations,
        "policy_warnings": [f"blocked_budget_value_used:{value}" for value in violations],
    }


def _validate_citations(
    answer: dict[str, Any],
    context_package: dict[str, Any],
) -> dict[str, Any]:
    blocks = context_package.get("evidence_blocks", [])
    if not answer.get("is_answerable") and answer.get("answer_status") != "not_found_in_context":
        return {"citation_valid": True, "invalid_citations": []}
    if not answer.get("citations"):
        return {"citation_valid": False, "invalid_citations": ["missing_citations"]}

    valid_chunk_ids = {str(block.get("chunk_id", "")) for block in blocks}
    valid_evidence_ids = {str(block.get("evidence_id", "")) for block in blocks if block.get("evidence_id")}
    valid_source_files = {
        unicodedata.normalize("NFC", str(block.get("source_file", "")))
        for block in blocks
    }
    valid_source_files.update(
        unicodedata.normalize("NFC", str(block.get("source_file_nfc", "")))
        for block in blocks
        if block.get("source_file_nfc")
    )
    context_text = normalize_text(context_package.get("context_text", ""))
    invalid = []
    for citation in answer.get("citations", []):
        if isinstance(citation, str):
            chunk_match = re.search(r"chunk_id=([^|,\s]+)", citation)
            source_match = re.search(r"source_file=([^|,]+)", citation)
            if not source_match and "|" in citation:
                source_candidate = citation.split("|", 1)[0].strip()
                source_match = re.match(r"(.+)", source_candidate)
            evidence_match = re.search(r"evidence_id=([^|,\s]+)", citation)
            evidence_id = evidence_match.group(1).strip() if evidence_match else ""
            chunk_id = chunk_match.group(1).strip() if chunk_match else ""
            source_file = source_match.group(1).strip() if source_match else ""
            evidence_text = ""
        elif isinstance(citation, dict):
            evidence_id = str(citation.get("evidence_id", ""))
            chunk_id = str(citation.get("chunk_id", ""))
            source_file = str(citation.get("source_file", ""))
            evidence_text = normalize_text(citation.get("evidence_text", ""))
        else:
            invalid.append("non_dict_citation")
            continue
        if evidence_id and evidence_id not in valid_evidence_ids and evidence_id not in valid_chunk_ids:
            invalid.append(f"unknown_evidence_id:{evidence_id}")
        if not evidence_id and not chunk_id and not source_file and not evidence_text:
            invalid.append("unparseable_citation")
            continue
        if chunk_id and chunk_id not in valid_chunk_ids:
            invalid.append(f"unknown_chunk_id:{chunk_id}")
        if source_file and unicodedata.normalize("NFC", source_file) not in valid_source_files:
            invalid.append(f"unknown_source_file:{source_file}")
        if evidence_text and evidence_text[:50] not in context_text:
            invalid.append("evidence_text_not_in_context")
    return {"citation_valid": not invalid, "invalid_citations": invalid}


def _validate_numeric_grounding(
    answer: dict[str, Any],
    context_package: dict[str, Any],
) -> dict[str, Any]:
    answer_text = json.dumps(
        {
            "answer": answer.get("answer", ""),
            "final_values": answer.get("final_values", {}),
            "documents": answer.get("documents", []),
        },
        ensure_ascii=False,
    )
    context_text = context_package.get("context_text", "")
    question_text = context_package.get("question", "")
    computed = context_package.get("core_summary", {}).get("computed_values") or {}
    values = _extract_grounding_values(answer_text)
    context_norm = _normalize_value_for_grounding(context_text)
    question_norm = _normalize_value_for_grounding(question_text)
    context_equiv = _equivalent_amount_norms(context_text)
    question_equiv = _equivalent_amount_norms(question_text)
    derived_norms = {
        _normalize_value_for_grounding(value)
        for value in _extract_grounding_values(json.dumps(computed, ensure_ascii=False))
    }
    derived_norms.update(_equivalent_amount_norms(json.dumps(computed, ensure_ascii=False)))
    ungrounded = []
    source_missing = []
    derived_values = []
    for value in values:
        norm = _normalize_value_for_grounding(value)
        value_equiv = _equivalent_amount_norms(value)
        if norm in context_norm or value_equiv & context_equiv:
            continue
        if norm in question_norm or value_equiv & question_equiv:
            continue
        if norm in derived_norms:
            derived_values.append(value)
            continue
        ungrounded.append(value)
        source_missing.append(value)
    derived_valid = True
    if computed.get("result") is not None:
        expected = _normalize_value_for_grounding(_format_won(computed.get("result")))
        answer_norm = _normalize_value_for_grounding(answer_text)
        derived_valid = expected in answer_norm or expected in derived_norms
    return {
        "numeric_grounded": not ungrounded,
        "source_numeric_grounded": not source_missing,
        "derived_numeric_valid": derived_valid,
        "ungrounded_values": ungrounded,
        "derived_values": derived_values,
    }


def _question_amounts_are_approximate(question: str) -> bool:
    q = normalize_text(question)
    if has_any(q, ["약", "대략", "정도", "가량", "쯤", "대강"]):
        return True
    return bool(re.search(r"\d+\.\d+\s*(?:억|백만원|천만원|만원)", str(question or "")))


def _compute_deterministic_values(
    question: str,
    analysis: dict[str, Any],
    evidence_blocks: list[EvidenceBlock] | None = None,
) -> dict[str, Any]:
    intents = set(analysis.get("intent_slots", []))
    question_amounts = _tag_amount_operands(_extract_amount_values(question), source="question")
    context_amounts = _collect_context_budget_operands(evidence_blocks or [], analysis)
    percents = [float(value) / 100.0 for value in PERCENT_RE.findall(question)]
    result: float | None = None
    operation = ""
    formula = ""
    steps: list[str] = []
    operand_sources: list[dict[str, Any]] = []

    def choose_amounts(required_count: int, *, prefer_context: bool = False) -> list[dict[str, Any]]:
        if prefer_context and len(context_amounts) >= required_count:
            return context_amounts[:required_count]
        if len(question_amounts) >= required_count:
            return question_amounts[:required_count]
        return _dedupe_amount_operands(question_amounts + context_amounts)[:required_count]

    prefer_context_for_target_math = bool(analysis.get("target_slots") and len(context_amounts) >= 2)
    prefer_context_for_target_ratio = bool(
        analysis.get("target_slots")
        and context_amounts
        and _question_amounts_are_approximate(question)
    )

    if "budget_difference" in intents:
        operands = choose_amounts(2, prefer_context=prefer_context_for_target_math)
        if len(operands) >= 2:
            left, right = operands[0]["won"], operands[1]["won"]
            result = abs(left - right)
            operation = "difference"
            bigger, smaller = (left, right) if left >= right else (right, left)
            steps = [f"{_format_won(bigger)} - {_format_won(smaller)} = {_format_won(result)}"]
            formula = f"abs({operands[0]['raw']} - {operands[1]['raw']})"
            operand_sources = operands
    elif "budget_sum" in intents:
        required_count = _expected_sum_operand_count(question, analysis)
        operands = choose_amounts(required_count, prefer_context=prefer_context_for_target_math)
        if len(operands) >= 2:
            result = sum(item["won"] for item in operands)
            operation = "sum"
            joined = " + ".join(_format_won(item["won"]) for item in operands)
            steps = [f"{joined} = {_format_won(result)}"]
            formula = " + ".join(item["raw"] for item in operands)
            operand_sources = operands
    elif "budget_ratio" in intents:
        operands = choose_amounts(1, prefer_context=prefer_context_for_target_ratio)
        if operands:
            base = operands[0]["won"]
            q = normalize_text(question)
            fraction = _extract_last_fraction(question) if "나머지" in q else _extract_first_fraction(question)
            if "월급" in q:
                people_match = re.search(r"(\d+)\s*명", question)
                month_match = re.search(r"(\d+)\s*개월", question)
                first_percent = percents[0] if percents else 0.0
                if people_match and month_match:
                    people = int(people_match.group(1))
                    months = int(month_match.group(1))
                    first_deduction = base * first_percent
                    remaining = base - first_deduction
                    per_person = remaining / people
                    result = per_person / months
                    operation = "monthly_unit_after_deduction"
                    formula = f"({operands[0]['raw']} × (1 - {_format_percent(first_percent)})) ÷ {people}명 ÷ {months}개월"
                    steps = [
                        f"{_format_won(base)} × {_format_percent(first_percent)} = {_format_won(first_deduction)}",
                        f"{_format_won(base)} - {_format_won(first_deduction)} = {_format_won(remaining)}",
                        f"{_format_won(remaining)} ÷ {people}명 = {_format_won(per_person)}",
                        f"{_format_won(per_person)} ÷ {months}개월 = {_format_won(result)}",
                    ]
            elif "남길" in q and len(percents) >= 2:
                first_deduction = base * percents[0]
                remaining = base - first_deduction
                second_deduction = remaining * percents[1]
                result = remaining - second_deduction
                operation = "remaining_after_two_percent_deductions"
                formula = f"{operands[0]['raw']} × (1 - {_format_percent(percents[0])}) × (1 - {_format_percent(percents[1])})"
                steps = [
                    f"{_format_won(base)} × {_format_percent(percents[0])} = {_format_won(first_deduction)}",
                    f"{_format_won(base)} - {_format_won(first_deduction)} = {_format_won(remaining)}",
                    f"{_format_won(remaining)} × {_format_percent(percents[1])} = {_format_won(second_deduction)}",
                    f"{_format_won(remaining)} - {_format_won(second_deduction)} = {_format_won(result)}",
                ]
            elif "단가" in q or "라이선스" in q:
                count_match = re.search(r"(\d+)\s*개", question)
                first_percent = percents[0] if percents else 0.0
                fraction = fraction or (1, 0)
                if count_match:
                    count = int(count_match.group(1))
                    spent_fraction = fraction[1] / fraction[0]
                    first_deduction = base * first_percent
                    remaining = base - first_deduction
                    fraction_spend = remaining * spent_fraction
                    final_pool = remaining - fraction_spend
                    result = final_pool / count
                    operation = "unit_price_after_deduction_and_fraction_spend"
                    formula = f"({operands[0]['raw']} × (1 - {_format_percent(first_percent)}) × (1 - {fraction[1]}/{fraction[0]})) ÷ {count}개"
                    steps = [
                        f"{_format_won(base)} × {_format_percent(first_percent)} = {_format_won(first_deduction)}",
                        f"{_format_won(base)} - {_format_won(first_deduction)} = {_format_won(remaining)}",
                        f"{_format_won(remaining)} × {fraction[1]}/{fraction[0]} = {_format_won(fraction_spend)}",
                        f"{_format_won(remaining)} - {_format_won(fraction_spend)} = {_format_won(final_pool)}",
                        f"{_format_won(final_pool)} ÷ {count}개 = {_format_won(result)}",
                    ]
            elif fraction and any(token in q for token in ["나머지", "신규", "코딩", "개발"]):
                result = base * fraction[1] / fraction[0]
                operation = "fraction_of_budget"
                formula = f"{operands[0]['raw']} × {fraction[1]}/{fraction[0]}"
                steps = [f"{_format_won(base)} × {fraction[1]}/{fraction[0]} = {_format_won(result)}"]
            elif percents:
                result = base * percents[0]
                operation = "percent_of_budget"
                formula = f"{operands[0]['raw']} × {_format_percent(percents[0])}"
                steps = [f"{_format_won(base)} × {_format_percent(percents[0])} = {_format_won(result)}"]
            elif fraction:
                result = base * fraction[1] / fraction[0]
                operation = "fraction_of_budget"
                formula = f"{operands[0]['raw']} × {fraction[1]}/{fraction[0]}"
                steps = [f"{_format_won(base)} × {fraction[1]}/{fraction[0]} = {_format_won(result)}"]
            operand_sources = operands
    if result is None:
        return {
            "operation": "",
            "operands": question_amounts,
            "context_operands": context_amounts,
            "operand_sources": [],
            "percents": percents,
            "formula": "",
            "steps": [],
            "result": None,
            "answer": "",
        }
    rounded = int(round(result))
    rounded_steps = [_normalize_step_amounts(step) for step in steps]
    return {
        "operation": operation,
        "operands": operand_sources,
        "context_operands": context_amounts,
        "operand_sources": operand_sources,
        "percents": percents,
        "formula": formula,
        "steps": rounded_steps,
        "result": rounded,
        "answer": _format_calculation_answer(operation, rounded_steps, rounded),
    }


def _expected_sum_operand_count(question: str, analysis: dict[str, Any]) -> int:
    target_count = len([slot for slot in analysis.get("target_slots", []) if slot.get("target_label")])
    if target_count >= 2:
        return target_count
    q = normalize_text(question)
    count_markers = [
        ("4개", 4),
        ("4곳", 4),
        ("네 곳", 4),
        ("네 사업", 4),
        ("3개", 3),
        ("3곳", 3),
        ("세 곳", 3),
        ("세 사업", 3),
    ]
    for marker, count in count_markers:
        if marker in q:
            return count
    return 2


def _tag_amount_operands(amounts: list[dict[str, Any]], *, source: str) -> list[dict[str, Any]]:
    return [
        {
            **amount,
            "source": source,
            "source_file": "",
            "fact_type": "",
            "chunk_id": "",
            "target_label": "",
        }
        for amount in amounts
    ]


def _collect_context_budget_operands(
    blocks: list[EvidenceBlock],
    analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    target_slots = [slot for slot in analysis.get("target_slots", []) if slot.get("matched_source_file")]
    if not target_slots:
        return []
    operands: list[dict[str, Any]] = []
    for slot in target_slots:
        matched_source = _normalize_doc_key(slot.get("matched_source_file", ""))
        source_blocks = [
            block
            for block in blocks
            if _normalize_doc_key(block.source_file) == matched_source
        ]
        if "total_allocation_lookup" in set(analysis.get("intent_slots", [])):
            allocation_candidates = [
                (block, "context_allocation")
                for block in source_blocks
                if _extract_total_allocation_amount_from_block(block)
            ]
            ranked_candidates = sorted(
                allocation_candidates,
                key=lambda item: item[0].score,
                reverse=True,
            )
        else:
            allowed_blocks = [block for block in source_blocks if _is_allowed_budget_operand_block(block)]
            fallback_blocks = [block for block in source_blocks if _is_budget_fallback_candidate_block(block)]
            ranked_candidates = sorted(
                [(block, "context") for block in allowed_blocks] + [(block, "context_fallback") for block in fallback_blocks],
                key=lambda item: item[0].score + (50.0 if item[1] == "context" else 0.0),
                reverse=True,
            )
        for block, source_kind in ranked_candidates:
            if source_kind == "context_allocation":
                amount = _extract_total_allocation_amount_from_block(block)
            else:
                amount = _extract_project_budget_amount_from_block(block, allow_fallback=source_kind == "context_fallback")
            if not amount:
                continue
            operands.append(
                {
                    **amount,
                    "source": source_kind,
                    "source_file": block.source_file,
                    "fact_type": block.fact_type,
                    "chunk_id": block.chunk_id,
                    "target_label": slot.get("target_label", ""),
                    "budget_operand_role": amount.get("budget_operand_role", "project_budget"),
                }
            )
            break
    return _dedupe_amount_operands(operands)


def _has_project_budget_operand(blocks: list[EvidenceBlock]) -> bool:
    return any(
        _is_allowed_budget_operand_block(block)
        or _is_budget_fallback_candidate_block(block)
        for block in blocks
    )


def _has_total_allocation_operand(blocks: list[EvidenceBlock]) -> bool:
    return any(_extract_total_allocation_amount_from_block(block) for block in blocks)


def _is_budget_fallback_candidate_block(block: EvidenceBlock) -> bool:
    if block.fact_type in BUDGET_BLOCKED_FACT_TYPES:
        return False
    if block.answer_policy == "route_only_not_final_answer":
        return False
    if block.fact_type in FINAL_BUDGET_FACT_TYPES and not _is_allowed_budget_operand_block(block):
        return False
    if block.chunk_type not in {"text", "table", "fact_candidates", ""}:
        return False
    return _extract_project_budget_amount_from_block(block, allow_fallback=True) is not None


def _extract_project_budget_amount_from_block(block: EvidenceBlock, *, allow_fallback: bool) -> dict[str, Any] | None:
    if not allow_fallback and not _is_allowed_budget_operand_block(block):
        return None
    if _is_allowed_budget_operand_block(block):
        amount = _extract_first_amount(block.text)
        if amount:
            amount["budget_operand_role"] = "project_budget_fact"
        return amount
    return _extract_strong_budget_context_amount(block.text)


def _extract_total_allocation_amount_from_block(block: EvidenceBlock) -> dict[str, Any] | None:
    if block.fact_type == "total_allocation":
        amount = _extract_first_amount(block.text)
        if amount:
            amount["budget_operand_role"] = "total_allocation_fact"
        return amount
    return _extract_budget_amount_near_keywords(block.text, ALLOCATION_BUDGET_CONTEXT_KEYWORDS, blocked_keywords=[])


def _extract_first_amount(text: str) -> dict[str, Any] | None:
    amounts = _extract_amount_values(text)
    return amounts[0] if amounts else None


def _extract_strong_budget_context_amount(text: str) -> dict[str, Any] | None:
    return _extract_budget_amount_near_keywords(
        text,
        STRONG_PROJECT_BUDGET_CONTEXT_KEYWORDS,
        blocked_keywords=BLOCKED_BUDGET_FALLBACK_CONTEXT_KEYWORDS,
    )


def _extract_budget_amount_near_keywords(
    text: str,
    keywords: Iterable[str],
    *,
    blocked_keywords: Iterable[str],
) -> dict[str, Any] | None:
    raw_text = str(text or "")
    normalized = normalize_text(raw_text)
    if not AMOUNT_RE.search(raw_text):
        return None
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for match in AMOUNT_RE.finditer(raw_text):
        start = max(0, match.start() - 80)
        end = min(len(raw_text), match.end() + 80)
        window = raw_text[start:end]
        normalized_window = normalize_text(window)
        if not has_any(normalized_window, keywords):
            continue
        if blocked_keywords and has_any(normalized_window, blocked_keywords):
            continue
        amount = _amount_to_won(match.group(0))
        if amount is None or amount <= 10_000:
            continue
        priority = _budget_context_priority(normalized_window)
        candidates.append((priority, -match.start(), {"raw": match.group(0), "won": amount, "budget_operand_role": "strong_text_or_table_budget_context"}))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def _budget_context_priority(normalized_window: str) -> int:
    priority_keywords = [
        "전체 배정액",
        "전체배정액",
        "배정액",
        "배정예산",
        "사업예산",
        "사업 예산",
        "사업비",
        "총사업비",
        "총 사업비",
        "사업금액",
        "사업 금액",
        "소요예산",
        "소요 예산",
        "용역금액",
        "용역 금액",
        "용역비",
        "추정가격",
        "추정 가격",
    ]
    return max((len(priority_keywords) - idx for idx, keyword in enumerate(priority_keywords) if keyword in normalized_window), default=0)


def _is_allowed_budget_operand_block(block: EvidenceBlock) -> bool:
    if block.fact_type not in (FINAL_BUDGET_FACT_TYPES | ALLOCATION_BUDGET_FACT_TYPES):
        return False
    if block.fact_type in BUDGET_BLOCKED_FACT_TYPES:
        return False
    if block.answer_policy == "route_only_not_final_answer":
        return False
    return _as_bool(block.budget_answer_enabled) or block.answer_policy == "allow_as_project_budget"


def _dedupe_amount_operands(operands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for operand in operands:
        key = (
            operand.get("won"),
            operand.get("source"),
            operand.get("source_file"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(operand)
    return deduped


def _format_percent(value: float) -> str:
    return f"{value * 100:g}%"


def _normalize_step_amounts(step: str) -> str:
    return re.sub(
        r"(?<![\d,])(\d+)(?:\.\d+)?원",
        lambda match: _format_won(match.group(1)),
        step,
    )


def _format_calculation_answer(operation: str, steps: list[str], result: int) -> str:
    step_lines = [f"{index}. {step}" for index, step in enumerate(steps, start=1)]
    if operation == "difference":
        conclusion = f"따라서 차액은 {_format_won(result)}입니다."
    elif operation == "sum":
        conclusion = f"따라서 합계는 {_format_won(result)}입니다."
    elif operation == "monthly_unit_after_deduction":
        conclusion = f"따라서 1인당 월 급여 환산액은 {_format_won(result)}입니다."
    elif operation == "unit_price_after_deduction_and_fraction_spend":
        conclusion = f"따라서 1개당 배분 단가는 {_format_won(result)}입니다."
    else:
        conclusion = f"따라서 계산 결과는 {_format_won(result)}입니다."
    return "계산 과정:\n" + "\n".join(step_lines + [conclusion])


def _extract_amount_values(text: str) -> list[dict[str, Any]]:
    values = []
    for match in NUMERIC_AMOUNT_RE.finditer(str(text or "")):
        raw = match.group(0)
        won = _amount_to_won(raw)
        if won is not None:
            values.append({"raw": raw, "won": won})
    return values


def _amount_to_won(raw: str) -> int | None:
    value = str(raw or "").replace(",", "").replace(" ", "")
    number_match = re.search(r"\d+(?:\.\d+)?", value)
    if not number_match:
        return None
    number = float(number_match.group(0))
    if "조" in value:
        number *= 1_000_000_000_000
    elif "억" in value:
        number *= 100_000_000
    elif "백만원" in value:
        number *= 1_000_000
    elif "백만" in value:
        number *= 1_000_000
    elif "천만원" in value:
        number *= 10_000_000
    elif "천만" in value:
        number *= 10_000_000
    elif "만원" in value:
        number *= 10_000
    elif "천원" in value:
        number *= 1_000
    return int(round(number))


def _format_won(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,}원"
    except (TypeError, ValueError):
        return str(value)


def _extract_first_fraction(text: str) -> tuple[int, int] | None:
    match = FRACTION_RE.search(str(text or ""))
    if not match:
        return None
    denominator = int(match.group(1))
    numerator = int(match.group(2))
    if denominator == 0:
        return None
    return denominator, numerator


def _extract_last_fraction(text: str) -> tuple[int, int] | None:
    matches = list(FRACTION_RE.finditer(str(text or "")))
    if not matches:
        return None
    match = matches[-1]
    denominator = int(match.group(1))
    numerator = int(match.group(2))
    if denominator == 0:
        return None
    return denominator, numerator


def _extract_grounding_values(text: str) -> list[str]:
    values = []
    values.extend(AMOUNT_RE.findall(text))
    values.extend(DATE_RE.findall(text))
    values.extend(DURATION_RE.findall(text))
    return _unique_preserve_order(values)


def _normalize_value_for_grounding(text: str) -> str:
    return re.sub(r"[\s,]", "", str(text or ""))


def _equivalent_amount_norms(text: str) -> set[str]:
    norms: set[str] = set()
    for amount in _extract_amount_values(text):
        raw = amount.get("raw")
        won = amount.get("won")
        if raw:
            norms.add(_normalize_value_for_grounding(raw))
        if won is not None:
            norms.add(_normalize_value_for_grounding(_format_won(won)))
            norms.add(str(int(won)))
    return norms


def _is_incomplete_multi_doc(
    answer: dict[str, Any],
    context_package: dict[str, Any],
) -> bool:
    analysis = context_package.get("question_analysis", {})
    if not analysis.get("is_multi_doc"):
        return False
    evidence_docs = {
        block.get("source_file", "")
        for block in context_package.get("evidence_blocks", [])
        if block.get("source_file")
    }
    if len(evidence_docs) < 2:
        return False
    answer_docs = answer.get("documents") if isinstance(answer.get("documents"), list) else []
    return len(answer_docs) < min(2, len(evidence_docs))


def _has_wrong_target_citation(answer: dict[str, Any], context_package: dict[str, Any]) -> bool:
    target_slots = context_package.get("question_analysis", {}).get("target_slots", [])
    if not target_slots or not answer.get("citations"):
        return False
    allowed_sources = {
        _normalize_doc_key(slot.get("matched_source_file", ""))
        for slot in target_slots
        if slot.get("matched_source_file")
    }
    if not allowed_sources:
        return False
    for citation in answer.get("citations", []):
        source = citation.get("source_file", "") if isinstance(citation, dict) else ""
        if source and _normalize_doc_key(source) not in allowed_sources:
            return True
    return False


def _has_wrong_target_field_selection(answer: dict[str, Any], context_package: dict[str, Any]) -> bool:
    if not answer.get("final_values"):
        return False
    analysis = context_package.get("question_analysis", {})
    if not analysis.get("target_slots"):
        return False
    if answer.get("answer_status") in ANSWERABLE_NEGATIVE_STATUSES:
        return False
    return _has_wrong_target_citation(answer, context_package)


def _missing_intents(answer: dict[str, Any], context_package: dict[str, Any]) -> list[str]:
    if answer.get("answer_status") in {"not_found_in_context", "insufficient_context", "retrieval_context_missing"}:
        return []
    analysis = context_package.get("question_analysis", {})
    intent_plan = analysis.get("intent_plan", []) or []
    if len(intent_plan) <= 1:
        return []
    answer_text = normalize_text(answer.get("answer", ""))
    missing = []
    for plan in intent_plan:
        intent = str(plan.get("intent", ""))
        if not _answer_covers_intent(answer_text, intent):
            missing.append(intent)
    return _unique_preserve_order(missing)


def _answer_covers_intent(answer_text: str, intent: str) -> bool:
    if intent in {"budget_lookup", "budget_difference", "budget_sum", "budget_ratio", "total_allocation_lookup"}:
        return bool(_extract_grounding_values(answer_text))
    if intent in {"purpose_summary", "requirements_summary", "technical_purpose_summary"}:
        return has_any(answer_text, ["목적", "배경", "효과", "효용", "전략", "현장", "r&d", "연구", "핵심", "요약", "요구"])
    if intent == "technical_requirement_lookup":
        return has_any(answer_text, ["db", "데이터베이스", "쿼리", "캐시", "캐싱", "튜닝", "최적화", "성능", "통계", "병목", "오프라인", "재연결", "동기화"])
    if intent == "risk_comparison":
        return has_any(answer_text, ["리스크", "피해", "장애", "비교", "차이", "파장", "복구", "안전"])
    if intent == "requirements_list":
        return has_any(answer_text, ["1.", "2.", "-", "·", "범위", "대상", "기능", "요구"])
    if intent == "multi_doc_comparison":
        return has_any(answer_text, ["비교", "차이", "공통", "각각", "반면", "문서", "사업"])
    if intent == "negative_check":
        return has_any(answer_text, ["없", "확인", "명시", "포함", "필요", "아닙"])
    if intent == "duration_lookup":
        return bool(DURATION_RE.search(answer_text) or DATE_RE.search(answer_text))
    if intent in {"submission_documents", "submission_logistics"}:
        return has_any(answer_text, ["제출", "서류", "제안서", "방문", "이메일", "우편", "기한"])
    if intent == "eligibility_check":
        return has_any(answer_text, ["자격", "실적", "인증", "공동수급", "입찰"])
    return True


def _missing_question_aspects(answer: dict[str, Any], context_package: dict[str, Any]) -> list[str]:
    if answer.get("answer_status") in {"not_found_in_context", "insufficient_context", "retrieval_context_missing"}:
        return []
    intents = set(context_package.get("question_analysis", {}).get("intent_slots", []))
    if not ({"purpose_summary", "requirements_summary", "technical_requirement_lookup", "technical_purpose_summary", "risk_comparison"} & intents):
        return []
    question_text = normalize_text(context_package.get("question", ""))
    answer_text = normalize_text(answer.get("answer", ""))
    missing: list[str] = []
    for rule in QUESTION_ASPECT_REQUIREMENTS:
        if has_any(question_text, rule["question_markers"]) and not has_any(answer_text, rule["answer_markers"]):
            missing.append(str(rule["aspect"]))
    return _unique_preserve_order(missing)


def _is_multi_intent_incomplete(answer: dict[str, Any], context_package: dict[str, Any]) -> bool:
    if answer.get("answer_status") in {"not_found_in_context", "insufficient_context", "retrieval_context_missing"}:
        return False
    intents = set(context_package.get("question_analysis", {}).get("intent_slots", []))
    if len(intents) <= 1:
        return False
    answer_text = normalize_text(answer.get("answer", ""))
    if "purpose_summary" in intents and not has_any(answer_text, ["목적", "효과", "현장", "r&d", "연구", "핵심", "요약"]):
        return True
    if ("technical_requirement_lookup" in intents) and not _answer_covers_intent(answer_text, "technical_requirement_lookup"):
        return True
    if ("risk_comparison" in intents) and not _answer_covers_intent(answer_text, "risk_comparison"):
        return True
    if any(intent.startswith("budget") or intent == "total_allocation_lookup" for intent in intents) and not _extract_grounding_values(answer_text):
        return True
    return False


def _has_target_doc_coverage_missing(context_package: dict[str, Any]) -> bool:
    target_slots = context_package.get("question_analysis", {}).get("target_slots", [])
    return any(slot.get("target_label") and not slot.get("matched_source_file") for slot in target_slots)


def _has_target_required_field_missing(context_package: dict[str, Any], field_name: str) -> bool:
    target_slots = context_package.get("question_analysis", {}).get("target_slots", [])
    return any(field_name in (slot.get("missing_fields") or []) for slot in target_slots)


def enrich_generation_record(
    answer: dict[str, Any],
    item: dict[str, Any],
    context_package: dict[str, Any],
    *,
    generation_ms: float | None = None,
    model_name: str = "",
    experiment_name: str = "",
    run_timestamp: str = "",
) -> dict[str, Any]:
    result = item.get("result", {}) if isinstance(item.get("result"), dict) else {}
    evidence_blocks = context_package.get("evidence_blocks", [])
    source_files = _unique_preserve_order(
        block.get("source_file", "") for block in evidence_blocks if block.get("source_file")
    )
    chunk_ids = _unique_preserve_order(
        block.get("chunk_id", "") for block in evidence_blocks if block.get("chunk_id")
    )
    contexts = [block.get("text", "") for block in evidence_blocks if block.get("text")]

    record = {
        "question_id": item.get("question_id") or result.get("id") or result.get("question_id") or "",
        "question": item.get("question") or result.get("question") or "",
        "ground_truth": result.get("ground_truth_answer") or result.get("ground_truth") or "",
        "ground_truth_docs": result.get("ground_truth_docs", ""),
        "retrieved_docs_top5": result.get("retrieved_docs_top5", ""),
        "model_name": model_name,
        "experiment_name": experiment_name,
        "run_timestamp": run_timestamp,
        "generation_ms": generation_ms,
        "context_text": context_package.get("context_text", ""),
        "contexts": contexts,
        "source_files": source_files,
        "chunk_ids": chunk_ids,
        "evidence_ids": _unique_preserve_order(
            block.get("evidence_id", "") for block in evidence_blocks if block.get("evidence_id")
        ),
        "question_analysis": context_package.get("question_analysis", {}),
        "core_summary": context_package.get("core_summary", {}),
        "target_slots": context_package.get("question_analysis", {}).get("target_slots", []),
        "intent_slots": context_package.get("question_analysis", {}).get("intent_slots", []),
        "intent_plan": context_package.get("question_analysis", {}).get("intent_plan", []),
        "computed_values": context_package.get("core_summary", {}).get("computed_values", {}),
        "use_source_store": context_package.get("use_source_store", False),
    }
    record.update(answer)
    record = _annotate_ground_truth_review(record)
    return record


REVIEW_AMOUNT_RE = re.compile(
    r"(?<!\d)(?:\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:조\s*원|억원|억\s*원|억|백만원|백만|천만원|천만|만원|천원|원)"
)
NEGATIVE_GT_MARKERS = ["미기재", "없", "않", "비공개", "확인되지", "명시되어 있지", "무관", "상정되지", "발견되지"]
NOT_FOUND_ANSWER_MARKERS = ["알 수 없습니다", "확인할 수 없습니다", "근거를 확인할 수 없어", "문서에 없습니다", "명시되어 있지 않습니다", "찾을 수 없습니다"]
REVIEW_TOKEN_STOPWORDS = {
    "사업", "문서", "해당", "관련", "대한", "위한", "으로", "에서", "하고", "하는", "합니다",
    "있습니다", "입니다", "것으로", "그리고", "또는", "통해", "기반", "구축", "용역",
}


def _annotate_ground_truth_review(record: dict[str, Any]) -> dict[str, Any]:
    """Add review-only diagnostics when an eval ground truth is available."""
    ground_truth = str(record.get("ground_truth") or "").strip()
    if not ground_truth:
        return record
    answer = str(record.get("answer") or "")
    existing_tags = list(record.get("_failure_tags", []))
    review_tags: list[str] = []

    if _is_budget_review_record(record):
        gt_amounts = _extract_amount_won_values_for_review(ground_truth)
        answer_amounts = _extract_amount_won_values_for_review(answer)
        if gt_amounts and answer_amounts and not _amount_lists_overlap(gt_amounts, answer_amounts):
            review_tags.append("gt_numeric_mismatch")

    if _looks_like_not_found_answer(answer) and not _looks_like_negative_ground_truth(ground_truth):
        review_tags.append("gt_expected_answer_but_model_not_found")

    if not existing_tags and not review_tags and str(record.get("confidence") or "") == "high":
        if not _looks_like_negative_ground_truth(ground_truth) and not _amounts_match_answer_and_gt(answer, ground_truth):
            recall = _review_token_recall(answer, ground_truth)
            if recall < 0.12:
                review_tags.append("gt_semantic_overlap_low")

    if review_tags:
        record["_gt_review_tags"] = _unique_preserve_order(list(record.get("_gt_review_tags", [])) + review_tags)
        record["_failure_tags"] = _unique_preserve_order(existing_tags + review_tags)
        record = _downgrade_confidence_for_failure_tags(record, record["_failure_tags"])
    return record


def _is_budget_review_record(record: dict[str, Any]) -> bool:
    intents = set(record.get("intent_slots", []) or [])
    answer_type = str(record.get("answer_type") or "")
    question = normalize_text(record.get("question", ""))
    return answer_type == "budget" or any(intent.startswith("budget") for intent in intents) or has_any(question, QUESTION_KEYWORDS["budget"])


def _extract_amount_won_values_for_review(text: str) -> list[int]:
    values: list[int] = []
    for match in REVIEW_AMOUNT_RE.finditer(str(text or "")):
        won = _amount_to_won(match.group(0))
        if won is not None:
            values.append(won)
    return _unique_preserve_order(values)


def _amount_lists_overlap(left: list[int], right: list[int]) -> bool:
    return any(abs(int(a) - int(b)) <= 1 for a in left for b in right)


def _amounts_match_answer_and_gt(answer: str, ground_truth: str) -> bool:
    gt_amounts = _extract_amount_won_values_for_review(ground_truth)
    answer_amounts = _extract_amount_won_values_for_review(answer)
    return bool(gt_amounts and answer_amounts and _amount_lists_overlap(gt_amounts, answer_amounts))


def _looks_like_negative_ground_truth(text: str) -> bool:
    normalized = normalize_text(text)
    return has_any(normalized, NEGATIVE_GT_MARKERS)


def _looks_like_not_found_answer(text: str) -> bool:
    normalized = normalize_text(text)
    return has_any(normalized, NOT_FOUND_ANSWER_MARKERS)


def _review_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", normalize_text(text))
    return {token for token in tokens if token not in REVIEW_TOKEN_STOPWORDS and not token.isdigit()}


def _review_token_recall(answer: str, ground_truth: str) -> float:
    gt_tokens = _review_tokens(ground_truth)
    if not gt_tokens:
        return 1.0
    answer_tokens = _review_tokens(answer)
    return len(gt_tokens & answer_tokens) / len(gt_tokens)


def create_generation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "total_questions": 0,
            "total": 0,
            "valid_json_count": 0,
            "valid_json_rate": math.nan,
            "citation_checked_count": 0,
            "citation_valid_rate": math.nan,
            "numeric_grounded_checked_count": 0,
            "numeric_grounded_rate": math.nan,
            "source_numeric_grounded_rate": math.nan,
            "derived_numeric_valid_rate": math.nan,
            "answerable_count": 0,
            "answerable_rate": math.nan,
            "empty_answer_count": 0,
            "empty_answer_rate": math.nan,
            "recovered_answer_count": 0,
            "recovered_answer_rate": math.nan,
            "parse_error_type_counts": {},
            "generation_ms_avg": math.nan,
            "failure_tag_counts": {},
        }

    failure_counter: Counter[str] = Counter()
    for record in records:
        failure_counter.update(record.get("_failure_tags", []))
    valid_json_count = sum(bool(record.get("_valid_json")) for record in records)
    empty_answer_count = sum(not str(record.get("answer", "")).strip() for record in records)
    recovered_answer_count = sum(bool(record.get("_recovered_answer")) for record in records)
    parse_error_counter = Counter(
        str(record.get("_parse_error_type", "") or "valid_json")
        for record in records
    )
    citation_checked_count = sum(record.get("_citation_valid") is not None for record in records)
    numeric_grounded_checked_count = sum(record.get("_numeric_grounded") is not None for record in records)
    source_numeric_grounded_checked_count = sum(record.get("_source_numeric_grounded") is not None for record in records)
    derived_numeric_valid_checked_count = sum(record.get("_derived_numeric_valid") is not None for record in records)
    answerable_count = sum(bool(record.get("is_answerable")) for record in records)
    generation_times = [
        _safe_float(record.get("generation_ms"), math.nan)
        for record in records
        if record.get("generation_ms") is not None and record.get("generation_ms") != ""
    ]
    generation_times = [value for value in generation_times if not math.isnan(value)]

    return {
        "total_questions": total,
        "total": total,
        "valid_json_count": valid_json_count,
        "valid_json_rate": _mean_bool(record.get("_valid_json") for record in records),
        "empty_answer_count": empty_answer_count,
        "empty_answer_rate": empty_answer_count / total,
        "answer_available_count": total - empty_answer_count,
        "answer_available_rate": (total - empty_answer_count) / total,
        "recovered_answer_count": recovered_answer_count,
        "recovered_answer_rate": recovered_answer_count / total,
        "parse_error_type_counts": dict(parse_error_counter),
        "citation_checked_count": citation_checked_count,
        "citation_valid_rate": _mean_bool(record.get("_citation_valid") for record in records),
        "numeric_grounded_checked_count": numeric_grounded_checked_count,
        "numeric_grounded_rate": _mean_bool(record.get("_numeric_grounded") for record in records),
        "source_numeric_grounded_checked_count": source_numeric_grounded_checked_count,
        "source_numeric_grounded_rate": _mean_bool(record.get("_source_numeric_grounded") for record in records),
        "derived_numeric_valid_checked_count": derived_numeric_valid_checked_count,
        "derived_numeric_valid_rate": _mean_bool(record.get("_derived_numeric_valid") for record in records),
        "answerable_count": answerable_count,
        "answerable_rate": _mean_bool(record.get("is_answerable") for record in records),
        "generation_ms_avg": (
            sum(generation_times) / len(generation_times)
            if generation_times
            else math.nan
        ),
        "failure_tag_counts": dict(failure_counter),
    }


def create_failure_tags_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for tag in record.get("_failure_tags", []):
            counter[tag] += 1
            if len(examples[tag]) < 5:
                examples[tag].append(
                    {
                        "question_id": record.get("question_id", ""),
                        "question": record.get("question", ""),
                        "answer_type": record.get("answer_type", ""),
                        "confidence": record.get("confidence", ""),
                    }
                )
    return {
        "failure_tag_counts": dict(counter),
        "examples": dict(examples),
    }


def build_review_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "question_id": record.get("question_id", ""),
                "question": record.get("question", ""),
                "predicted_answer_type": record.get("answer_type", ""),
                "retrieved_docs_top5": record.get("retrieved_docs_top5", ""),
                "source_files": " | ".join(str(value) for value in record.get("source_files", [])),
                "chunk_ids": " | ".join(str(value) for value in record.get("chunk_ids", [])),
                "context_summary": truncate_text(
                    json.dumps(record.get("core_summary", {}), ensure_ascii=False),
                    1200,
                ),
                "answer": record.get("answer", ""),
                "confidence": record.get("confidence", ""),
                "is_answerable": record.get("is_answerable", ""),
                "answer_status": record.get("answer_status", ""),
                "intent_slots": json.dumps(record.get("intent_slots", []), ensure_ascii=False),
                "intent_plan": json.dumps(record.get("intent_plan", []), ensure_ascii=False),
                "target_slots": json.dumps(record.get("target_slots", []), ensure_ascii=False),
                "computed_values": json.dumps(record.get("computed_values", {}), ensure_ascii=False),
                "final_values": json.dumps(record.get("final_values", {}), ensure_ascii=False),
                "citations": json.dumps(record.get("citations", []), ensure_ascii=False),
                "missing_info": json.dumps(record.get("missing_info", []), ensure_ascii=False),
                "warnings": json.dumps(record.get("warnings", []), ensure_ascii=False),
                "failure_tags": json.dumps(record.get("_failure_tags", []), ensure_ascii=False),
                "gt_review_tags": json.dumps(record.get("_gt_review_tags", []), ensure_ascii=False),
                "missing_intents": json.dumps(record.get("_missing_intents", []), ensure_ascii=False),
                "valid_json": record.get("_valid_json", ""),
                "recovered_answer": record.get("_recovered_answer", ""),
                "parse_error_type": record.get("_parse_error_type", ""),
                "generation_ms": record.get("generation_ms", ""),
            }
        )
    return rows


def build_llm_answer_review_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        raw_text = str(record.get("_raw_text") or "")
        parsed_answer = _extract_json_string_field(raw_text, "answer")
        if not parsed_answer:
            parsed_answer = str(record.get("answer") or "")
        rows.append(
            {
                "question_id": record.get("question_id", ""),
                "question": record.get("question", ""),
                "ground_truth": record.get("ground_truth", ""),
                "ground_truth_docs": record.get("ground_truth_docs", ""),
                "retrieved_docs_top5": record.get("retrieved_docs_top5", ""),
                "raw_llm_text": raw_text,
                "parsed_answer": parsed_answer,
                "final_answer": record.get("answer", ""),
                "answer_type": record.get("answer_type", ""),
                "confidence": record.get("confidence", ""),
                "is_answerable": record.get("is_answerable", ""),
                "intent_slots": json.dumps(record.get("intent_slots", []), ensure_ascii=False),
                "intent_plan": json.dumps(record.get("intent_plan", []), ensure_ascii=False),
                "missing_intents": json.dumps(record.get("_missing_intents", []), ensure_ascii=False),
                "valid_json": record.get("_valid_json", ""),
                "recovered_answer": record.get("_recovered_answer", ""),
                "parse_error_type": record.get("_parse_error_type", ""),
                "failure_tags": json.dumps(record.get("_failure_tags", []), ensure_ascii=False),
                "gt_review_tags": json.dumps(record.get("_gt_review_tags", []), ensure_ascii=False),
                "warnings": json.dumps(record.get("warnings", []), ensure_ascii=False),
                "missing_info": json.dumps(record.get("missing_info", []), ensure_ascii=False),
            }
        )
    return rows


def write_llm_answer_review_html(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cards = []
    for idx, row in enumerate(rows, start=1):
        valid_json = bool(row.get("valid_json"))
        recovered = bool(row.get("recovered_answer"))
        status = "valid-json" if valid_json else ("recovered" if recovered else "invalid-json")
        status_label = "valid JSON" if valid_json else ("recovered answer" if recovered else "invalid JSON")
        cards.append(
            f"""
<article class="card {status}">
  <div class="card-header">
    <span class="qid">{html.escape(str(row.get('question_id') or f'row-{idx}'))}</span>
    <span class="badge">{html.escape(status_label)}</span>
    <span class="meta">{html.escape(str(row.get('answer_type', '')))} / {html.escape(str(row.get('confidence', '')))}</span>
  </div>
  <section>
    <h2>질문</h2>
    <pre>{html.escape(str(row.get('question', '')))}</pre>
  </section>
  <section>
    <h2>GT</h2>
    <pre>{html.escape(str(row.get('ground_truth', '')))}</pre>
    <p class="docs">{html.escape(str(row.get('ground_truth_docs', '')))}</p>
  </section>
  <section>
    <h2>Retrieved Docs Top5</h2>
    <pre>{html.escape(str(row.get('retrieved_docs_top5', '')))}</pre>
  </section>
  <section class="grid">
    <div>
      <h2>Raw LLM Text</h2>
      <pre>{html.escape(str(row.get('raw_llm_text', '')))}</pre>
    </div>
    <div>
      <h2>Parsed Answer</h2>
      <pre>{html.escape(str(row.get('parsed_answer', '')))}</pre>
      <h2>Final Answer</h2>
      <pre>{html.escape(str(row.get('final_answer', '')))}</pre>
    </div>
  </section>
  <section>
    <h2>Intent Plan</h2>
    <pre>{html.escape(str(row.get('intent_plan', '')))}</pre>
  </section>
  <section class="diagnostics">
    <span>parse_error_type: {html.escape(str(row.get('parse_error_type', '')))}</span>
    <span>is_answerable: {html.escape(str(row.get('is_answerable', '')))}</span>
    <span>intent_slots: {html.escape(str(row.get('intent_slots', '')))}</span>
    <span>missing_intents: {html.escape(str(row.get('missing_intents', '')))}</span>
    <span>failure_tags: {html.escape(str(row.get('failure_tags', '')))}</span>
    <span>gt_review_tags: {html.escape(str(row.get('gt_review_tags', '')))}</span>
    <span>warnings: {html.escape(str(row.get('warnings', '')))}</span>
  </section>
</article>
"""
        )

    document = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LLM Answer Review</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR", sans-serif; background: #0b1020; color: #e8edf7; font-size: 17px; }}
  header {{ position: sticky; top: 0; z-index: 2; padding: 22px 32px; background: rgba(9, 14, 28, 0.96); border-bottom: 1px solid #29344f; backdrop-filter: blur(10px); }}
  h1 {{ margin: 0; font-size: 30px; letter-spacing: -0.02em; }}
  .summary {{ margin-top: 8px; color: #aab6cf; font-size: 16px; }}
  main {{ padding: 28px; display: grid; gap: 22px; }}
  .card {{ background: #121a2d; border: 1px solid #2a3653; border-left: 8px solid #7f91b8; border-radius: 14px; padding: 24px; box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28); }}
  .card.valid-json {{ border-left-color: #50d890; }}
  .card.recovered {{ border-left-color: #f2b84b; }}
  .card.invalid-json {{ border-left-color: #ff6b6b; }}
  .card-header {{ display: flex; gap: 12px; align-items: center; margin-bottom: 18px; flex-wrap: wrap; }}
  .qid {{ font-weight: 850; font-size: 24px; color: #ffffff; }}
  .badge {{ padding: 5px 11px; border-radius: 999px; background: #22304c; color: #dbe7ff; font-size: 14px; font-weight: 800; }}
  .meta {{ color: #b8c4dc; font-size: 15px; }}
  h2 {{ margin: 18px 0 8px; font-size: 15px; color: #9fc2ff; text-transform: uppercase; letter-spacing: 0.05em; }}
  pre {{ margin: 0; padding: 16px; white-space: pre-wrap; overflow-wrap: anywhere; background: #080d19; border: 1px solid #26334f; border-radius: 10px; line-height: 1.68; font-size: 16px; color: #edf3ff; }}
  .grid {{ display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 18px; }}
  .docs {{ margin: 8px 0 0; color: #aab6cf; font-size: 15px; }}
  .diagnostics {{ display: flex; gap: 9px; flex-wrap: wrap; margin-top: 16px; color: #c0c9dc; font-size: 14px; }}
  .diagnostics span {{ padding: 6px 9px; background: #1b2740; border: 1px solid #2d3b59; border-radius: 7px; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} main {{ padding: 16px; }} header {{ padding: 18px; }} pre {{ font-size: 15px; }} }}
</style>
</head>
<body>
<header>
  <h1>LLM Answer Review</h1>
  <div class="summary">Raw LLM Text → Parsed Answer → Final Answer를 GT와 함께 비교하기 위한 검토용 산출물입니다. 총 {len(rows)}개 문항.</div>
</header>
<main>
{''.join(cards)}
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def build_ragas_eval_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ragas_records = []
    for record in records:
        ragas_records.append(
            {
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "contexts": record.get("contexts", []),
                "ground_truth": record.get("ground_truth", ""),
                "question_id": record.get("question_id", ""),
                "answer_type": record.get("answer_type", ""),
                "source_files": record.get("source_files", []),
                "retrieved_docs_top5": record.get("retrieved_docs_top5", ""),
                "chunk_ids": record.get("chunk_ids", []),
            }
        )
    return ragas_records


def save_generation_outputs(
    output_dir: str | Path,
    records: list[dict[str, Any]],
    *,
    run_config: dict[str, Any],
    ragas_metrics_summary: dict[str, Any] | None = None,
    ragas_per_question: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary = create_generation_summary(records)
    failure_summary = create_failure_tags_summary(records)
    review_rows = build_review_rows(records)
    llm_answer_review_rows = build_llm_answer_review_rows(records)
    ragas_input = build_ragas_eval_records(records)
    ragas_metrics = ragas_metrics_summary or {"status": "not_run"}
    ragas_rows = ragas_per_question or []

    paths = {
        "generated_answers": str(output_dir / "generated_answers.jsonl"),
        "review_samples": str(output_dir / "review_samples.csv"),
        "llm_answer_review": str(output_dir / "llm_answer_review.csv"),
        "llm_answer_review_html": str(output_dir / "llm_answer_review.html"),
        "metrics_summary": str(output_dir / "metrics_summary.json"),
        "failure_tags_summary": str(output_dir / "failure_tags_summary.json"),
        "ragas_eval_input": str(output_dir / "ragas_eval_input.jsonl"),
        "ragas_metrics_summary": str(output_dir / "ragas_metrics_summary.json"),
        "ragas_per_question": str(output_dir / "ragas_per_question.csv"),
        "run_config": str(output_dir / "run_config.json"),
    }

    write_jsonl(paths["generated_answers"], records)
    write_json(paths["metrics_summary"], metrics_summary)
    write_json(paths["failure_tags_summary"], failure_summary)
    write_jsonl(paths["ragas_eval_input"], ragas_input)
    write_json(paths["ragas_metrics_summary"], ragas_metrics)
    write_json(paths["run_config"], run_config)
    _write_dict_rows_csv(paths["review_samples"], review_rows)
    _write_dict_rows_csv(paths["llm_answer_review"], llm_answer_review_rows)
    write_llm_answer_review_html(paths["llm_answer_review_html"], llm_answer_review_rows)
    _write_dict_rows_csv(paths["ragas_per_question"], ragas_rows)
    return paths


def summarize_ragas_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    summary: dict[str, Any] = {}
    for metric_name in metric_names:
        values = [
            _safe_float(row.get(metric_name), math.nan)
            for row in rows
            if row.get(metric_name) is not None and row.get(metric_name) != ""
        ]
        values = [value for value in values if not math.isnan(value)]
        summary[f"{metric_name}_mean"] = (
            sum(values) / len(values)
            if values
            else math.nan
        )
        summary[f"{metric_name}_low_questions"] = [
            row.get("question_id", "")
            for row in rows
            if row.get(metric_name) is not None and row.get(metric_name) != ""
            and not math.isnan(_safe_float(row.get(metric_name), math.nan))
            and _safe_float(row.get(metric_name), math.nan) < 0.5
        ][:20]
    if not any(row.get("ground_truth") for row in rows):
        summary["context_recall_status"] = "not_run_missing_ground_truth"
    return summary


def write_summary_csv(path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in summary.items()
    }
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)


def _write_dict_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_bool(values: Iterable[Any]) -> float:
    vals = [1.0 if bool(value) else 0.0 for value in values]
    return sum(vals) / len(vals) if vals else math.nan


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen = set()
    unique = []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique
