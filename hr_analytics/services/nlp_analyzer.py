"""
NLP Analyzer — Open-ended HR check-in text classification pipeline.

Pipeline stages:
    1. Column detection — locate closed (Q1) + open (Q2) question columns.
    2. Conditional extraction — skip rows that don't meet Q1 trigger rules.
    3. Preprocessing — strip, normalise whitespace, detect non-answers.
    4. Batching — group records for efficient LLM calls (default 25).
    5. LLM call — GitHub Models API (openai/gpt-4.1), strict JSON output.
    6. Validation — Pydantic parse + auto-retry/repair on invalid JSON.
    7. Caching — SHA-256 keyed by (text + question_id + taxonomy_version).

Environment:
    GITHUB_MODELS_TOKEN — required. Fine-grained PAT with models:read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd

from hr_analytics.services.nlp_schema import (
    BUILTIN_PAIRS,
    NLPBatchOutput,
    NLPRecordOutput,
    QuestionPair,
    TAXONOMY_VERSION,
    THEME_LABELS,
)
from hr_analytics.services.github_models_client import GitHubModelsClient
from hr_analytics.services.column_detection import _find_col_by_keywords, _norm

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

# Tokens that count as a "non-answer"
_NON_ANSWER_TOKENS: set[str] = {
    "", "-", "--", "---", "n/a", "na", "none", "nil", "nothing",
    "no comment", "no comments", "لا", "لا شيء", "لا يوجد", ".",
}

_BATCH_SIZE = 25          # records per LLM call
_MAX_RETRIES = 2          # repair retries if JSON is invalid
_MAX_WORKERS = 4          # parallel API calls
_MODEL = "openai/gpt-4.1"


# ============================================================
# Preprocessing helpers
# ============================================================

def _is_non_answer(text: str) -> bool:
    """Return True when *text* is blank, a dash, 'n/a', etc."""
    return _clean(text).lower() in _NON_ANSWER_TOKENS


def _clean(text: Any) -> str:
    """Strip and normalise whitespace; coerce NaN / None to ''."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _detect_language(text: str) -> str:
    """Cheap heuristic: if ≥30 % Arabic-script chars → 'ar', else 'en'."""
    if not text:
        return "en"
    arabic = sum(
        1 for ch in text
        if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F"
    )
    ratio = arabic / max(len(text), 1)
    if ratio >= 0.30:
        return "ar"
    return "en"


def _cache_key(text: str, question_id: str) -> str:
    """SHA-256 of text + question_id + taxonomy version for caching."""
    blob = f"{question_id}|{TAXONOMY_VERSION}|{text}"
    return hashlib.sha256(blob.encode()).hexdigest()


# ============================================================
# Column matching for question pairs
# ============================================================

def _find_pair_columns(
    df: pd.DataFrame,
    pair: QuestionPair,
) -> tuple[str | None, str | None]:
    """
    Return (q1_col, q2_col) by token-matching against *df* column headers.

    Uses the ``q1_tokens`` and ``q2_tokens`` lists on the QuestionPair.
    Tries all tokens first; falls back to first 3 if that fails.
    """
    q1_col: str | None = None
    q2_col: str | None = None

    # --- Q1 (closed question) ---
    if pair.q1_tokens:
        q1_col = _find_col_by_keywords(df, pair.q1_tokens)
        # Relaxed fallback with fewer tokens
        if q1_col is None and len(pair.q1_tokens) > 3:
            q1_col = _find_col_by_keywords(df, pair.q1_tokens[:3])

    # --- Q2 (open-ended question) ---
    if pair.q2_tokens:
        q2_col = _find_col_by_keywords(df, pair.q2_tokens)
        if q2_col is None and len(pair.q2_tokens) > 3:
            q2_col = _find_col_by_keywords(df, pair.q2_tokens[:3])

    return q1_col, q2_col


# ============================================================
# Conditional extraction — Q1 trigger evaluation
# ============================================================

def _q1_passes_trigger(q1_value: str, trigger: str | None) -> bool:
    """
    Evaluate whether *q1_value* satisfies the trigger condition.

    Returns True (include the record) when:
    - trigger is None or empty  → always passes (unconditional Q2)
    - trigger is ``"== Yes"``   → q1_value matches 'yes'
    - trigger is ``"== No"``    → q1_value matches 'no'
    - trigger is ``"in {1,2}"`` → q1_value (after stripping) is in the set
    - trigger is ``"in {Indifferent,Detached}"`` → case-insensitive membership

    When Q1 column was not found (q1_value is empty) AND the open-text
    column has content, the caller should still allow analysis.
    """
    if not trigger:
        return True

    val = q1_value.strip().lower()

    # "== <value>"
    eq_match = re.match(r"^==\s*(.+)$", trigger.strip())
    if eq_match:
        expected = eq_match.group(1).strip().lower()
        return val == expected

    # "in {val1, val2, ...}"
    in_match = re.match(r"^in\s*\{(.+)\}$", trigger.strip())
    if in_match:
        members = {m.strip().lower() for m in in_match.group(1).split(",")}
        return val in members

    return True  # unknown trigger format → allow


# ============================================================
# Prompt construction
# ============================================================

_SYSTEM_PROMPT = (
    "You are an HR text-classification engine. "
    "Return ONLY valid JSON. No prose, no markdown."
)

_REPAIR_PROMPT = (
    "Your previous output was invalid JSON or did not match the schema. "
    "Return ONLY corrected JSON that matches the schema. Do not add any text."
)


def _build_user_prompt(records: list[dict]) -> str:
    """Build the user message for a batch of records."""
    taxonomy_block = "\n".join(f"  - {t}" for t in THEME_LABELS)
    records_block = json.dumps(records, ensure_ascii=False, indent=2)

    return (
        "You will receive a JSON array called \"records\". Each record contains:\n"
        "- employee_id, department, manager_name, year\n"
        "- question_id, pair_id\n"
        "- q1 (optional): {question, answer, options}\n"
        "- q2: {question, answer}\n\n"
        "Task:\n"
        "Analyze ONLY q2.answer using q1 as context when present.\n"
        "Return a JSON object with key \"results\" containing an array.\n"
        "Each element:\n"
        "{\n"
        '  "employee_id": "<from input>",\n'
        '  "pair_id": "<from input>",\n'
        '  "question_id": "<from input>",\n'
        '  "language": "en" or "ar" or "mixed",\n'
        '  "is_non_answer": true/false,\n'
        '  "themes": [{"label": "<taxonomy label>", "confidence": 0.0-1.0}],\n'
        '  "sentiment": {"label": "positive|neutral|negative", "score": -1.0 to 1.0},\n'
        '  "severity": 0-3,\n'
        '  "actionability": 0-2,\n'
        '  "summary": "<max 20 words>",\n'
        '  "recommendation": "<one actionable recommendation>"\n'
        "}\n\n"
        "Rules:\n"
        '- If q2.answer is empty or non-answer ("-", "none", "n/a"): '
        'is_non_answer=true, themes=[{"label":"Other/Unclear","confidence":1.0}], '
        "sentiment neutral, severity 0, actionability 0.\n"
        "- Use ONLY the allowed taxonomy labels.\n"
        "- Do not invent information.\n"
        "- If unsure, use Other/Unclear.\n\n"
        f"FIXED TAXONOMY (use ONLY these labels for themes):\n{taxonomy_block}\n\n"
        "SCORING RULES:\n"
        "- sentiment.label: positive | neutral | negative\n"
        "- sentiment.score: float in [-1, 1]\n"
        "- severity: int 0..3 (0=none, 3=high)\n"
        "- actionability: int 0..2 (0=low, 2=high)\n"
        "- themes: up to 3 labels from the taxonomy above, each with confidence 0..1\n\n"
        f"RECORDS:\n{records_block}\n\n"
        "Return ONLY the JSON object. No markdown fences, no extra text."
    )


# ============================================================
# JSON parsing + validation + repair
# ============================================================

def _extract_json(raw: str) -> dict | list:
    """
    Extract JSON from an LLM response that might include markdown fences
    or leading/trailing prose.
    """
    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = raw.strip("`").strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find first { or [
    for start_ch, end_ch in [("{", "}"), ("[", "]")]:
        idx = raw.find(start_ch)
        if idx == -1:
            continue
        ridx = raw.rfind(end_ch)
        if ridx > idx:
            try:
                return json.loads(raw[idx : ridx + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError("Could not extract valid JSON from LLM response.")


def _validate_batch(raw_json: Any) -> NLPBatchOutput:
    """Parse raw JSON (dict or list) into a validated NLPBatchOutput."""
    if isinstance(raw_json, list):
        raw_json = {"results": raw_json}
    return NLPBatchOutput.model_validate(raw_json)


# ============================================================
# Main Analyzer class
# ============================================================

class NLPAnalyzer:
    """
    End-to-end NLP pipeline for HR open-ended check-in answers.

    The analyzer:
    1. Scans the Employee Check-In DataFrame columns to find Q1/Q2 columns.
    2. Extracts per-row records, honouring conditional triggers on Q1.
    3. Sends open-ended answers in batches to GitHub Models (openai/gpt-4.1).
    4. Validates with Pydantic and retries on invalid JSON.
    5. Caches results keyed by SHA-256(text + question_id + taxonomy_version).
    """

    def __init__(
        self,
        token: str | None = None,
        model: str = _MODEL,
        batch_size: int = _BATCH_SIZE,
        max_workers: int = _MAX_WORKERS,
        cache: dict[str, dict] | None = None,
    ):
        self._token = token or os.environ.get("GITHUB_MODELS_TOKEN", "")
        if not self._token:
            raise EnvironmentError(
                "GITHUB_MODELS_TOKEN is not set. "
                "Create a fine-grained PAT with models:read permission and "
                "set it as the GITHUB_MODELS_TOKEN environment variable."
            )
        self._client = GitHubModelsClient(token=self._token, model=model)
        self._batch_size = batch_size
        self._max_workers = max_workers
        # External cache dict (e.g. from st.session_state) for persistence
        # across multiple runs.  Falls back to an ephemeral in-memory dict.
        self._ext_cache: dict[str, dict] = cache if cache is not None else {}
        self._cache: dict[str, NLPRecordOutput] = {}
        # Rehydrate Pydantic objects from the external (serialised) cache
        for k, v in self._ext_cache.items():
            try:
                self._cache[k] = NLPRecordOutput.model_validate(v)
            except Exception:
                pass  # stale entry → ignore

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def detect_available_pairs(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[QuestionPair, str | None, str | None]]:
        """
        Scan *df* columns and return
        ``[(pair, q1_col, q2_col), ...]``
        for every built-in pair whose open-ended column (Q2) is found.
        """
        found: list[tuple[QuestionPair, str | None, str | None]] = []
        for pair in BUILTIN_PAIRS:
            q1_col, q2_col = _find_pair_columns(df, pair)
            if q2_col is not None:
                found.append((pair, q1_col, q2_col))
        return found

    def analyze(
        self,
        df: pd.DataFrame,
        pair_ids: list[str] | None = None,
        question_ids: list[str] | None = None,
        name_col: str | None = None,
        dept_col: str | None = None,
        mgr_col: str | None = None,
        year_col: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> pd.DataFrame:
        """
        Run the full NLP pipeline on *df* for the requested question pairs.

        Parameters
        ----------
        df : DataFrame
            Cleaned employee check-in data (columns = questions, rows = employees).
        pair_ids / question_ids : list[str] | None
            Filter which pairs to analyse.  ``None`` → all detected.
        name_col, dept_col, mgr_col, year_col : str | None
            Override auto-detection of metadata columns.
        progress_callback : callable(current, total) | None
            Optional progress reporter (e.g. Streamlit progress bar).

        Returns
        -------
        DataFrame with one row per employee × question, including all NLP
        output fields plus original Q1/Q2 text for audit.
        """
        pairs = self._resolve_pairs(pair_ids, question_ids)

        # Auto-detect metadata columns
        name_col = name_col or self._auto_detect_col(
            df, ["Mena Name", "Your Name", "Employee Name", "Name"]
        )
        dept_col = dept_col or self._auto_detect_dept_col(df)
        mgr_col = mgr_col or self._auto_detect_col(
            df, ["Your Manager's Name", "Manager Name", "Name on Mena"]
        )
        year_col = year_col or self._auto_detect_col(df, ["Year", "year"])

        # ---- Build records for every pair ----
        total_records = 0
        pair_record_map: dict[str, list[dict]] = {}

        for pair in pairs:
            q1_col, q2_col = _find_pair_columns(df, pair)
            if q2_col is None:
                logger.warning(
                    "Could not find Q2 column for question_id '%s' (%s) – skipping.",
                    pair.question_id, pair.q2_label,
                )
                continue
            recs = self._extract_records(
                df, pair, q1_col, q2_col, name_col, dept_col, mgr_col, year_col
            )
            pair_record_map[pair.question_id] = recs
            total_records += len(recs)

        all_results: list[dict] = []
        processed = 0
        _progress_lock = threading.Lock()

        # ---- Resolve cached & non-answers first (no API) ----
        all_to_send: list[dict] = []  # flat list of records needing LLM

        for pair in pairs:
            records = pair_record_map.get(pair.question_id, [])
            if not records:
                continue

            for rec in records:
                key = _cache_key(rec["_q2_raw"], rec["question_id"])

                if key in self._cache:
                    row = self._merge_meta(self._cache[key].model_dump(), rec)
                    all_results.append(row)
                    processed += 1
                elif rec["_is_non_answer"]:
                    na_out = self._non_answer_output(rec)
                    self._cache[key] = na_out
                    self._ext_cache[key] = na_out.model_dump()
                    row = self._merge_meta(na_out.model_dump(), rec)
                    all_results.append(row)
                    processed += 1
                else:
                    all_to_send.append(rec)

        if progress_callback:
            progress_callback(processed, total_records)

        # ---- Build batches across all pairs ----
        batches: list[list[dict]] = [
            all_to_send[i : i + self._batch_size]
            for i in range(0, len(all_to_send), self._batch_size)
        ]

        def _handle_batch(batch: list[dict]) -> list[dict]:
            """Process one batch and return merged result rows."""
            batch_results = self._process_batch(batch)
            rows: list[dict] = []
            for r in batch_results:
                matching = [
                    rec for rec in batch
                    if rec["employee_id"] == r.employee_id
                    and rec["question_id"] == r.question_id
                ]
                meta_rec = matching[0] if matching else batch[0]
                cache_text = meta_rec["_q2_raw"]
                key = _cache_key(cache_text, r.question_id)
                self._cache[key] = r
                self._ext_cache[key] = r.model_dump()
                rows.append(self._merge_meta(r.model_dump(), meta_rec))
            return rows

        # ---- Parallel LLM calls ----
        workers = min(self._max_workers, len(batches)) or 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_handle_batch, b): b for b in batches
            }
            for future in as_completed(futures):
                try:
                    rows = future.result()
                except Exception as exc:
                    logger.error("Batch failed: %s", exc)
                    rows = []
                with _progress_lock:
                    all_results.extend(rows)
                    processed += len(rows)
                    if progress_callback:
                        progress_callback(processed, total_records)

        if not all_results:
            return pd.DataFrame()

        result_df = pd.DataFrame(all_results)

        # Strip internal __idx suffix from employee_id for clean display
        if "employee_id" in result_df.columns:
            result_df["employee_id"] = result_df["employee_id"].str.replace(
                r"__idx\d+$", "", regex=True
            )

        # Flatten themes list → readable string
        if "themes" in result_df.columns:
            result_df["themes_display"] = result_df["themes"].apply(
                lambda ts: "; ".join(
                    f"{t['label']} ({t['confidence']:.0%})" for t in ts
                ) if ts else ""
            )

        # Flatten sentiment dict → separate columns
        if "sentiment" in result_df.columns:
            result_df["sentiment_label"] = result_df["sentiment"].apply(
                lambda s: s.get("label", "") if isinstance(s, dict) else ""
            )
            result_df["sentiment_score"] = result_df["sentiment"].apply(
                lambda s: s.get("score", 0) if isinstance(s, dict) else 0
            )

        return result_df

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    @staticmethod
    def _auto_detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Return first column from *candidates* that exists in *df*."""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _auto_detect_dept_col(df: pd.DataFrame) -> str | None:
        """Find the department column (often 'Company Name / Department')."""
        for c in df.columns:
            cn = _norm(c)
            if "company" in cn and "department" in cn:
                return c
        return None

    def _resolve_pairs(
        self,
        pair_ids: list[str] | None,
        question_ids: list[str] | None,
    ) -> list[QuestionPair]:
        """Return the subset of BUILTIN_PAIRS matching filters."""
        if question_ids is not None:
            qid_set = set(question_ids)
            return [p for p in BUILTIN_PAIRS if p.question_id in qid_set]
        if pair_ids is not None:
            pid_set = set(pair_ids)
            return [p for p in BUILTIN_PAIRS if p.pair_id in pid_set]
        return list(BUILTIN_PAIRS)

    def _extract_records(
        self,
        df: pd.DataFrame,
        pair: QuestionPair,
        q1_col: str | None,
        q2_col: str | None,
        name_col: str | None,
        dept_col: str | None,
        mgr_col: str | None,
        year_col: str | None,
    ) -> list[dict]:
        """
        Build per-row record dicts from the DataFrame for one QuestionPair.

        Applies conditional extraction: if the pair has a ``q1_trigger``,
        only rows where Q1 meets the trigger yield a record.
        If Q1 column is not found but Q2 has content, the record is still
        created (allows analysis of orphan open text).
        """
        if q2_col is None:
            return []

        records: list[dict] = []

        for idx, row in df.iterrows():
            q2_text = _clean(row.get(q2_col, ""))
            if not q2_text:
                continue  # completely empty → skip

            q1_val = _clean(row.get(q1_col, "")) if q1_col else ""

            # --- Conditional extraction ---
            if pair.q1_trigger:
                if q1_col and q1_val:
                    # Q1 column found and has a value → enforce trigger
                    if not _q1_passes_trigger(q1_val, pair.q1_trigger):
                        continue
                # If q1_col is None or q1_val is empty but q2 has text,
                # we still allow the record (orphan open text).

            emp_name = _clean(row.get(name_col, "")) if name_col else ""
            emp_id = emp_name or str(idx)
            dept = _clean(row.get(dept_col, "")) if dept_col else ""
            mgr_name = _clean(row.get(mgr_col, "")) if mgr_col else ""
            year_val = _clean(row.get(year_col, "")) if year_col else ""

            # Unique record index to avoid collisions when names repeat
            _record_idx = f"{emp_id}__idx{idx}"

            records.append({
                # Identifiers sent to LLM
                "employee_id": _record_idx,
                "department": dept,
                "manager_name": mgr_name,
                "year": year_val,
                "question_id": pair.question_id,
                "pair_id": pair.pair_id,
                # Q1 context for the LLM
                "q1": {
                    "question": pair.q1_label or "",
                    "answer": q1_val,
                } if (pair.q1_label and q1_col) else None,
                # Q2 answer
                "q2": {
                    "question": pair.q2_label,
                    "answer": q2_text,
                },
                # --- Internal fields (not sent to LLM) ---
                "_is_non_answer": _is_non_answer(q2_text),
                "_lang": _detect_language(q2_text),
                "_q2_raw": q2_text,
                "_q1_raw": q1_val,
                "_q1_question": pair.q1_label or "",
                "_q2_question": pair.q2_label,
                "_employee_name": emp_name,
            })

        return records

    @staticmethod
    def _merge_meta(model_output: dict, input_rec: dict) -> dict:
        """
        Merge LLM output dict with audit/metadata fields from input record.
        """
        model_output["employee_name"] = input_rec.get("_employee_name", "")
        model_output["department"] = input_rec.get("department", "")
        model_output["manager_name"] = input_rec.get("manager_name", "")
        model_output["year"] = input_rec.get("year", "")
        model_output["q1_question"] = input_rec.get("_q1_question", "")
        model_output["q1_answer"] = input_rec.get("_q1_raw", "")
        model_output["q2_question"] = input_rec.get("_q2_question", "")
        model_output["q2_answer_raw"] = input_rec.get("_q2_raw", "")
        return model_output

    @staticmethod
    def _non_answer_output(rec: dict) -> NLPRecordOutput:
        """Create a deterministic output for non-answers (no LLM needed)."""
        return NLPRecordOutput(
            employee_id=rec["employee_id"],
            pair_id=rec["pair_id"],
            question_id=rec["question_id"],
            language=rec.get("_lang", "en"),
            is_non_answer=True,
            themes=[{"label": "Other/Unclear", "confidence": 1.0}],
            sentiment={"label": "neutral", "score": 0.0},
            severity=0,
            actionability=0,
            summary="Non-answer",
            recommendation="N/A",
        )

    def _process_batch(self, batch: list[dict]) -> list[NLPRecordOutput]:
        """Send one batch to the LLM, validate, and optionally repair."""
        # Build payload: strip internal keys, keep only what the model needs
        payload = []
        for rec in batch:
            row: dict[str, Any] = {
                "employee_id": rec["employee_id"],
                "department": rec["department"],
                "manager_name": rec["manager_name"],
                "year": rec["year"],
                "question_id": rec["question_id"],
                "pair_id": rec["pair_id"],
                "q2": rec["q2"],
            }
            if rec.get("q1"):
                row["q1"] = rec["q1"]
            payload.append(row)

        prompt = _build_user_prompt(payload)
        raw = ""
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                if attempt == 0:
                    raw = self._client.complete(_SYSTEM_PROMPT, prompt)
                else:
                    repair_msg = (
                        f"{_REPAIR_PROMPT}\n\n"
                        f"Original prompt:\n{prompt}\n\n"
                        f"Your previous (invalid) output:\n{raw}"
                    )
                    raw = self._client.complete(_SYSTEM_PROMPT, repair_msg)

                parsed = _extract_json(raw)
                validated = _validate_batch(parsed)
                return validated.results

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "NLP batch attempt %d/%d failed: %s",
                    attempt + 1, _MAX_RETRIES + 1, exc,
                )
                time.sleep(1)

        # All retries exhausted → return fallback records
        logger.error(
            "NLP batch failed after %d attempts. Last error: %s",
            _MAX_RETRIES + 1, last_error,
        )
        return [
            NLPRecordOutput(
                employee_id=rec.get("employee_id", "?"),
                pair_id=rec.get("pair_id", "?"),
                question_id=rec.get("question_id", "?"),
                language=rec.get("_lang", "en"),
                is_non_answer=False,
                themes=[{"label": "Other/Unclear", "confidence": 0.0}],
                sentiment={"label": "neutral", "score": 0.0},
                severity=0,
                actionability=0,
                summary=f"Analysis failed: {last_error}",
                recommendation="Manual review required.",
            )
            for rec in batch
        ]


# ============================================================
# Self-test
# ============================================================

def _self_test() -> None:  # pragma: no cover
    """Quick smoke test on synthetic data. Requires GITHUB_MODELS_TOKEN."""
    token = os.environ.get("GITHUB_MODELS_TOKEN", "")
    if not token:
        print("Set GITHUB_MODELS_TOKEN to run the self-test.")
        return

    df = pd.DataFrame({
        "Mena Name": ["Alice Smith", "Bob Jones", "Carol Lee"],
        "Company Name / Department": ["Engineering", "Marketing", "HR"],
        "Your Manager's Name": ["Manager A", "Manager B", "Manager C"],
        "Year": ["2025", "2025", "2025"],
        # Q1 for adaptation (Likert 1-5)
        "During the year 2025, I was able to adapt to changes in my job requirements.": [
            2, 4, 1,
        ],
        # Q2 for adaptation
        "If answered 1 or 2, please further elaborate on the reasons you were not able to adjust to changing job requirements.": [
            "Too many process changes with no training provided.",
            "-",
            "New systems were rolled out without documentation.",
        ],
        # Standalone Q2: stress reasons
        "In your opinion, what are the reasons behind this stress?": [
            "Heavy workload and tight deadlines.",
            "n/a",
            "Poor communication from leadership.",
        ],
    })

    analyzer = NLPAnalyzer(token=token)
    results = analyzer.analyze(
        df,
        question_ids=["ADAPTATION_STATEMENT"],
    )
    print(f"Processed {len(results)} records.")
    if not results.empty:
        print(
            results[
                ["employee_id", "question_id", "themes_display",
                 "sentiment_label", "severity", "q2_answer_raw"]
            ].to_string()
        )


if __name__ == "__main__":
    _self_test()
