"""
Column-detection helpers for HR check-in data.

Every ``_find_*_col`` function scans a DataFrame's column headers using
keyword matching and returns the first matching column name (or ``None``).
The core normalizer ``_norm()`` and the generic ``_find_col_by_keywords()``
live here so that all higher-level finders can reuse them.
"""

from __future__ import annotations

import re
from typing import Callable

import pandas as pd


# ============================================================
# Core text normalizer (used by almost everything)
# ============================================================

def _norm(text: str) -> str:
    """Lower-case, collapse whitespace, strip non-alphanumeric chars."""
    s = str(text).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ============================================================
# Generic finder factory & keyword finder
# ============================================================

def _make_col_finder(*keyword_groups: list[str]) -> Callable[[pd.DataFrame], str | None]:
    """
    Return a function that scans ``df.columns`` for the first column
    whose normalised header contains *all* keywords in at least one of
    the supplied keyword groups.
    """
    def finder(df: pd.DataFrame) -> str | None:
        for kws in keyword_groups:
            for c in df.columns:
                h = _norm(c)
                if all(k in h for k in kws):
                    return c
        return None
    return finder


def _find_col_by_keywords(df: pd.DataFrame, keywords_all: list[str]) -> str | None:
    """Return first column whose header contains ALL keywords."""
    if df is None or df.empty:
        return None
    keys = [_norm(k) for k in keywords_all]
    for c in df.columns:
        cn = _norm(c)
        if all(k in cn for k in keys):
            return c
    return None


# ============================================================
# Pre-built finders (via factory)
# ============================================================

_find_aligned_goals_col = _make_col_finder(
    ["aligned", "goals", "department"],
    ["aligned", "goals"],
)

_find_if_no_elaborate_col = _make_col_finder(
    ["if", "no", "elaborate"],
    ["if", "answered", "no"],
)


# ============================================================
# Yes-rate / scale-rate helpers (keyword-based KPIs)
# ============================================================

def _yes_rate_from_keywords(
    df: pd.DataFrame,
    keywords: list[str],
) -> tuple[float, str | None]:
    """Return (yes_rate_0_to_1, column_name_or_None)."""
    col = _find_col_by_keywords(df, keywords)
    if not col:
        return 0.0, None
    s = df[col].astype(str).str.strip().str.lower()
    valid = s[s.str.len() > 0]
    if valid.empty:
        return 0.0, col
    yes = valid.str.startswith(("yes", "y", "true", "1"))
    return yes.mean(), col


def _scale_good_rate_from_keywords(
    df: pd.DataFrame,
    keywords: list[str],
    good_min: int = 4,
) -> tuple[float, str | None]:
    """Return (rate_of_values >= good_min, column_name_or_None)."""
    col = _find_col_by_keywords(df, keywords)
    if not col:
        return 0.0, None
    s = pd.to_numeric(df[col], errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return 0.0, col
    return (valid >= good_min).mean(), col


# ============================================================
# Specific column finders
# ============================================================

# -- Manager behaviour question (employee check-in) --
def _find_mgr_behavior_question_col_emp(df: pd.DataFrame) -> str | None:
    # Employee question: "Please check the true statement(s) below about your manager"
    return (
        _find_col_by_keywords(df, ["true", "statement", "manager"])
        or _find_col_by_keywords(df, ["check", "statement", "manager"])
        or _find_col_by_keywords(df, ["below", "about", "your", "manager"])
    )


# -- Manager name inside the Employee check-in --
def _find_manager_name_col_emp(df: pd.DataFrame) -> str | None:
    """
    Detect the column in EMPLOYEE check-in that contains the manager name
    (needed for Slide 12 grouping by manager).
    """
    if df is None or df.empty:
        return None

    # Prefer exact common labels if your cleaner standardizes them
    for cand in [
        "Manager Name",
        "Direct Manager Name",
        "Direct Manager",
        "My Manager",
        "Manager",
        "Reporting Manager",
        "Line Manager",
    ]:
        if cand in df.columns:
            return cand

    # Keyword-based fallback
    for c in df.columns:
        h = _norm(c)
        if ("manager" in h) and ("name" in h or "direct" in h or "report" in h or "line" in h):
            return c

    # Last resort: any column containing 'manager'
    for c in df.columns:
        if "manager" in _norm(c):
            return c

    return None


# -- Promotion readiness --
def _find_ready_for_promotion_col(df: pd.DataFrame) -> str | None:
    # Adjust keywords if your header is slightly different
    return _find_col_by_keywords(df, ["ready", "promotion"])


# -- Employee / Your-Name / Subordinate columns --
def _find_employee_name_col(df: pd.DataFrame) -> str | None:
    # common possibilities in cleaned files
    for cand in ["Employee Name", "employee_name", "Name", "Full Name", "Employee"]:
        if cand in df.columns:
            return cand
    # fallback: try keyword match
    for c in df.columns:
        h = _norm(c)
        if ("employee" in h and "name" in h) or (h == "name") or ("full name" in h):
            return c
    return None


def _find_your_name_col(df: pd.DataFrame) -> str | None:
    """Detect the respondent-name column in an *employee* check-in."""
    prefs = ["Mena Name", "Your Name", "Employee Name"]
    for p in prefs:
        if p in df.columns:
            return p
    for c in df.columns:
        h = _norm(c)
        if ("your" in h and "name" in h) or ("employee" in h and "name" in h):
            return c
    return None


def _find_subordinate_name_col(df: pd.DataFrame) -> str | None:
    # Prefer exact cleaned canonical field
    if df is None or df.empty:
        return None
    if "Subordinate Name" in df.columns:
        return "Subordinate Name"
    # Fallback: keyword-based detection
    col = _find_col_by_keywords(df, ["subordinate", "name"])  # token match
    if col:
        return col
    # Last resort: any column whose normalized header contains both tokens
    for c in df.columns:
        h = _norm(c)
        if ("subordinate" in h) and ("name" in h):
            return c
    return None


# -- Promotion follow-up columns --
def _find_promo_position_col(df: pd.DataFrame) -> str | None:
    # matches: "If yes, please state the position"
    return _find_col_by_keywords(df, ["state", "position"]) or _find_col_by_keywords(df, ["position"])


def _find_promo_review_date_col(df: pd.DataFrame) -> str | None:
    # matches "suggest a date" / "review" / "consider promotion"
    return (
        _find_col_by_keywords(df, ["suggest", "date"]) or
        _find_col_by_keywords(df, ["review", "date"]) or
        _find_col_by_keywords(df, ["consider", "promotion"]) or
        _find_col_by_keywords(df, ["promotion", "date"])
    )


# -- Risk / PIP --
def _find_risk_reason_col(df: pd.DataFrame) -> str | None:
    # Prefer exact header provided by user
    exact = "If yes, please elaborate on the case"
    if exact in df.columns:
        return exact
    # Fallback: explicit reason columns near risk context
    col = _find_col_by_keywords(df, ["reason", "risk"]) or _find_col_by_keywords(df, ["reason"]) or _find_col_by_keywords(df, ["explain"]) or _find_col_by_keywords(df, ["incident"]) or _find_col_by_keywords(df, ["elaborate"])
    # Ensure we don't pick yes/no fields
    if col:
        from hr_analytics.services.normalizers import _looks_like_yesno
        if _looks_like_yesno(df[col]):
            col = None
    return col


def _find_pip_reason_no_col(df: pd.DataFrame) -> str | None:
    # Prefer exact header provided by user
    exact = "If No, please elaborate"
    if exact in df.columns:
        return exact
    # Reason if no PIP
    for keys in (["reason", "if", "no"], ["if no", "reason"], ["why", "no", "pip"]):
        c = _find_col_by_keywords(df, list(keys))
        if c:
            return c
    # fallback to any elaboration column that mentions reason
    for c in df.columns:
        h = _norm(c)
        if "reason" in h and ("if no" in h or "if your answer is no" in h or "if not" in h):
            return c
    return None


# -- Check-in frequency --
def _find_checkin_freq_col_emp(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["had", "check", "in", "meeting", "with", "my", "manager"]) or \
           _find_col_by_keywords(df, ["check", "in", "meeting", "my", "manager"])


def _find_checkin_freq_col_mgr(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["had", "check", "in", "meeting", "with", "this", "person"]) or \
           _find_col_by_keywords(df, ["check", "in", "meeting", "this", "person"])


# -- Resources --
def _find_company_resources_col_emp(df: pd.DataFrame) -> str | None:
    # Your exact header:
    # "What company resources or practices do I use to ease and facilitate my experience within the work culture?"
    for c in df.columns:
        h = _norm(c)
        if ("company resources" in h or "resources" in h) and ("work culture" in h) and ("ease" in h or "facilitate" in h):
            return c
    # fallback
    for c in df.columns:
        h = _norm(c)
        if ("resources" in h) and ("work culture" in h):
            return c
    return None


def _find_resources_other_text_col_emp(df: pd.DataFrame) -> str | None:
    # tries to find the follow-up text for "Other"
    for c in df.columns:
        h = _norm(c)
        if ("other" in h) and ("specify" in h or "please specify" in h):
            return c
    return None


# -- Pulse check --
def _find_pulse_yn_col_emp(df: pd.DataFrame) -> str | None:
    # The big header text you mentioned contains "pulse check meeting"
    for c in df.columns:
        h = _norm(c)
        if "pulse check" in h and "meeting" in h:
            return c
    return None


def _find_pulse_reason_col_emp(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if _norm(c).startswith("if answered yes please specify the reason"):
            return c
    return None


# -- Stress --
def _find_stress_freq_col_emp(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        h = _norm(c)
        if ("rate" in h) and ("stress" in h) and ("frequency" in h):
            return c
    for c in df.columns:
        h = _norm(c)
        if ("stress levels" in h) and ("frequency" in h):
            return c
    return None


def _find_stress_reason_col_emp(df: pd.DataFrame) -> str | None:
    # best-effort: looks for follow-up reason wording
    for c in df.columns:
        h = _norm(c)
        if ("elaborate" in h or "reason" in h or "why" in h) and ("stress" in h):
            return c
    # fallback: any column mentioning stress + open ended
    for c in df.columns:
        h = _norm(c)
        if "stress" in h and ("elabor" in h or "reason" in h):
            return c
    return None


def _find_stress_reasons_col(df: pd.DataFrame) -> str | None:
    """Multi-select stress reasons (employee or manager)."""
    return (
        _find_col_by_keywords(df, ["most", "common", "reason", "stress"])
        or _find_col_by_keywords(df, ["reason", "stress"])
        or _find_col_by_keywords(df, ["causes", "stress"])
    )


def _find_mgr_stress_freq_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        h = _norm(c)
        if ("frequently" in h or "frequency" in h) and ("reflect stress" in h or ("stress" in h and "employee" in h)):
            return c
    # fallback: stress + frequent
    for c in df.columns:
        h = _norm(c)
        if ("stress" in h) and ("frequent" in h):
            return c
    return None


# -- Manager dynamics --
def _find_mgr_dynamics_col_mgr(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        h = _norm(c)
        if ("contributes" in h) and ("department dynamics" in h) and ("enhance" in h or "enhancement" in h):
            return c
    for c in df.columns:
        h = _norm(c)
        if ("dynamics" in h) and ("contribut" in h):
            return c
    return None


# -- Employee dynamics --
def _find_dynamics_col_emp(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        h = _norm(c)
        if ("department dynamics" in h) and ("enhancement" in h or "enhance" in h) and ("3 areas" in h or "in what 3" in h):
            return c
    for c in df.columns:
        h = _norm(c)
        if ("dynamics" in h) and ("enhance" in h or "enhancement" in h):
            return c
    return None


# -- Team integration --
def _find_team_integration_col_emp(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["team", "integration", "i", "am"]) or \
           _find_col_by_keywords(df, ["team", "integration"])


def _find_team_integration_col_mgr(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["team", "integration", "this", "person"]) or \
           _find_col_by_keywords(df, ["team", "integration"])


# -- Input seeking --
def _find_input_seek_col_emp(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["actively", "seek", "consider", "input", "department"]) or \
           _find_col_by_keywords(df, ["seek", "consider", "input"])


def _find_input_seek_col_mgr(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["actively", "seek", "consider", "team", "members", "input"]) or \
           _find_col_by_keywords(df, ["seek", "consider", "team", "input"]) or \
           _find_col_by_keywords(df, ["seek", "consider", "input"])


# -- Job change --
def _find_job_change_yn_col_emp(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["encountered", "changes", "job", "requirements"])


def _find_job_change_yn_col_mgr(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["this", "person", "encountered", "changes", "job", "requirements"]) or \
           _find_col_by_keywords(df, ["encountered", "changes", "job", "requirements"])


def _find_job_change_types_col(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["if", "yes", "what", "changes"]) or \
           _find_col_by_keywords(df, ["what", "changes", "encounter"])
