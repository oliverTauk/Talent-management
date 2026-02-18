"""
Normalizers, value mappers, and counting helpers for HR check-in data.

This module contains:
- ``_norm_*`` functions that canonicalize free-text survey answers
- ``*_ORDER`` / ``*_CATS`` constants defining canonical display orders
- Boolean mask helpers (``_yes_mask``, ``_no_mask``, ``_count_yes_no``)
- Data-type detection helpers (``_looks_like_yesno``, ``_looks_like_scale_1_5``)
- Multi-select splitting / counting helpers
- Date-parsing helpers
- Department helpers (finding, filtering, options)
- Check-in frequency normalizers
- KPI metrics aggregator (``_metrics``)
"""

from __future__ import annotations

import re

import pandas as pd

from hr_analytics.services.column_detection import _norm, _find_col_by_keywords
from hr_analytics.services.kpis import KPIService


# ============================================================
# Boolean mask helpers
# ============================================================

def _yes_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.str.startswith(("yes", "y", "true", "1"))


def _no_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.str.startswith(("no", "n", "false", "0"))


def _count_yes_no(df: pd.DataFrame, col: str) -> tuple[int, int, float, float]:
    """Returns (yes_count, no_count, yes_pct, no_pct) for a yes/no column."""
    if not col or col not in df.columns:
        return 0, 0, 0.0, 0.0
    ym = _yes_mask(df[col])
    nm = _no_mask(df[col])
    yes_count = int(ym.sum())
    no_count = int(nm.sum())
    total = yes_count + no_count
    yes_pct = (yes_count / total * 100) if total > 0 else 0.0
    no_pct = (no_count / total * 100) if total > 0 else 0.0
    return yes_count, no_count, yes_pct, no_pct


# ============================================================
# Data-type detection helpers
# ============================================================

IDENTITY_HINTS = [
    "timestamp", "email", "name", "department", "company", "manager", "subordinate",
]


def _is_identity_col(col_name: str) -> bool:
    h = _norm(col_name)
    return any(tok in h for tok in IDENTITY_HINTS)


def _looks_like_yesno(series: pd.Series) -> bool:
    if series is None or len(series) == 0:
        return False
    s = series.astype(str).str.strip().str.lower()
    yes = s.str.startswith(("yes", "y", "true", "1"))
    no = s.str.startswith(("no", "n", "false", "0"))
    valid = yes | no
    non_empty = s.ne("")
    denom = max(int(non_empty.sum()), 1)
    return int(valid.sum()) / denom >= 0.6


def _looks_like_scale_1_5(series: pd.Series) -> bool:
    if series is None or len(series) == 0:
        return False
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if len(non_null) == 0:
        return False
    in_range = non_null.between(1, 5)
    return int(in_range.sum()) / len(non_null) >= 0.6


# ============================================================
# Multi-select splitting helpers
# ============================================================

def _split_multiselect(v: str) -> list[str]:
    if pd.isna(v):
        return []
    s = str(v).strip()
    if not s:
        return []
    parts = re.split(r"[,\|;\n]+", s)
    return [p.strip() for p in parts if p.strip()]


def _multiselect_counts(series: pd.Series) -> pd.DataFrame:
    items = []
    for v in series.dropna().astype(str):
        items.extend(_split_multiselect(v))
    if not items:
        return pd.DataFrame(columns=["Option", "Count"])
    vc = pd.Series(items).value_counts()
    return vc.rename_axis("Option").reset_index(name="Count")


# ============================================================
# Manager behavior tags
# ============================================================

POS_TAGS = {
    "POS: clear guidance",
    "POS: continuous feedback",
    "POS: recognition",
    "POS: growth",
}

NEG_TAGS = {
    "NEG: insufficient info",
    "NEG: insufficient feedback",
    "NEG: rarely gives credit",
    "NEG: rarely gives chances",
}


def _norm_behavior_opt(opt: str) -> str:
    s = _norm(opt)
    # POSITIVE
    if "clear" in s and "guid" in s:
        return "POS: clear guidance"
    if "continuous" in s and "feedback" in s:
        return "POS: continuous feedback"
    if "recognition" in s or ("recogn" in s and "work" in s):
        return "POS: recognition"
    if ("opportunit" in s and ("develop" in s or "grow" in s)) or ("develop" in s and "grow" in s):
        return "POS: growth"
    # NEGATIVE
    if "insufficient" in s and ("information" in s or "info" in s):
        return "NEG: insufficient info"
    if "insufficient" in s and "feedback" in s:
        return "NEG: insufficient feedback"
    if ("rarely" in s and "credit" in s) or ("rarely" in s and "good performance" in s):
        return "NEG: rarely gives credit"
    if ("rarely" in s and ("chance" in s or "chances" in s) and ("advance" in s or "improve" in s)):
        return "NEG: rarely gives chances"
    return "OTHER"


# ============================================================
# Stress helpers
# ============================================================

STRESS_FREQ_ORDER = ["Extremely frequent", "Frequent", "Less frequent"]


def _norm_stress_freq(x: str) -> str:
    s = _norm(x)
    if "less" in s and "frequent" in s:
        return "Less frequent"
    if s.startswith("frequent") or ("frequent" in s and "extreme" not in s):
        return "Frequent"
    if "extreme" in s and "frequent" in s:
        return "Extremely frequent"
    return "Unknown"


STRESS_REASON_ORDER = [
    "Personal",
    "Unrealistic Goals",
    "Tight Deadlines",
    "Workload",
    "Relationship with manager",
    "Relationship with team members",
    "Others",
]


def _norm_stress_reason(x: str) -> str | None:
    s = _norm(x)

    if "personal" in s:
        return "Personal"
    if "unrealistic" in s and "goal" in s:
        return "Unrealistic Goals"
    if "tight" in s and "deadline" in s:
        return "Tight Deadlines"
    if "workload" in s:
        return "Workload"
    if "relationship" in s and "manager" in s:
        return "Relationship with manager"
    if "relationship" in s and ("team members" in s or "team" in s):
        return "Relationship with team members"
    if "other" in s:
        return "Others"

    return None


def _stress_reason_counts(series: pd.Series) -> pd.Series:
    items: list[str] = []
    for v in series.dropna().astype(str):
        for part in re.split(r"[,\|;\n]+", v):
            p = part.strip()
            if not p:
                continue
            k = _norm_stress_reason(p)
            if k:
                items.append(k)
    vc = pd.Series(items).value_counts()
    return vc.reindex(STRESS_REASON_ORDER, fill_value=0)


# ============================================================
# Manager dynamics
# ============================================================

MGR_DYNAMICS_ORDER = [
    "Having Clear and Transparent Communication",
    "Having a positive influence on the team",
    "Being trustful",
    "Abiding by defined roles and responsibilities",
    "Adopting a healthy conflict resolution approach",
    "Collaborating and supporting the team",
    "Not applicable",
]


def _norm_mgr_dynamics_opt(x: str) -> str | None:
    s = _norm(x)

    if "clear" in s and "transparent" in s and "communication" in s:
        return "Having Clear and Transparent Communication"
    if "collabor" in s and ("support" in s or "team" in s):
        return "Collaborating and supporting the team"
    if "healthy" in s and "conflict" in s:
        return "Adopting a healthy conflict resolution approach"
    if ("abiding" in s or "defined" in s) and ("roles" in s or "responsibil" in s):
        return "Abiding by defined roles and responsibilities"
    if "trust" in s:
        return "Being trustful"
    if "positive" in s and "influence" in s:
        return "Having a positive influence on the team"
    if "not applicable" in s or s == "na":
        return "Not applicable"

    return None


def _mgr_dynamics_counts(series: pd.Series) -> pd.Series:
    items: list[str] = []
    for v in series.dropna().astype(str):
        for part in re.split(r"[,\|;\n]+", v):
            p = part.strip()
            if not p:
                continue
            k = _norm_mgr_dynamics_opt(p)
            if k:
                items.append(k)
    vc = pd.Series(items).value_counts()
    return vc.reindex(MGR_DYNAMICS_ORDER, fill_value=0)


# ============================================================
# Employee dynamics (Slide 21)
# ============================================================

DYNAMICS_ORDER = [
    "Clear & Transparent Communication",
    "Team Collaboration",
    "Healthy Conflict Resolution",
    "Defined Responsibilities",
    "Trust",
    "Negative Influence",
    "Not Applicable",
]


def _norm_dynamics_opt(x: str) -> str | None:
    s = _norm(x)
    if "clear" in s and "transparent" in s and "communication" in s:
        return "Clear & Transparent Communication"
    if "team" in s and ("collaboration" in s or "support" in s):
        return "Team Collaboration"
    if "healthy" in s and "conflict" in s:
        return "Healthy Conflict Resolution"
    if ("defined" in s or "organized" in s) and ("roles" in s or "responsibil" in s):
        return "Defined Responsibilities"
    if "trust" in s:
        return "Trust"
    if "negative" in s and "influence" in s:
        return "Negative Influence"
    if "not applicable" in s or s == "na":
        return "Not Applicable"
    return None


def _dynamics_counts(series: pd.Series) -> pd.Series:
    items: list[str] = []
    for v in series.dropna().astype(str):
        for part in re.split(r"[,\|;\n]+", v):
            p = part.strip()
            if not p:
                continue
            k = _norm_dynamics_opt(p)
            if k:
                items.append(k)
    vc = pd.Series(items).value_counts()
    return vc.reindex(DYNAMICS_ORDER, fill_value=0)


# ============================================================
# Team integration
# ============================================================

TEAM_INTEGRATION_ORDER = [
    "Well integrated with the team",
    "Indifferent with the team",
    "Detached from the team",
]


def _norm_team_integration(x: str) -> str:
    s = _norm(x)
    if "well" in s and "integr" in s:
        return "Well integrated with the team"
    if "indifferent" in s:
        return "Indifferent with the team"
    if "detached" in s:
        return "Detached from the team"
    return "Unknown"


# ============================================================
# Mistake-handling options
# ============================================================

MISTAKE_ORDER = [
    "Private Discussion",
    "Immediate Correction",
    "Immediate Confrontation",
    "Blame Approach",
    "Address Cause of The Problem",
    "Constructive Criticism",
    "Direct Escalation",
]

UNDESIRED = {"Immediate Confrontation", "Blame Approach"}


def _norm_mistake_opt(x: str) -> str | None:
    s = _norm(x)
    if "private" in s and "discuss" in s:
        return "Private Discussion"
    if "immediate" in s and "correct" in s:
        return "Immediate Correction"
    if "immediate" in s and "confront" in s:
        return "Immediate Confrontation"
    if "blame" in s:
        return "Blame Approach"
    if ("root" in s or "cause" in s) and ("problem" in s):
        return "Address Cause of The Problem"
    if "construct" in s and "critic" in s:
        return "Constructive Criticism"
    if "direct" in s and "escalat" in s:
        return "Direct Escalation"
    return None


def _mistake_counts(series: pd.Series) -> pd.Series:
    items: list[str] = []
    for v in series.dropna().astype(str):
        for part in re.split(r"[,\|;\n]+", v):
            p = part.strip()
            if not p:
                continue
            k = _norm_mistake_opt(p)
            if k:
                items.append(k)
    vc = pd.Series(items).value_counts()
    return vc.reindex(MISTAKE_ORDER, fill_value=0)


# ============================================================
# Recognition options
# ============================================================

RECOG_ORDER = [
    "Sending a recognition e-mail",
    "Providing continuous positive feedback",
    "Offering a wider scope of responsibilities",
    "Announcing it publicly",
    "Offering incentives",
]


def _norm_recog_opt(x: str) -> str | None:
    s = _norm(x)
    if "not applicable" in s or s == "na" or s == "n a":
        return None
    if "recognition" in s and ("mail" in s or "email" in s or "e mail" in s or "e-mail" in s):
        return "Sending a recognition e-mail"
    if "continuous" in s and "positive" in s and "feedback" in s:
        return "Providing continuous positive feedback"
    if ("wider" in s or "scope" in s) and "responsibil" in s:
        return "Offering a wider scope of responsibilities"
    if "announc" in s and ("public" in s or "publicly" in s):
        return "Announcing it publicly"
    if "incentive" in s or "incentives" in s:
        return "Offering incentives"
    return None


def _recog_counts(series: pd.Series) -> pd.Series:
    items: list[str] = []
    for v in series.dropna().astype(str):
        for part in re.split(r"[,\|;\n]+", v):
            p = part.strip()
            if not p:
                continue
            k = _norm_recog_opt(p)
            if k:
                items.append(k)
    counts = pd.Series(items).value_counts()
    return counts.reindex(RECOG_ORDER, fill_value=0)


# ============================================================
# Job-change categories
# ============================================================

CHANGE_CATS = [
    "Change in management",
    "Transfer to another department",
    "Shift in role",
    "Added responsibilities",
    "Promotion",
    "Other",
]


def _norm_change_option(x: str) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    if "promotion" in s:
        return "Promotion"
    if "added" in s and "respons" in s:
        return "Added responsibilities"
    if "shift" in s and "role" in s:
        return "Shift in role"
    if "transfer" in s and "department" in s:
        return "Transfer to another department"
    if "change" in s and "management" in s:
        return "Change in management"
    if "other" in s:
        return "Other"
    return "Other"


def _change_counts(series: pd.Series) -> pd.Series:
    items = []
    for v in series.dropna().astype(str):
        for part in re.split(r"[,\|;\n]+", v):
            p = part.strip()
            if p:
                items.append(_norm_change_option(p))
    counts = pd.Series(items).value_counts()
    return counts.reindex(CHANGE_CATS, fill_value=0)


# ============================================================
# Check-in frequency
# ============================================================

FREQ_CATS = ["Weekly", "Monthly", "Few", "Zero"]


def _norm_freq(x: str) -> str:
    s = _norm(x)
    if "week" in s:
        return "Weekly"
    if "month" in s:
        return "Monthly"
    if "few" in s:
        return "Few"
    if "zero" in s or s == "0":
        return "Zero"
    return "Unknown"


# ============================================================
# Resource options (Slide 27)
# ============================================================

RES_ORDER = [
    "Enrolling in the group's wellness sessions",
    "Seeking advice from HR",
    "Expressing complaints or miscellaneous ideas",
    "Other",
    "Not Applicable",
]


def _norm_resource_opt(x: str) -> str | None:
    s = _norm(x)

    if "wellness" in s or ("group" in s and "wellness" in s):
        return "Enrolling in the group's wellness sessions"
    if "seeking" in s and "advice" in s and "hr" in s:
        return "Seeking advice from HR"
    if "complaint" in s or ("miscellaneous" in s and "idea" in s):
        return "Expressing complaints or miscellaneous ideas"
    if "not applicable" in s:
        return "Not Applicable"
    if "other" in s:
        return "Other"

    return None


# ============================================================
# Pulse-check reasons
# ============================================================

PULSE_REASON_ORDER = [
    "To discuss career path",
    "To address challenges faced",
    "Compensation & Benefits concern",
    "To address personal issues",
    "To share ideas and insights with HR",
]


def _norm_pulse_reason(x: str) -> str | None:
    s = _norm(x)
    if "career" in s and "path" in s:
        return "To discuss career path"
    if "challenge" in s:
        return "To address challenges faced"
    if "compensation" in s or ("benefit" in s):
        return "Compensation & Benefits concern"
    if "personal" in s:
        return "To address personal issues"
    if ("share" in s and "idea" in s) or ("insight" in s and "hr" in s):
        return "To share ideas and insights with HR"
    return None


def _pulse_reason_counts(series: pd.Series) -> pd.Series:
    items: list[str] = []
    for v in series.dropna().astype(str):
        for part in re.split(r"[,\|;\n]+", v):
            p = part.strip()
            if not p:
                continue
            k = _norm_pulse_reason(p)
            if k:
                items.append(k)
    vc = pd.Series(items).value_counts()
    return vc.reindex(PULSE_REASON_ORDER, fill_value=0)


# ============================================================
# Date parsing helpers
# ============================================================

def _parse_any_date(series: pd.Series) -> pd.Series:
    """Parse dates robustly, handling text and Excel serial numbers."""
    if series is None or len(series) == 0:
        return pd.to_datetime(pd.Series([], dtype=object))

    raw = series.copy()
    dt_mf = pd.to_datetime(raw, errors="coerce", dayfirst=False)
    need_df = dt_mf.isna()
    if need_df.any():
        dt_df = pd.to_datetime(raw[need_df], errors="coerce", dayfirst=True)
        dt_mf.loc[need_df] = dt_df

    still_na = dt_mf.isna()
    if still_na.any():
        nums = pd.to_numeric(raw[still_na], errors="coerce")
        mask_num = nums.notna()
        if mask_num.any():
            dt_mf.loc[still_na[still_na].index[mask_num]] = pd.to_datetime(
                nums[mask_num], unit="d", origin="1899-12-30", errors="coerce"
            )
    return dt_mf


def _extract_month_year_label(raw: pd.Series) -> pd.Series:
    """Best-effort extraction of Month YYYY from raw date strings."""
    import calendar

    def norm_year(y: int) -> int:
        return 2000 + y if y < 100 else y

    def to_label(m: int, y: int) -> str | None:
        if 1 <= m <= 12 and y >= 1900:
            return f"{calendar.month_name[m]} {y}"
        return None

    months = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
        "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    }

    out = []
    for v in raw.astype(str).fillna(""):
        s = v.strip().lower()
        label = None
        if not s:
            out.append(label)
            continue
        m = re.search(
            r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
            r"\b[\s\-_/]*([0-9]{2,4})",
            s,
        )
        if m:
            mm = months.get(m.group(1), None)
            yy = norm_year(int(m.group(2)))
            label = to_label(mm, yy) if mm else None
        if not label:
            m2 = re.search(r"\b(\d{1,2})[\-/](\d{1,2})[\-/](\d{2,4})\b", s)
            if m2:
                mm = int(m2.group(1))
                yy = norm_year(int(m2.group(3)))
                label = to_label(mm, yy)
        if not label:
            m3 = re.search(r"\b(\d{1,2})[\-/](\d{2,4})\b", s)
            if m3:
                mm = int(m3.group(1))
                yy = norm_year(int(m3.group(2)))
                label = to_label(mm, yy)
        out.append(label)
    return pd.Series(out, index=raw.index)


# ============================================================
# Department helpers
# ============================================================

def _find_dept_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    for c in df.columns:
        lc_raw = str(c)
        lc = re.sub(r"\s+", " ", lc_raw).strip().lower().replace("\\", "/")
        lc_nospace = lc.replace(" ", "")
        if lc == "company name / department" or lc == "company name/department" or lc_nospace == "companyname/department":
            return c
    for c in df.columns:
        cn = re.sub(r"\s+", "", str(c).lower())
        if "company" in cn and "department" in cn:
            return c
    return None


def _norm_dept_value(x: str) -> str:
    x = str(x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def _filter_by_dept(df: pd.DataFrame, dept_col: str | None, selected: str) -> pd.DataFrame:
    if df is None or df.empty or not dept_col or selected == "All departments":
        return df
    s = df[dept_col].astype(str).map(_norm_dept_value)
    return df.loc[s == selected].copy()


def _dept_options(*dfs: pd.DataFrame) -> list[str]:
    vals: set[str] = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        c = None
        if "Company Name / Department" in df.columns:
            c = "Company Name / Department"
        elif "Company Name/Department" in df.columns:
            c = "Company Name/Department"
        else:
            c = _find_dept_col(df)
        if not c:
            for col in df.columns:
                norm_col = re.sub(r"[^a-z]+", " ", str(col).lower())
                if "company" in norm_col and "department" in norm_col:
                    c = col
                    break
        if c:
            vals |= set(df[c].dropna().astype(str).map(_norm_dept_value).unique().tolist())
    vals = {v for v in vals if v.strip()}
    return ["All departments"] + sorted(vals)


# ============================================================
# Year extraction from timestamps
# ============================================================

def _add_year_from_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    candidates = ["Timestamp", "timestamp", "Submission Timestamp", "Submitted at", "Date", "date"]
    ts_col = None
    for c in candidates:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        for c in df.columns:
            if "timestamp" in str(c).lower():
                ts_col = c
                break

    df = df.copy()
    if ts_col is None:
        df["Year"] = pd.NA
        return df

    ts = pd.to_datetime(df[ts_col], errors="coerce", infer_datetime_format=True)
    df["Year"] = ts.dt.year
    return df


# ============================================================
# KPI metrics aggregator
# ============================================================

def _metrics(emp_df: pd.DataFrame, mgr_df: pd.DataFrame) -> dict:
    kpi = KPIService()
    stress_h = kpi.detect_stress_header(emp_df)
    hr_h = kpi.detect_hr_pulse_header(emp_df)
    risk_h = kpi.detect_manager_risk_header(mgr_df)
    mgr_stress_h = kpi.detect_manager_stress_header(mgr_df)

    return {
        "emp_completion": kpi.employee_completion_rate_df(emp_df, expected_total=500),
        "mgr_completion": kpi.manager_completion_rate_df(mgr_df, expected_total=30),
        "emp_stress": (kpi.stress_rate(emp_df, stress_h) if stress_h else 0.0),
        "hr_pulse": (kpi.hr_pulse_rate(emp_df, hr_h) if hr_h else 0.0),
        "emp_at_risk": (
            kpi.at_risk_employee_rate(
                mgr_df, denominator="fixed", total_employees=500, risk_header=risk_h,
            )
            if risk_h
            else 0.0
        ),
        "mgr_stress": (kpi.manager_stress_rate(mgr_df, mgr_stress_h) if mgr_stress_h else 0.0),
    }


# ============================================================
# Percentage formatting helper
# ============================================================

def _pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "0.0%"
