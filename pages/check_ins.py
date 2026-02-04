import io
import os
import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from hr_analytics.services.checkin_cleaner_excel import CheckInExcelCleaner
from hr_analytics.services.kpis import KPIService
from hr_analytics.ui.style import apply_global_style, page_header, section_title, divider


# --------------------
# Page config MUST be first Streamlit call
# --------------------
st.set_page_config(page_title="Check-ins", layout="wide")


# --------------------
# Apply global UI
# --------------------
apply_global_style()
page_header("Check-ins", "KPIs • Analysis • Compare")


# --------------------
# Sidebar navigation
# --------------------
st.sidebar.title("Check-ins")
section = st.sidebar.radio(
    "Sections",
    ["KPIs", "Analysis", "Compare"],
    index=0
)

# ------------------------------------------------------------
# Defaults for analysis constants to avoid NameError at runtime
# ------------------------------------------------------------
# Lightweight English stopwords; keep minimal and safe
STOPWORDS: set[str] = {
    "the","a","an","and","or","but","if","then","so","of","in","on","at","to","for","from","by",
    "is","are","was","were","be","been","being","it","its","as","with","that","this","these","those",
    "we","us","our","you","your","they","them","their","i","me","my","mine",
}

# Topic buckets used for grouping open-ended responses
TOPIC_BUCKETS: dict[str, list[str]] = {
    "Workload & Pressure": [
        "workload","pressure","overload","deadline","deadlines","overtime","hours","capacity","too much","overworked"
    ],
    "Tools & Systems": [
        "tool","tools","system","systems","software","app","portal","access","login","bug","bugs","slow","issue","issues","downtime"
    ],
    "Process & Approvals": [
        "process","processes","approval","approvals","approve","workflow","procedure","procedures","steps","delay","delays","blocked","bottleneck","policy","policies"
    ],
    "Communication": [
        "communication","communicate","feedback","meeting","meetings","update","updates","clarity","information","email","emails","message","messages","respond","response"
    ],
    "Management & Leadership": [
        "manager","management","leadership","supervisor","support","micromanage","micromanagement","recognition","guidance","direction","decisions"
    ],
    "Culture & Team": [
        "culture","team","teams","collaboration","collaborate","conflict","environment","respect","inclusive","inclusion","toxic","engagement"
    ],
    "Career & Growth": [
        "career","growth","promotion","promotions","develop","development","training","learning","skills","skill","path","progression"
    ],
    "Compensation & Benefits": [
        "compensation","salary","salaries","pay","benefits","bonus","bonuses","allowance","package","raise","raises"
    ],
    "Wellbeing & Stress": [
        "wellbeing","wellness","stress","stressed","burnout","mental","health","work-life","balance","leave","time off","vacation"
    ],
}


# ============================================================
# Keyword matching helpers (KPIs + Analysis)
# ============================================================


def _make_col_finder(*keyword_sets):
    """Factory to create column finder functions with fallback keyword sets."""
    def finder(df: pd.DataFrame) -> str | None:
        for keywords in keyword_sets:
            col = _find_col_by_keywords(df, keywords)
            if col:
                return col
        return None
    return finder

# Column finders using factory pattern
_find_aligned_goals_col = _make_col_finder(
    ["aware", "aligned", "manager", "department", "goal"],
    ["aligned", "department", "goals"]
)

_find_if_no_elaborate_col = _make_col_finder(
    ["if", "no", "elaborate"],
    ["if", "no", "please", "elaborate"]
)



def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)          # collapse spaces/newlines
    s = re.sub(r"[^\w\s]", " ", s)      # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_mgr_behavior_question_col_emp(df: pd.DataFrame) -> str | None:
    # Employee question: "Please check the true statement(s) below about your manager"
    return (
        _find_col_by_keywords(df, ["true", "statement", "manager"]) or
        _find_col_by_keywords(df, ["check", "statement", "manager"]) or
        _find_col_by_keywords(df, ["below", "about", "your", "manager"])
    )

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

    # fallback
    return "OTHER"

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

def _yes_rate_from_keywords(df: pd.DataFrame, keywords_all: list[str]) -> tuple[float, str | None]:
    """Compute %Yes for a Yes/No question found by keywords. Returns (rate, colname)."""
    col = _find_col_by_keywords(df, keywords_all)
    if not col:
        return 0.0, None
    s = df[col].astype(str).str.strip().str.lower()
    yes = s.str.startswith(("yes", "y", "true", "1"))
    no  = s.str.startswith(("no", "n", "false", "0"))
    valid = yes | no
    rate = float(yes[valid].mean()) if valid.sum() > 0 else 0.0
    return rate, col

def _scale_good_rate_from_keywords(df: pd.DataFrame, keywords_all: list[str], good_min: int = 4) -> tuple[float, str | None]:
    """Compute % >= good_min for a 1–5 scale question found by keywords."""
    col = _find_col_by_keywords(df, keywords_all)
    if not col:
        return 0.0, None
    s = pd.to_numeric(df[col], errors="coerce")
    valid = s.notna()
    rate = float((s[valid] >= good_min).mean()) if valid.sum() > 0 else 0.0
    return rate, col

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


# --- Manager KPI helpers ---
def _find_ready_for_promotion_col(df: pd.DataFrame) -> str | None:
    # Adjust keywords if your header is slightly different
    return _find_col_by_keywords(df, ["ready", "promotion"])

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

def _yes_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.str.startswith(("yes", "y", "true", "1"))

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

def _no_mask(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.str.startswith(("no", "n", "false", "0"))

# --------------------
# Manager: At-Risk detail detection helpers
# --------------------
def _find_risk_reason_col(df: pd.DataFrame) -> str | None:
    # Prefer exact header provided by user
    exact = "If yes, please elaborate on the case"
    if exact in df.columns:
        return exact
    # Fallback: explicit reason columns near risk context
    col = _find_col_by_keywords(df, ["reason", "risk"]) or _find_col_by_keywords(df, ["reason"]) or _find_col_by_keywords(df, ["explain"]) or _find_col_by_keywords(df, ["incident"]) or _find_col_by_keywords(df, ["elaborate"]) 
    # Ensure we don't pick yes/no fields
    if col and _looks_like_yesno(df[col]):
        col = None
    return col

def _find_pip_col(df: pd.DataFrame) -> str | None:
    # Look for PIP indicator
    for keys in (["pip"], ["performance", "improvement", "plan"], ["enroll", "pip"], ["should", "pip"]):
        c = _find_col_by_keywords(df, keys)
        if c:
            return c
    return None

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

def _parse_any_date(series: pd.Series) -> pd.Series:
    """Parse dates robustly, handling text formats and Excel serial numbers.
    Prefers month/day/year (dayfirst=False) and falls back to day-first only when needed.
    Returns a datetime64[ns] Series with NaT where parsing fails.
    """
    if series is None or len(series) == 0:
        return pd.to_datetime(pd.Series([], dtype=object))

    raw = series.copy()
    # First pass: month-first (MM/DD/YYYY)
    dt_mf = pd.to_datetime(raw, errors="coerce", dayfirst=False)
    # Second pass: day-first (DD/MM/YYYY) only where first failed
    need_df = dt_mf.isna()
    if need_df.any():
        dt_df = pd.to_datetime(raw[need_df], errors="coerce", dayfirst=True)
        dt_mf.loc[need_df] = dt_df

    # Handle Excel serial numbers (days since 1899-12-30) only where still NaT
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
    """Best-effort extraction of Month YYYY from raw date strings when parsing fails.
    Handles cases like 'Dec 2025', 'December-2025', '12/15/2025', '12/2025', '12-25-25'.
    Assumes month-first for numeric formats.
    """
    import calendar
    def norm_year(y: int) -> int:
        return 2000 + y if y < 100 else y

    def to_label(m: int, y: int) -> str | None:
        if 1 <= m <= 12 and y >= 1900:
            return f"{calendar.month_name[m]} {y}"
        return None

    months = {
        'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
        'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,
        'sep':9,'sept':9,'september':9,'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12
    }

    out = []
    for v in raw.astype(str).fillna(""):
        s = v.strip().lower()
        label = None
        if not s:
            out.append(label)
            continue
        # Month name + year
        m = re.search(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b[\s\-_/]*([0-9]{2,4})", s)
        if m:
            mm = months.get(m.group(1), None)
            yy = norm_year(int(m.group(2)))
            label = to_label(mm, yy) if mm else None
        if not label:
            # MM/DD/YYYY or M/D/YY (assume month-first)
            m2 = re.search(r"\b(\d{1,2})[\-/](\d{1,2})[\-/](\d{2,4})\b", s)
            if m2:
                mm = int(m2.group(1)); yy = norm_year(int(m2.group(3)))
                label = to_label(mm, yy)
        if not label:
            # MM/YYYY
            m3 = re.search(r"\b(\d{1,2})[\-/](\d{2,4})\b", s)
            if m3:
                mm = int(m3.group(1)); yy = norm_year(int(m3.group(2)))
                label = to_label(mm, yy)
        out.append(label)
    return pd.Series(out, index=raw.index)

# ------------------------------------------------------------
# Identity + datatype detection helpers (Analysis)
# ------------------------------------------------------------
IDENTITY_HINTS = [
    "timestamp", "email", "name", "department", "company", "manager", "subordinate"
]

def _is_identity_col(col_name: str) -> bool:
    h = _norm(col_name)
    return any(tok in h for tok in IDENTITY_HINTS)

def _looks_like_yesno(series: pd.Series) -> bool:
    if series is None or len(series) == 0:
        return False
    s = series.astype(str).str.strip().str.lower()
    yes = s.str.startswith(("yes", "y", "true", "1"))
    no  = s.str.startswith(("no", "n", "false", "0"))
    valid = yes | no
    # Consider it yes/no-like if a majority of non-empty values match yes/no
    non_empty = s.ne("")
    denom = max(int(non_empty.sum()), 1)
    return int(valid.sum()) / denom >= 0.6

def _count_yes_no(df: pd.DataFrame, col: str) -> tuple[int, int, float, float]:
    """Returns (yes_count, no_count, yes_pct, no_pct) for a yes/no column."""
    if not col or col not in df.columns:
        return 0, 0, 0.0, 0.0
    yes_mask = _yes_mask(df[col])
    no_mask = _no_mask(df[col])
    yes_count = int(yes_mask.sum())
    no_count = int(no_mask.sum())
    total = yes_count + no_count
    yes_pct = (yes_count / total * 100) if total > 0 else 0.0
    no_pct = (no_count / total * 100) if total > 0 else 0.0
    return yes_count, no_count, yes_pct, no_pct

def _looks_like_scale_1_5(series: pd.Series) -> bool:
    """Check if a series looks like a 1-5 scale rating."""
    if series is None or len(series) == 0:
        return False
    numeric = pd.to_numeric(series, errors='coerce')
    non_null = numeric.dropna()
    if len(non_null) == 0:
        return False
    # Check if most values are between 1 and 5
    in_range = non_null.between(1, 5)
    return int(in_range.sum()) / len(non_null) >= 0.6

def _safe_percentage(numerator: int, denominator: int) -> float:
    """Safely calculate percentage, returning 0 if denominator is 0."""
    return (numerator / denominator * 100) if denominator > 0 else 0.0

def _display_yes_no_metrics(df: pd.DataFrame, col: str, yes_label: str = "YES", no_label: str = "NO"):
    """Display yes/no metrics in two columns with counts and percentages."""
    yes_count, no_count, yes_pct, no_pct = _count_yes_no(df, col)
    c1, c2 = st.columns(2)
    c1.metric(yes_label, yes_count, f"{yes_pct:.1f}%")
    c2.metric(no_label, no_count, f"{no_pct:.1f}%")
    return yes_count, no_count

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
                mgr_df, denominator="fixed", total_employees=500, risk_header=risk_h
            ) if risk_h else 0.0
        ),
        "mgr_stress": (kpi.manager_stress_rate(mgr_df, mgr_stress_h) if mgr_stress_h else 0.0),
    }

# ============================================================
# Department detection and filtering helpers
# ============================================================
def _find_dept_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    # Prefer exact label if present
    for c in df.columns:
        lc_raw = str(c)
        lc = re.sub(r"\s+", " ", lc_raw).strip().lower().replace("\\", "/")
        lc_nospace = lc.replace(" ", "")
        if lc == "company name / department" or lc == "company name/department" or lc_nospace == "companyname/department":
            return c
    # Fallback: any header containing both tokens
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
    vals = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        # Prefer explicit known headers
        c = None
        if "Company Name / Department" in df.columns:
            c = "Company Name / Department"
        elif "Company Name/Department" in df.columns:
            c = "Company Name/Department"
        else:
            c = _find_dept_col(df)

        # Broader token-based detection as last resort
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


# --------------------
# Check-in frequency helpers (used in Employee & Manager and Compare sections)
# --------------------
def _find_checkin_freq_col_emp(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["had", "check", "in", "meeting", "with", "my", "manager"]) or \
        _find_col_by_keywords(df, ["check", "in", "meeting", "my", "manager"])

def _find_checkin_freq_col_mgr(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["had", "check", "in", "meeting", "with", "this", "person"]) or \
        _find_col_by_keywords(df, ["check", "in", "meeting", "this", "person"])

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


# --------------------
# KPI maps (keyword-based)
# --------------------
EMP_YN_KPIS = [
    ("Discussed professional goals", ["discuss", "professional", "goals"]),
    ("Tasks aligned with growth", ["tasks", "aligned", "growth"]),
    ("Manager considers input", ["seek", "consider", "input"]),
    ("Collaborative culture", ["culture", "collaborative", "support"]),
    ("Recommend company", ["recommend", "company"]),
    ("Job requirements changed", ["encountered", "changes", "job", "requirements"]),
   # ("HR pulse requested", ["pulse", "check", "hr"]),
]

EMP_SCALE_KPIS = [
    ("Adapted well (4–5)", ["able", "adapt", "job", "requirements"], 4),
]


# ============================================================
# YoY 
# ============================================================
def _find_data_candidates(base: Path) -> list[Path]:
    data_dir = base / "Data"
    exts = {".xlsx", ".xls", ".csv"}
    found: list[Path] = []
    if data_dir.exists():
        for p in data_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("~$"):
                found.append(p)
    return found

def _classify_year_files(paths: list[Path]) -> dict[int, dict[str, Path]]:
    by_year: dict[int, dict[str, Path]] = {}
    for p in paths:
        name = p.name.lower()

        years_in_name = [int(m) for m in re.findall(r"\b(20\d{2})\b", p.name)]
        yr = years_in_name[-1] if years_in_name else None
        if yr is None:
            years_in_path = [int(m) for m in re.findall(r"\b(20\d{2})\b", str(p))]
            yr = years_in_path[-1] if years_in_path else None
        if yr is None:
            continue

        toks = name.replace("-", " ").replace("_", " ").split()
        def any_has(*kw):
            return any(any(k in t for t in toks) for k in kw)

        if any_has("manager") and any_has("check", "checkin", "check-ins"):
            by_year.setdefault(yr, {})["mgr"] = p
            continue
        if any_has("employee") and any_has("check", "checkin", "check-ins"):
            by_year.setdefault(yr, {})["emp"] = p
            continue
        if ("performance" in name) and any_has("check", "checkin", "check-ins") and not any_has("employee"):
            by_year.setdefault(yr, {})["mgr"] = p
            continue

    return by_year


def _available_years_from_data(base_path: Path) -> tuple[list[int], dict[int, dict[str, Path]]]:
    paths = _find_data_candidates(base_path)
    by_year = _classify_year_files(paths)
    years = sorted([y for y in by_year.keys() if "emp" in by_year[y] and "mgr" in by_year[y]])
    return years, by_year

@st.cache_data(show_spinner=True)
def _load_and_clean_year_from_data(year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: (cleaned_emp, cleaned_mgr, combined) for the selected year
    """
    base_path = Path(os.getcwd())
    years, by_year = _available_years_from_data(base_path)

    if year not in by_year or "emp" not in by_year[year] or "mgr" not in by_year[year]:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_mena = _load_static_mena()
    if df_mena.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _read_any(p: Path) -> pd.DataFrame:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(str(p))
        return pd.read_excel(str(p))

    emp_raw = _read_any(by_year[year]["emp"])
    mgr_raw = _read_any(by_year[year]["mgr"])

    cleaner = CheckInExcelCleaner()
    cleaned_emp = cleaner.clean_employee_checkin(emp_raw, df_mena)
    cleaned_mgr = cleaner.clean_manager_checkin(mgr_raw, df_mena)
    combined = cleaner.combine_cleaned(cleaned_emp, cleaned_mgr)

    return cleaned_emp, cleaned_mgr, combined


# ============================================================
# Shared helpers
# ============================================================
def _download_fig_png(fig, filename: str, key: str):
    if fig is None:
        st.caption("No chart to download.")
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="⬇️ Download Chart (PNG)",
        data=buf,
        file_name=filename,
        mime="image/png",
        use_container_width=True,
        key=key,
    )


def _download_df_csv(df: pd.DataFrame, filename: str, key: str):
    """Download a dataframe as CSV."""
    if df is None or df.empty:
        st.caption("No data to download.")
        return
    st.download_button(
        label="⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
        key=key,
    )

def _download_fig_png(fig, filename: str, key: str):
    """Download a matplotlib figure as PNG."""
    if fig is None:
        st.caption("No chart to download.")
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="⬇️ Download Chart (PNG)",
        data=buf,
        file_name=filename,
        mime="image/png",
        use_container_width=True,
        key=key,
    )


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


PULSE_REASON_ORDER = [
    "To discuss career path",
    "To address challenges faced",
    "Compensation & Benefits concern",
    "To address personal issues",
    "To share ideas and insights with HR",
]

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


EMP_STRESS_REASON_ORDER = [
    "Personal",
    "Unrealistic Goals",
    "Tight Deadlines",
    "Workload",
    "Relationship with manager",
    "Relationship with team members",
    "Others",
]

def _norm_emp_stress_reason(x: str) -> str | None:
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

def _find_emp_stress_reasons_col(df: pd.DataFrame) -> str | None:
    # "In your opinion, what are the reasons behind this stress?"
    for c in df.columns:
        h = _norm(c)
        if ("reasons" in h) and ("behind" in h) and ("stress" in h):
            return c
    for c in df.columns:
        h = _norm(c)
        if ("reasons" in h) and ("stress" in h):
            return c
    return None

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

def _find_mgr_stress_reasons_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        h = _norm(c)
        if ("reasons" in h) and ("behind" in h) and ("stress" in h):
            return c
    # fallback: reasons + stress
    for c in df.columns:
        h = _norm(c)
        if ("reasons" in h) and ("stress" in h):
            return c
    return None

MGR_STRESS_REASON_ORDER = [
    "Workload",
    "Tight Deadlines",
    "Personal",
    "Unrealistic Goals",
    "Relationship with manager",
    "Relationship with team members",
    "Others",
]

def _norm_mgr_stress_reason(x: str) -> str | None:
    s = _norm(x)
    if "workload" in s:
        return "Workload"
    if "tight" in s and "deadline" in s:
        return "Tight Deadlines"
    if "personal" in s:
        return "Personal"
    if "unrealistic" in s and "goal" in s:
        return "Unrealistic Goals"
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
            k = _norm_mgr_stress_reason(p)
            if k:
                items.append(k)
    vc = pd.Series(items).value_counts()
    return vc.reindex(MGR_STRESS_REASON_ORDER, fill_value=0)


STRESS_FREQ_ORDER = ["Less frequent", "Frequent", "Extremely frequent"]

def _norm_stress_freq(x: str) -> str:
    s = _norm(x)
    if "less" in s and "frequent" in s:
        return "Less frequent"
    if s.startswith("frequent") or ("frequent" in s and "extreme" not in s):
        return "Frequent"
    if "extreme" in s and "frequent" in s:
        return "Extremely frequent"
    return "Unknown"

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

def _find_mgr_dynamics_col_mgr(df: pd.DataFrame) -> str | None:
    # "This member contributes to enhance the department dynamics by"
    for c in df.columns:
        h = _norm(c)
        if ("contributes" in h) and ("department dynamics" in h) and ("enhance" in h or "enhancement" in h):
            return c
    # fallback: dynamics + contributes
    for c in df.columns:
        h = _norm(c)
        if ("dynamics" in h) and ("contribut" in h):
            return c
    return None


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

def _find_dynamics_col_emp(df: pd.DataFrame) -> str | None:
    # "In what 3 areas do you believe that the department dynamics need enhancement"
    for c in df.columns:
        h = _norm(c)
        if ("department dynamics" in h) and ("enhancement" in h or "enhance" in h) and ("3 areas" in h or "in what 3" in h):
            return c
    # fallback: dynamics + enhancement
    for c in df.columns:
        h = _norm(c)
        if ("dynamics" in h) and ("enhance" in h or "enhancement" in h):
            return c
    return None


WORK_CULTURE_ORDER = [
    "Clan Culture",
    "Hierarchy Culture",
    "Innovative Culture",
    "Market Driven Culture",
    "Purpose Driven Culture",
    "Creative Culture",
    "Adhocracy Culture",
    "Customer Focus Culture",
]

def _norm_work_culture_opt(x: str) -> str | None:
    s = _norm(x)

    if "clan" in s:
        return "Clan Culture"
    if "hierarchy" in s:
        return "Hierarchy Culture"
    if "innov" in s:
        return "Innovative Culture"
    if "market" in s and "driven" in s:
        return "Market Driven Culture"
    if "purpose" in s and "driven" in s:
        return "Purpose Driven Culture"
    if "creative" in s:
        return "Creative Culture"
    if "adhocracy" in s:
        return "Adhocracy Culture"
    if "customer" in s and "focus" in s:
        return "Customer Focus Culture"

    return None

def _work_culture_counts(series: pd.Series) -> pd.Series:
    items: list[str] = []
    for v in series.dropna().astype(str):
        for part in re.split(r"[,\|;\n]+", v):
            p = part.strip()
            if not p:
                continue
            k = _norm_work_culture_opt(p)
            if k:
                items.append(k)
    vc = pd.Series(items).value_counts()
    return vc.reindex(WORK_CULTURE_ORDER, fill_value=0)


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

def _find_team_integration_col_emp(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["team", "integration", "i", "am"]) or \
           _find_col_by_keywords(df, ["team", "integration"])

def _find_team_integration_col_mgr(df: pd.DataFrame) -> str | None:
    return _find_col_by_keywords(df, ["team", "integration", "this", "person"]) or \
           _find_col_by_keywords(df, ["team", "integration"])


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


def _find_input_seek_col_emp(df: pd.DataFrame) -> str | None:
    # Employee: "Does your manager actively seek and consider your input..."
    return _find_col_by_keywords(df, ["actively", "seek", "consider", "input", "department"]) or \
           _find_col_by_keywords(df, ["seek", "consider", "input"])

def _find_input_seek_col_mgr(df: pd.DataFrame) -> str | None:
    # Manager: "Do you actively seek and consider your team members’ input?"
    return _find_col_by_keywords(df, ["actively", "seek", "consider", "team", "members", "input"]) or \
           _find_col_by_keywords(df, ["seek", "consider", "team", "input"]) or \
           _find_col_by_keywords(df, ["seek", "consider", "input"])


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

    # If the text doesn't match, return None so it doesn't pollute the table
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


def _find_job_change_yn_col_emp(df: pd.DataFrame) -> str | None:
    # employee: "During the year 2025, I encountered changes in my job requirements"
    return _find_col_by_keywords(df, ["encountered", "changes", "job", "requirements"])

def _find_job_change_yn_col_mgr(df: pd.DataFrame) -> str | None:
    # manager: "During the Year 2025, this person encountered changes in his/her job requirements"
    return _find_col_by_keywords(df, ["this", "person", "encountered", "changes", "job", "requirements"]) or \
           _find_col_by_keywords(df, ["encountered", "changes", "job", "requirements"])

def _find_job_change_types_col(df: pd.DataFrame) -> str | None:
    # both forms: "If yes, what changes did ... encounter"
    return _find_col_by_keywords(df, ["if", "yes", "what", "changes"]) or _find_col_by_keywords(df, ["what", "changes", "encounter"])

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
    # ensure all categories exist
    return counts.reindex(CHANGE_CATS, fill_value=0)


def _completion_stats(sent: int, received: int) -> dict:
    sent = int(max(sent, 0))
    received = int(max(received, 0))
    received = min(received, sent) if sent > 0 else received  # avoid >100%
    pending = max(sent - received, 0)
    completion = (received / sent * 100.0) if sent > 0 else 0.0
    pending_pct = (pending / sent * 100.0) if sent > 0 else 0.0
    return {
        "sent": sent,
        "received": received,
        "pending": pending,
        "completion": completion,
        "pending_pct": pending_pct,
    }


def _pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "0.0%"

def _scale_counts_1_5(df: pd.DataFrame, col: str | None) -> dict[int, int]:
    if df is None or df.empty or not col or col not in df.columns:
        return {i: 0 for i in range(1, 6)}
    s = pd.to_numeric(df[col], errors="coerce")
    counts = s.value_counts(dropna=True).to_dict()
    return {i: int(counts.get(i, 0)) for i in range(1, 6)}

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
# Department helpers
# ============================================================
# Keyword matching helpers (KPIs + Analysis)
# ============================================================

def _best_topic_for_text(text: str) -> tuple[str, int]:
    """
    Assign EACH text to only ONE best topic.
    Returns (topic, score). If no keywords match -> ("Uncategorized", 0)
    """
    t = (text or "").lower()

    best_topic = "Uncategorized"
    best_score = 0

    # tie-breaker: earlier topic in TOPIC_BUCKETS wins if same score
    for topic, kws in TOPIC_BUCKETS.items():
        score = sum(1 for k in kws if k in t)
        if score > best_score:
            best_topic = topic
            best_score = score

    return best_topic, best_score


def _topic_summary_with_sentiment(texts: list[str]) -> pd.DataFrame:
    """
    Improvements:
    - Each answer counted in ONLY ONE topic (best keyword-hit topic)
    - Adds Uncategorized
    - Adds Example neutral
    - Removes Avg sentiment (keeps only counts)
    """
    total = len(texts)

    # Assign each text to one topic
    buckets: dict[str, list[str]] = {topic: [] for topic in TOPIC_BUCKETS.keys()}
    buckets["Uncategorized"] = []

    for txt in texts:
        topic, score = _best_topic_for_text(txt)
        buckets[topic].append(txt)

    rows = []
    for topic in list(TOPIC_BUCKETS.keys()) + ["Uncategorized"]:
        matched = buckets.get(topic, [])
        if not matched:
            rows.append([topic, 0, 0.0, 0, 0, 0, "", "", ""])
            continue

        scores = [_sentiment_score(t) for t in matched]
        labels = [_sentiment_label(s) for s in scores]

        neg = sum(1 for l in labels if l == "Negative")
        neu = sum(1 for l in labels if l == "Neutral")
        pos = sum(1 for l in labels if l == "Positive")

        neg_examples = [t[:160] for t, l in zip(matched, labels) if l == "Negative"][:2]
        neu_examples = [t[:160] for t, l in zip(matched, labels) if l == "Neutral"][:2]
        pos_examples = [t[:160] for t, l in zip(matched, labels) if l == "Positive"][:2]

        pct_responses = (len(matched) / total * 100.0) if total else 0.0

        rows.append([
            topic,
            len(matched),
            round(pct_responses, 1),
            neg,
            neu,
            pos,
            " | ".join(neg_examples),
            " | ".join(neu_examples),
            " | ".join(pos_examples),
        ])

    df = pd.DataFrame(rows, columns=[
        "Topic",
        "Mentions",
        "% of responses",
        "Negative",
        "Neutral",
        "Positive",
        "Example negative",
        "Example neutral",
        "Example positive",
    ])

    # Sort by most mentioned first
    return df.sort_values(["Mentions", "% of responses"], ascending=[False, False])


# ============================================================
# Optional: Convert "Reasons for No" topics into Action Plan
# ============================================================
TOPIC_ACTIONS = {
    "Workload & Pressure": [
        "Review workload distribution and deadline pressure; identify overload peaks.",
        "Agree on top priorities and remove/shift non-essential tasks."
    ],
    "Tools & Systems": [
        "Collect tool/system pain points and prioritize fixes with IT (access, slowness, bugs).",
        "Provide quick training / cheat-sheets for the main tools used."
    ],
    "Process & Approvals": [
        "Map approval bottlenecks and simplify steps or set clear SLAs.",
        "Clarify ownership + escalation path when requests are blocked."
    ],
    "Communication": [
        "Set a consistent feedback cadence (e.g., monthly 1:1) and align expectations.",
        "Share clearer priorities/goals and confirm understanding."
    ],
    "Management & Leadership": [
        "Coach managers on support/recognition and reduce micromanagement patterns.",
        "Standardize 1:1 agenda templates (goals, blockers, development)."
    ],
    "Culture & Team": [
        "Improve collaboration norms (handoffs, retros, rules of engagement).",
        "Address repeated conflict patterns through HR/leadership intervention."
    ],
    "Career & Growth": [
        "Introduce a simple development plan template (skills, goals, next steps).",
        "Clarify career paths and training options; track follow-ups."
    ],
    "Compensation & Benefits": [
        "Collect themes and benchmark; communicate what’s feasible and timeline.",
        "If changes aren’t possible, improve transparency and alternatives."
    ],
    "Wellbeing & Stress": [
        "Promote wellbeing resources and encourage time-off planning.",
        "Reduce stress drivers via workload/process fixes; monitor trends."
    ],
}

def _reasons_action_plan(texts: list[str]) -> pd.DataFrame:
    total = len(texts)
    rows = []

    for topic, kws in TOPIC_BUCKETS.items():
        matched = [t for t in texts if any(k in t.lower() for k in kws)]
        if not matched:
            continue

        scores = [_sentiment_score(t) for t in matched]
        labels = [_sentiment_label(s) for s in scores]
        neg = sum(1 for l in labels if l == "Negative")
        neu = sum(1 for l in labels if l == "Neutral")
        pos = sum(1 for l in labels if l == "Positive")

        pct = (len(matched) / total * 100.0) if total else 0.0
        avg_sent = float(sum(scores) / len(scores)) if scores else 0.0

        examples = [t[:140] for t in matched][:2]
        actions = TOPIC_ACTIONS.get(topic, ["Review and address this theme."])

        rows.append({
            "Theme": topic,
            "% of No elaborations": round(pct, 1),
            "Mentions": len(matched),
            "Avg sentiment (-1..1)": round(avg_sent, 3),
            "Negative": neg,
            "Neutral": neu,
            "Positive": pos,
            "Recommended actions": " • ".join(actions[:2]),
            "Example snippets": " | ".join(examples)
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Theme","% of No elaborations","Mentions","Avg sentiment (-1..1)",
            "Negative","Neutral","Positive","Recommended actions","Example snippets"
        ])

    return pd.DataFrame(rows).sort_values(
        ["Mentions", "Avg sentiment (-1..1)"], ascending=[False, True]
    )

# ============================================================
# "Reasons for No" pairing helpers
# ============================================================
ELAB_NO_KEYWORDS = ["if no", "if answered no", "if you answered no", "if not", "if your answer is no"]

def _is_elaboration_header(col_name: str) -> bool:
    h = _norm(col_name)

    # exclude YES follow-ups
    if "if yes" in h or "if answered yes" in h or "if you answered yes" in h:
        return False

    # exclude non-no followups like "indifferent/detached" or numeric followups
    if "indifferent" in h or "detached" in h:
        return False
    if "if answered 1" in h or "1 or 2" in h:
        return False

    # ✅ only accept if it explicitly refers to NO
    return any(k in h for k in ELAB_NO_KEYWORDS)


def _is_yesno_series(series: pd.Series) -> bool:
    return _looks_like_yesno(series)

def _shorten_header(h: str, max_len: int = 70) -> str:
    h = str(h).replace("\n", " ").strip()
    h = re.sub(r"\s+", " ", h)
    return (h[:max_len] + "…") if len(h) > max_len else h

def _build_reasons_for_no_pairs(df: pd.DataFrame) -> list[dict]:
    cols = list(df.columns)
    pairs = []
    for i, c in enumerate(cols):
        if not _is_elaboration_header(c):
            continue
        parent = None
        for j in range(i - 1, -1, -1):
            cj = cols[j]
            if _is_elaboration_header(cj):
                continue
            if _is_yesno_series(df[cj]):
                parent = cj
                break
        if parent:
            label = f"Reasons for No — {_shorten_header(parent)}"
            count = sum(1 for p in pairs if p["label"].startswith(label))
            if count > 0:
                label = f"{label} ({count+1})"

            pairs.append({
                "label": label,
                "parent_col": parent,
                "elab_col": c
            })

    return pairs


# ============================================================
# Session state init
# ============================================================
if "clean_ready" not in st.session_state:
    st.session_state.clean_ready = False
if "clean_error" not in st.session_state:
    st.session_state.clean_error = None
if "cleaned_emp" not in st.session_state:
    st.session_state.cleaned_emp = None
    st.session_state.cleaned_mgr = None
    st.session_state.combined = None


if "yoy_ready" not in st.session_state:
    st.session_state.yoy_ready = False
    st.session_state.yoy_payload = None


# Back button (always visible)
if st.button("⬅ Back to Home"):
    st.switch_page("Home.py")
  # change back to streamlit_app.py if you didn't rename


# ============================================================
# Data loaders
# ============================================================
@st.cache_data(show_spinner=True)
def _load_static_mena() -> pd.DataFrame:
    base = Path(os.getcwd())
    exts = {".xlsx", ".xls", ".csv"}

    def _read_any(p: Path) -> pd.DataFrame:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(str(p))
        return pd.read_excel(str(p))

    # ✅ Force this exact file name (place it inside Data/)
    forced = base / "Data" / "Mena Report - Copy Tech.xlsx"
    if forced.exists():
        return _read_any(forced)

    mena_files = []
    data_dir = base / "Data"
    if data_dir.exists():
        for p in data_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("~$"):
                n = p.name.lower()
                if ("mena" in n) and ("report" in n):
                    mena_files.append(p)

    if not mena_files:
        for name in [
            "Mena Report.xlsx", "Mena_Report.xlsx", "mena_report.xlsx",
            "Mena Report.csv", "Mena_Report.csv", "mena_report.csv",
        ]:
            p = base / name
            if p.exists():
                mena_files.append(p)

    if mena_files:
        p = max(mena_files, key=lambda q: q.stat().st_mtime)
        return _read_any(p)

    return pd.DataFrame()

@st.cache_data(show_spinner=True)
def _load_df(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = file.name.lower()
    data = file.read()
    file.seek(0)
    bio = io.BytesIO(data)
    if name.endswith(".csv"):
        return pd.read_csv(bio)
    return pd.read_excel(bio)


def _clean_text(x: str) -> str:
    x = str(x)
    x = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[email]", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _detect_open_ended_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    if df is None or df.empty:
        return cols

    for c in df.columns:
        if _is_identity_col(c):
            continue

        s = df[c]

        # exclude yes/no and 1–5 scale columns
        if _looks_like_yesno(s) or _looks_like_scale_1_5(s):
            continue

        if s.dtype == object:
            vals = s.dropna().astype(str)
            if vals.empty:
                continue

            avg_len = vals.map(len).mean()
            uniq_ratio = vals.nunique() / max(len(vals), 1)

            # heuristic: open-ended tends to have longer + more unique responses
            if avg_len >= 20 and uniq_ratio >= 0.25:
                cols.append(c)

    return cols


# -------------------------
# Sentiment helpers (VADER)
# -------------------------
_analyzer = SentimentIntensityAnalyzer()

def _sentiment_score(text: str) -> float:
    return _analyzer.polarity_scores(str(text))["compound"]

def _sentiment_label(score: float) -> str:
    if score <= -0.05:
        return "Negative"
    if score >= 0.05:
        return "Positive"
    return "Neutral"


def _tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) >= 3 and t not in STOPWORDS]

def _top_keywords(texts: list[str], top_n: int = 25) -> pd.DataFrame:
    counter = Counter()
    for t in texts:
        counter.update(_tokenize(t))
    most = counter.most_common(top_n)
    return pd.DataFrame(most, columns=["Keyword", "Count"])


def _analysis_ui(df_raw: pd.DataFrame, who: str, year: int | None = None):
    df = df_raw.copy()
    section_title(f"{who} analysis" + (f" — {year}" if year else ""))


    # ✅ define these so they always exist (prevents NameError)
    pulse_col = None
    reason_col = None

    # -------------------------
    # HR Pulse: Reasons (Yes responders only) — categorical breakdown
    # -------------------------
    pulse_col = _find_col_by_keywords(df, ["pulse", "check", "hr"])
    reason_col = _find_col_by_keywords(df, ["specify", "reason"])
    if reason_col is None:
        reason_col = _find_col_by_keywords(df, ["career", "path"])  # fallback

    if pulse_col and reason_col:
        divider()
        section_title("HR Pulse checks — Reasons (only employees who answered Yes)")

        s = df[pulse_col].astype(str).str.strip().str.lower()
        yes_mask = s.str.startswith(("yes", "y", "true", "1"))
        yes_df = df[yes_mask].copy()

        st.caption(f"Yes responders: {len(yes_df)}")

        if len(yes_df) == 0:
            st.info("No employees answered Yes for HR pulse checks in the selected year.")
        else:
            # Multi-select breakdown (comma / | separated)
            vals = yes_df[reason_col].dropna().astype(str)

            items = []
            for v in vals:
                for part in v.replace("|", ",").split(","):
                    p = part.strip()
                    if p:
                        items.append(p)

            if items:
                counts = pd.Series(items).value_counts().reset_index()
                counts.columns = ["Reason", "Count"]
                counts["% of Yes responders"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
                st.dataframe(counts, use_container_width=True, hide_index=True)
                _download_df_csv(counts, "hr_pulse_reasons_breakdown.csv", key="dl_hr_pulse_reasons")

            else:
                st.info("Reason column exists but contains no selections.")

    # -------------------------
    # Reasons for "No" (paired elaborations)
    # -------------------------
    divider()
    pairs = _build_reasons_for_no_pairs(df)

    if pairs:
        section_title("Reasons for “No” (linked follow-ups)")
        pick = st.selectbox(
            "Choose a follow-up to analyze",
            options=pairs,
            format_func=lambda x: x["label"],
            key=f"reasons_pair_{who}"
        )

        parent_col = pick["parent_col"]
        elab_col = pick["elab_col"]

        s = df[parent_col].astype(str).str.strip().str.lower()
        no_mask = s.str.startswith(("no", "n", "false", "0"))

        texts = df.loc[no_mask, elab_col].dropna().astype(str).map(_clean_text).tolist()
        texts = [t for t in texts if t.strip()]

        st.caption(f"No responses: {int(no_mask.sum())} | Non-empty elaborations: {len(texts)}")

        if texts:
            col1, col2 = st.columns([1.2, 0.8])
            with col1:
                section_title("Recurring topics (with sentiment)")
                df_topics = _topic_summary_with_sentiment(texts)
                st.dataframe(df_topics, use_container_width=True, hide_index=True)
                _download_df_csv(df_topics, "reasons_for_no_topics_sentiment.csv", key="dl_no_topics_sentiment")

                divider()

                section_title("Action plan (Themes → Suggested actions)")

                plan_df = _reasons_action_plan(texts)

                if plan_df.empty:
                    st.info("No clear themes detected to generate an action plan.")
                else:
                    st.dataframe(plan_df, use_container_width=True, hide_index=True)
                    _download_df_csv(plan_df, "reasons_for_no_action_plan.csv", key="dl_no_action_plan")

                    st.markdown("**Top priorities (quick view):**")
                    for _, r in plan_df.head(3).iterrows():
                        st.write(f"- **{r['Theme']}** ({r['% of No elaborations']}%): {r['Recommended actions']}")

            with col2:
                section_title("Top keywords")
                df_kw = _top_keywords(texts, top_n=25)
                st.dataframe(df_kw, use_container_width=True, hide_index=True)
                _download_df_csv(df_kw, "reasons_for_no_top_keywords.csv", key="dl_no_top_keywords")
        else:
            st.info("No elaboration text found for employees who answered 'No'.")
    else:
        section_title("Reasons for “No” (linked follow-ups)")
        st.info("No 'If no, please elaborate' follow-up columns detected for this dataset/year.")

    divider()

    # -------------------------
    # General open-ended analysis (exclude elaborations + exclude HR pulse reason column)
    # -------------------------
    auto_cols = _detect_open_ended_columns(df)
    auto_cols = [c for c in auto_cols if not _is_elaboration_header(c)]
    if reason_col and reason_col in auto_cols:
        auto_cols.remove(reason_col)

    show_all = st.checkbox(
        "Show all columns (advanced)",
        value=False,
        key=f"show_all_general_{who}"
    )

    options = list(df.columns) if show_all else auto_cols
    if not options:
        options = list(df.columns)

    chosen_cols = st.multiselect(
        "Open-ended columns to analyze (general)",
        options=options,
        default=auto_cols if (not show_all and auto_cols) else [],
        key=f"open_cols_general_{who}"
    )

    if not chosen_cols:
        st.info("Select at least one open-ended column.")
        return

    row_texts = (
        df[chosen_cols]
        .fillna("")
        .astype(str)
        .apply(lambda r: " | ".join([_clean_text(v) for v in r.values if str(v).strip()]), axis=1)
        .tolist()
    )
    row_texts = [t for t in row_texts if t.strip()]

    scores_all = [_sentiment_score(t) for t in row_texts]
    labels_all = [_sentiment_label(s) for s in scores_all]
    overall = (
        pd.Series(labels_all)
        .value_counts()
        .reindex(["Negative", "Neutral", "Positive"])
        .fillna(0)
        .astype(int)
    )

    s1, s2, s3 = st.columns(3)
    s1.metric("Negative", int(overall["Negative"]))
    s2.metric("Neutral", int(overall["Neutral"]))
    s3.metric("Positive", int(overall["Positive"]))

    divider()

    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        section_title("Recurring topics (with sentiment)")
        df_topics = _topic_summary_with_sentiment(row_texts)
        st.dataframe(df_topics, use_container_width=True, hide_index=True)
        _download_df_csv(df_topics, "recurring_topics_sentiment.csv", key="dl_topics_sentiment")

    with col2:
        section_title("Top keywords")
        df_keywords = _top_keywords(row_texts, top_n=25)
        st.dataframe(df_keywords, use_container_width=True, hide_index=True)
        _download_df_csv(df_keywords, "top_keywords.csv", key="dl_top_keywords")


# ============================================================
# SECTION: KPIs
# ============================================================
if section == "KPIs":

    section_title("KPIs")
        
    divider()

    with st.expander("Data Source & Filters", expanded=not st.session_state.clean_ready):

        st.markdown(
            """
            <div style="padding: 12px 14px; border: 1px solid #e5e7eb; border-radius: 10px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
              <div style="font-weight: 700; font-size: 15px; margin-bottom: 6px;">Data source</div>
              <div style="color: #475467;">Pick a source for KPI calculations. Using the curated Data folder keeps years and names aligned automatically.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        source_mode = st.radio(
            "Choose source",
            options=["Data folder (recommended)", "Manual upload"],
            horizontal=True,
            key="kpi_source_mode",
        )
        use_data = source_mode.startswith("Data folder")
        st.session_state.use_data_kpis = use_data

        base_path = Path(os.getcwd())
        years, _ = _available_years_from_data(base_path)

        # Initialize file upload variables
        emp_file = None
        mgr_file = None

        if use_data:
            if not years:
                st.warning("No complete year files detected in Data/ (need both employee + manager files with year in name).")
        else:
            col_left, col_right = st.columns(2)
            with col_left:
                emp_file = st.file_uploader(
                    "Employee Check-In File (.xlsx/.xls/.csv)",
                    type=["xlsx", "xls", "csv"],
                    key="emp",
                )
            with col_right:
                mgr_file = st.file_uploader(
                    "Manager Check-In File (.xlsx/.xls/.csv)",
                    type=["xlsx", "xls", "csv"],
                    key="mgr",
                )

        mena_df_preview = _load_static_mena()
        if mena_df_preview.empty:
            st.warning("No Mena Report file detected in Data/ or root. Place one before cleaning.")

        # Filters (always visible). Populate immediately from available sources.
        year_options = ["All years"]
        dept_choices = ["All departments"]
        default_year = st.session_state.get("kpi_year_filter")

        # Data folder mode: show all detected years immediately
        if use_data and years:
            year_options = ["All years"] + list(years)
            if default_year is None or default_year not in year_options:
                default_year = years[-1]
            
            # Load raw files from Data folder to get departments (use latest year)
            try:
                _, by_year = _available_years_from_data(base_path)
                latest_year = years[-1]
                if latest_year in by_year and "emp" in by_year[latest_year] and "mgr" in by_year[latest_year]:
                    def _read(p):
                        if p.suffix.lower() == ".csv":
                            return pd.read_csv(str(p))
                        return pd.read_excel(str(p))
                    
                    emp_raw = _read(by_year[latest_year]["emp"])
                    mgr_raw = _read(by_year[latest_year]["mgr"])
                    dept_choices = _dept_options(emp_raw, mgr_raw)
            except Exception:
                pass  # Keep default if loading fails
        
        # Manual upload mode: populate when files are uploaded
        elif not use_data and emp_file and mgr_file:
            try:
                emp_raw = _load_df(emp_file)
                mgr_raw = _load_df(mgr_file)
                
                # Extract years
                emp_with_year = _add_year_from_timestamp(emp_raw)
                mgr_with_year = _add_year_from_timestamp(mgr_raw)
                
                emp_years = sorted([int(y) for y in emp_with_year["Year"].dropna().unique()]) if "Year" in emp_with_year.columns else []
                mgr_years = sorted([int(y) for y in mgr_with_year["Year"].dropna().unique()]) if "Year" in mgr_with_year.columns else []
                detected_years = sorted(set(emp_years) | set(mgr_years))
                if detected_years:
                    year_options = ["All years"] + detected_years
                
                # Extract departments
                dept_choices = _dept_options(emp_raw, mgr_raw)
            except Exception:
                pass  # Keep defaults if loading fails
        
        # Override with cleaned data if available
        if st.session_state.clean_ready and st.session_state.cleaned_emp is not None and st.session_state.cleaned_mgr is not None:
            emp_for_filters = st.session_state.cleaned_emp.copy()
            mgr_for_filters = st.session_state.cleaned_mgr.copy()
            
            if "Year" not in emp_for_filters.columns:
                emp_for_filters = _add_year_from_timestamp(emp_for_filters)
            if "Year" not in mgr_for_filters.columns:
                mgr_for_filters = _add_year_from_timestamp(mgr_for_filters)

            emp_years = sorted([int(y) for y in emp_for_filters["Year"].dropna().unique()]) if "Year" in emp_for_filters.columns else []
            mgr_years = sorted([int(y) for y in mgr_for_filters["Year"].dropna().unique()]) if "Year" in mgr_for_filters.columns else []
            detected_years = sorted(set(emp_years) | set(mgr_years))
            if detected_years:
                year_options = ["All years"] + detected_years

            dept_choices = _dept_options(emp_for_filters, mgr_for_filters)

        # Resolve indices for selects
        if default_year in year_options:
            year_index = year_options.index(default_year)
        else:
            year_index = 0

        c_year, c_dept = st.columns(2)
        with c_year:
            st.selectbox(
                "Filter by year",
                options=year_options,
                index=year_index,
                key="kpi_year_filter",
            )
        with c_dept:
            st.selectbox(
                "Filter by department",
                options=dept_choices,
                index=0,
                key="kpi_dept_filter",
            )

        # Show status after cleaning
        if st.session_state.clean_ready and st.session_state.cleaned_emp is not None and st.session_state.cleaned_mgr is not None:
            emp_dept_col = "Company Name / Department" if "Company Name / Department" in st.session_state.cleaned_emp.columns else _find_dept_col(st.session_state.cleaned_emp)
            mgr_dept_col = "Company Name/Department" if "Company Name/Department" in st.session_state.cleaned_mgr.columns else _find_dept_col(st.session_state.cleaned_mgr)
            st.caption(f"📊 Filters active: {len(year_options)-1} year(s), {len(dept_choices)-1} department(s) | Emp dept col: {emp_dept_col or 'Not found'}, Mgr dept col: {mgr_dept_col or 'Not found'}")

        
    run_btn = st.button("Run", type="primary", use_container_width=True)

    if run_btn:
        st.session_state.clean_error = None
        st.session_state.cleaned_emp = None
        st.session_state.cleaned_mgr = None
        st.session_state.combined = None
        st.session_state.clean_ready = False

        cleaned_emp = pd.DataFrame()
        cleaned_mgr = pd.DataFrame()
        combined = pd.DataFrame()


        if use_data:
            # Choose year to load: prefer selected filter year if valid; else latest detected year.
            selected_year_data = st.session_state.get("kpi_year_filter")
            if selected_year_data == "All years" or selected_year_data not in years:
                selected_year_data = years[-1] if years else None

            cleaned_emp, cleaned_mgr, combined = _load_and_clean_year_from_data(selected_year_data)

            if cleaned_emp.empty or cleaned_mgr.empty:
                st.session_state.clean_error = f"Could not load/clean files for year {selected_year_data} from Data/."
            else:
                st.session_state.cleaned_emp = cleaned_emp
                st.session_state.cleaned_mgr = cleaned_mgr
                st.session_state.combined = combined
                st.session_state.clean_ready = True

        else:
            # upload mode
            if not (emp_file and mgr_file):
                st.session_state.clean_error = "Upload both Employee and Manager check-in files."
            else:
                df_mena = _load_static_mena()
                if df_mena.empty:
                    st.session_state.clean_error = "Static Mena Report file not found. Place 'Mena Report.xlsx' in Data/ or root."
                else:
                    df_emp = _load_df(emp_file)
                    df_mgr = _load_df(mgr_file)

                    if df_emp.empty or df_mgr.empty:
                        st.session_state.clean_error = "One of the uploaded files appears empty or unreadable."
                    else:
                        cleaner = CheckInExcelCleaner()
                        cleaned_emp = cleaner.clean_employee_checkin(df_emp, df_mena)
                        cleaned_mgr = cleaner.clean_manager_checkin(df_mgr, df_mena)
                        combined = cleaner.combine_cleaned(cleaned_emp, cleaned_mgr)

                        st.session_state.cleaned_emp = cleaned_emp
                        st.session_state.cleaned_mgr = cleaned_mgr
                        st.session_state.combined = combined
                        st.session_state.clean_ready = True

    # -------------------------
    # Feedback after clicking Run Cleaning
    # -------------------------
    if st.session_state.clean_error:
        st.error(st.session_state.clean_error)

    if st.session_state.clean_ready and st.session_state.cleaned_emp is not None and st.session_state.cleaned_mgr is not None:
        st.caption(f"Employee rows: {len(st.session_state.cleaned_emp)} | Manager rows: {len(st.session_state.cleaned_mgr)}")

        cleaned_emp = _add_year_from_timestamp(st.session_state.cleaned_emp)
        cleaned_mgr = _add_year_from_timestamp(st.session_state.cleaned_mgr)

        # Persist updated data with Year column
        st.session_state.cleaned_emp = cleaned_emp
        st.session_state.cleaned_mgr = cleaned_mgr

        # Pull selected filters (set in Data Source & Filters expander)
        selected_year = st.session_state.get("kpi_year_filter", "All years")
        selected_dept = st.session_state.get("kpi_dept_filter", "All departments")

        # Apply Year filter first
        def _filter_by_year(df: pd.DataFrame, year_sel):
            if df is None or df.empty or year_sel == "All years" or "Year" not in df.columns:
                return df
            return df[df["Year"] == year_sel].copy()

        emp_tmp = _filter_by_year(cleaned_emp, selected_year)
        mgr_tmp = _filter_by_year(cleaned_mgr, selected_year)

        # Then apply Department filter
        emp_dept_col = "Company Name / Department" if "Company Name / Department" in emp_tmp.columns else _find_dept_col(emp_tmp)
        mgr_dept_col = "Company Name/Department" if "Company Name/Department" in mgr_tmp.columns else _find_dept_col(mgr_tmp)

        emp_f = _filter_by_dept(emp_tmp, emp_dept_col, selected_dept)
        mgr_f = _filter_by_dept(mgr_tmp, mgr_dept_col, selected_dept)

        divider()
        section_title("Completion Rate (manual inputs)")

        # Make keys unique per year+dept view so it doesn't mix inputs
        ctx = f"{selected_year}_{selected_dept}"
        ctx_key = re.sub(r"[^a-zA-Z0-9_]", "_", str(ctx))


        cA, cB = st.columns(2)

        with cA:
            st.subheader("Managers")
            sent_mgr = st.number_input("Sent to Managers", min_value=0, step=1, key=f"sent_mgr_{ctx_key}")
            rec_mgr  = st.number_input("Received from Managers", min_value=0, step=1, key=f"rec_mgr_{ctx_key}")

        with cB:
            st.subheader("Employees")
            sent_emp = st.number_input("Sent to Employees", min_value=0, step=1, key=f"sent_emp_{ctx_key}")
            rec_emp  = st.number_input("Received from Employees", min_value=0, step=1, key=f"rec_emp_{ctx_key}")

        mgr = _completion_stats(sent_mgr, rec_mgr)
        emp = _completion_stats(sent_emp, rec_emp)

        divider()

        left, right = st.columns(2)

        with left:
            st.markdown(
                f"**The organization has received a {mgr['completion']:.0f}% completion rate from Managers.**"
            )
            st.markdown(
                f"**{mgr['sent']} FORMS**"
            )
            k1, k2 = st.columns(2)
            k1.metric("Closed", f"{mgr['completion']:.0f}%", mgr["received"])
            k2.metric("Pending", f"{mgr['pending_pct']:.0f}%", mgr["pending"])

        with right:
            st.markdown(
                f"**The organization has received a {emp['completion']:.0f}% completion rate from Employees.**"
            )
            st.markdown(
                f"**{emp['sent']} FORMS**"
            )
            k1, k2 = st.columns(2)
            k1.metric("Closed", f"{emp['completion']:.0f}%", emp["received"])
            k2.metric("Pending", f"{emp['pending_pct']:.0f}%", emp["pending"])


        # --- KPIs on filtered data ---
        m = _metrics(emp_f, mgr_f)

        # Optional view switcher
        view_choice = st.radio(
            "View",
            options=["Employee", "Manager", "Employee & Manager"],
            index=0,
            horizontal=True,
            key="kpi_view_choice",
        )


        # --- Employee KPIs ---
        if view_choice == "Employee":
            divider()
            section_title("Employee KPIs")

            # ✅ Additional Employee KPIs MUST be inside here
            divider()
            with st.expander("Additional Employee KPIs (keyword-based)", expanded=False):

                # 1st KPI: Alignment on Department Goals
                section_title("Alignment on Department Goals")

                aligned_col = _find_aligned_goals_col(emp_f)
                elab_col = _find_if_no_elaborate_col(emp_f)

                if not aligned_col:
                    st.info("Could not detect the alignment question.")
                else:
                    yes_count, no_count, yes_pct, no_pct = _count_yes_no(emp_f, aligned_col)

                    c1, c2 = st.columns(2)
                    c1.metric("Yes (Aligned)", yes_count, f"{yes_pct:.1f}%")
                    c2.metric("No (Not aligned)", no_count, f"{no_pct:.1f}%")

                    if no_count > 0 and elab_col:
                        no_mask = _no_mask(emp_f[aligned_col])
                        reasons = emp_f.loc[no_mask, elab_col].dropna().astype(str).map(_clean_text)
                        reasons = [r for r in reasons.tolist() if r.strip()]
                        if reasons:
                            df_kw = _top_keywords(reasons, top_n=15)
                            st.dataframe(df_kw, use_container_width=True, hide_index=True)
                            _download_df_csv(df_kw, "alignment_no_reasons_top_keywords.csv", key="dl_align_no_keywords")

                divider()

                # Remaining KPIs in specified order
                cols = st.columns(4)
                
                # 2nd: Discussed professional goals
                rate, col = _yes_rate_from_keywords(emp_f, ["discuss", "professional", "goals"])
                cols[0].metric("Discussed professional goals", _pct(rate) if col else "N/A")
                
                # 3rd: Job requirements changed
                rate, col = _yes_rate_from_keywords(emp_f, ["encountered", "changes", "job", "requirements"])
                cols[1].metric("Job requirements changed", _pct(rate) if col else "N/A")
                
                # 4th: Adapted well (4–5)
                rate, col = _scale_good_rate_from_keywords(emp_f, ["able", "adapt", "job", "requirements"], good_min=4)
                cols[2].metric("Adapted well (4–5)", _pct(rate) if col else "N/A")
                
                # 5th: Tasks aligned with growth
                rate, col = _yes_rate_from_keywords(emp_f, ["tasks", "aligned", "growth"])
                cols[3].metric("Tasks aligned with growth", _pct(rate) if col else "N/A")
                
                cols = st.columns(4)
                
                # 6th: Manager considers input
                rate, col = _yes_rate_from_keywords(emp_f, ["seek", "consider", "input"])
                cols[0].metric("Manager considers input", _pct(rate) if col else "N/A")
                
                # 7th: HR pulse request rate
                cols[1].metric("HR pulse request rate", _pct(m["hr_pulse"]))
                
                # 8th: Recommend company
                rate, col = _yes_rate_from_keywords(emp_f, ["recommend", "company"])
                cols[2].metric("Recommend company", _pct(rate) if col else "N/A")
                
                # Employee Stress (additional metric, not in the ordered list)
                cols[3].metric("Stress rate (Employees)", _pct(m["emp_stress"]))


            divider()
            section_title("Managers Behaviors (Slide 12)")

            with st.expander("Managers Behaviors", expanded=True):
                # This slide uses ONLY employee check-in data
                q_col = _find_mgr_behavior_question_col_emp(emp_f)
                mgr_name_col = _find_manager_name_col_emp(emp_f)

                if not q_col:
                    st.info("Could not detect the 'true statements about your manager' question column in employee data.")
                elif not mgr_name_col:
                    st.info("Could not detect the Manager Name column in employee data (needed to group by manager).")
                else:
                    tmp = emp_f[[mgr_name_col, q_col]].copy()

                    def has_pos_neg(cell: str) -> tuple[bool, bool]:
                        opts = _split_multiselect(cell)
                        tags = [_norm_behavior_opt(o) for o in opts]
                        pos = any(t in POS_TAGS for t in tags)
                        neg = any(t in NEG_TAGS for t in tags)
                        return pos, neg

                    flags = tmp[q_col].astype(str).apply(has_pos_neg)
                    tmp["has_pos"] = flags.apply(lambda x: x[0])
                    tmp["has_neg"] = flags.apply(lambda x: x[1])

                    g = tmp.groupby(mgr_name_col).agg(
                        pos_any=("has_pos", "max"),
                        neg_any=("has_neg", "max"),
                        team_size=(mgr_name_col, "size"),
                    ).reset_index().rename(columns={mgr_name_col: "Manager"})

                    g["Category"] = "In Between"
                    g.loc[g["pos_any"] & ~g["neg_any"], "Category"] = "Positive only"
                    g.loc[~g["pos_any"] & g["neg_any"], "Category"] = "Negative only"

                    total_mgr = len(g) if len(g) else 1
                    left_pct  = round((g["Category"].eq("Positive only").sum() / total_mgr) * 100)
                    mid_pct   = round((g["Category"].eq("In Between").sum() / total_mgr) * 100)
                    right_pct = round((g["Category"].eq("Negative only").sum() / total_mgr) * 100)

                    # Simple Venn-like visual
                    fig, ax = plt.subplots(figsize=(10.5, 4.8))
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 6)
                    ax.axis("off")

                    ax.add_patch(Circle((4, 3), 2.2, alpha=0.35))
                    ax.add_patch(Circle((6, 3), 2.2, alpha=0.35))

                    ax.text(3.2, 3.0, f"{left_pct}%", fontsize=18, fontweight="bold")
                    ax.text(4.9, 3.0, f"{mid_pct}%", fontsize=16, fontweight="bold")
                    ax.text(6.7, 3.0, f"{right_pct}%", fontsize=18, fontweight="bold")

                    ax.text(1.0, 0.8,
                            "Managers who\n"
                            "• Provide clear guidance\n"
                            "• Provide continuous feedback\n"
                            "• Provide recognition\n"
                            "• Offer opportunities to grow",
                            fontsize=11, va="top")

                    ax.text(7.5, 0.8,
                            "Managers who\n"
                            "• Provide insufficient information\n"
                            "• Provide insufficient feedback\n"
                            "• Rarely give credit\n"
                            "• Rarely give chances to advance",
                            fontsize=11, va="top")

                    st.pyplot(fig, clear_figure=False)
                    _download_fig_png(fig, "slide12_managers_behaviors.png", key="dl_slide12_behaviors_png")


            divider()
            section_title("Work Culture and Environment — Supportive Work Environment (Slide 20)")

            # Robust detection for the long header
            env_col = None
            for c in emp_f.columns:
                h = _norm(c)
                if ("foster" in h) and ("collabor" in h) and ("support" in h) and ("work environment" in h):
                    env_col = c
                    break

            # Fallback (less strict)
            if env_col is None:
                for c in emp_f.columns:
                    h = _norm(c)
                    if ("collabor" in h) and ("support" in h) and ("environment" in h):
                        env_col = c
                        break

            if not env_col:
                st.info("Could not detect the supportive work environment question in employee data.")
            else:
                yes_mask = _yes_mask(emp_f[env_col])
                no_mask  = _no_mask(emp_f[env_col])

                yes_count = int(yes_mask.sum())
                no_count  = int(no_mask.sum())

                env_summary = pd.DataFrame([
                {"Response": "YES", "Employees": yes_count},
                {"Response": "NO",  "Employees": no_count},
                ])
                st.dataframe(env_summary, use_container_width=True, hide_index=True)
                _download_df_csv(env_summary, "supportive_work_environment_summary.csv", key="dl_supportive_env_summary")


                # Names of employees who answered NO
                if no_count > 0:
                    name_col = "Your Name"
                    if name_col in emp_f.columns:
                        no_names = emp_f.loc[no_mask, name_col].astype(str).fillna("").str.strip()
                        no_names = [n for n in no_names.tolist() if n]
                        if no_names:
                            st.caption(", ".join(no_names))
                    else:
                        st.caption("Employee name column 'Your Name' not found to list NO responders.")

            divider()
            section_title("Enhancing Department Dynamics — Employees (Slide 21)")

            dyn_col = _find_dynamics_col_emp(emp_f)
            if not dyn_col:
                st.info("Could not detect the 'department dynamics need enhancement' multi-select question in employee data.")
            else:
                counts = _dynamics_counts(emp_f[dyn_col])

                # Horizontal bar chart like PPT
                labels = DYNAMICS_ORDER
                values = [int(counts.get(k, 0)) for k in labels]
                y = np.arange(len(labels))

                fig, ax = plt.subplots(figsize=(10.5, 4.8))
                ax.barh(y, values)
                ax.set_yticks(y)
                ax.set_yticklabels(labels)
                ax.invert_yaxis()  # top item on top like slide
                ax.set_xlabel("Count")

                # Value labels at end of bars
                for i, v in enumerate(values):
                    ax.text(v + 0.1, i, str(int(v)), va="center")

                st.pyplot(fig, clear_figure=False)
                _download_fig_png(fig, "slide21_employee_dynamics.png", key="dl_slide21_emp_dyn_png")


            divider()
            with st.expander("Stress — Employees (Slide 23)", expanded=True):
                tab1, tab2 = st.tabs(["Frequency", "Reasons (Who mentioned what)"])

                # -------------------------
                # Tab 1: Stress Frequency
                # -------------------------
                with tab1:
                    freq_col = _find_stress_freq_col_emp(emp_f)
                    reason_col_freq = _find_stress_reason_col_emp(emp_f)  # optional (keywords under buckets)

                    if not freq_col:
                        st.info("Could not detect the employee stress frequency question in employee data.")
                    else:
                        tmp = emp_f.copy()
                        tmp["_stress_freq"] = tmp[freq_col].astype(str).map(_norm_stress_freq)

                        counts = tmp["_stress_freq"].value_counts().reindex(STRESS_FREQ_ORDER, fill_value=0)

                        c1, c2, c3 = st.columns(3)

                        def top3_reasons(texts: list[str]) -> str:
                            if not texts:
                                return ""
                            kw = _top_keywords(texts, top_n=3)
                            if kw.empty:
                                return ""
                            return ", ".join(kw["Keyword"].tolist())

                        for col_ui, label in zip([c1, c2, c3], STRESS_FREQ_ORDER):
                            with col_ui:
                                st.subheader(label.upper())
                                st.metric("Employees", int(counts[label]))

                                if reason_col_freq and reason_col_freq in tmp.columns:
                                    texts = tmp.loc[tmp["_stress_freq"] == label, reason_col_freq].dropna().astype(str).map(_clean_text)
                                    texts = [t for t in texts.tolist() if t.strip()]
                                    if texts:
                                        st.caption(top3_reasons(texts))

                # -------------------------
                # Tab 2: Reasons table
                # -------------------------
                with tab2:
                    reason_col = _find_emp_stress_reasons_col(emp_f)
                    name_col = "Your Name"

                    if not reason_col:
                        st.info("Could not detect the stress reasons multi-select question in employee data.")
                    elif name_col not in emp_f.columns:
                        st.info("Employee name column 'Your Name' not found in employee data.")
                    else:
                        reason_to_names: dict[str, set[str]] = {k: set() for k in EMP_STRESS_REASON_ORDER}

                        for _, r in emp_f[[name_col, reason_col]].iterrows():
                            emp_name = str(r[name_col]).strip()
                            if not emp_name:
                                continue

                            picks = _split_multiselect(r[reason_col])
                            for p in picks:
                                k = _norm_emp_stress_reason(p)
                                if k:
                                    reason_to_names[k].add(emp_name)

                        rows = []
                        for reason in EMP_STRESS_REASON_ORDER:
                            names = sorted(reason_to_names.get(reason, set()))
                            if not names:
                                continue
                            rows.append({
                                "Reason": reason,
                                "Employees who mentioned it": ", ".join(names)
                            })

                        if rows:
                            df_out = pd.DataFrame(rows)
                            st.dataframe(df_out, use_container_width=True, hide_index=True)
                            _download_df_csv(df_out, "stress_reasons_who_mentioned.csv", key="dl_stress_reasons_table")
                        else:
                            st.caption("No reasons were selected.")


            divider()
            with st.expander("Pulse-Check Meeting (Slide 26)", expanded=False):
                pulse_col = _find_pulse_yn_col_emp(emp_f)
                reason_col = _find_pulse_reason_col_emp(emp_f)

                tab1, tab2 = st.tabs(["Overview", "Reasons (YES only)"])

                with tab1:
                    if not pulse_col:
                        st.info("Could not detect the Pulse-Check meeting Yes/No question in employee data.")
                    else:
                        yes_mask = _yes_mask(emp_f[pulse_col])
                        no_mask  = _no_mask(emp_f[pulse_col])

                        yes_count = int(yes_mask.sum())
                        no_count  = int(no_mask.sum())

                        left, right = st.columns(2)
                        with left:
                            st.subheader("Employees who said YES")
                            st.metric("YES", yes_count)
                        with right:
                            st.subheader("Employees who said NO")
                            st.metric("NO", no_count)

                with tab2:
                    if not pulse_col:
                        st.info("Could not detect the Pulse-Check meeting Yes/No question in employee data.")
                    elif not reason_col or reason_col not in emp_f.columns:
                        st.caption("Reason column not detected (or no YES responses).")
                    else:
                        yes_mask = _yes_mask(emp_f[pulse_col])
                        counts = _pulse_reason_counts(emp_f.loc[yes_mask, reason_col])
                        if counts.empty:
                            st.caption("Reason column not detected (or no YES responses).")
                        else:
                            for reason, cnt in counts.items():
                                if int(cnt) > 0:
                                    st.write(f"• {reason}: **{int(cnt)}**")

            divider()
            section_title("Company Resources (Slide 27)")

            res_col = _find_company_resources_col_emp(emp_f)
            other_text_col = _find_resources_other_text_col_emp(emp_f)
            name_col = "Your Name"

            if not res_col:
                st.info("Could not detect the Company Resources question column in employee data.")
            elif name_col not in emp_f.columns:
                st.info("Employee name column 'Your Name' not found in employee data.")
            else:
                counts = {k: 0 for k in RES_ORDER}
                other_entries = []

                for _, r in emp_f.iterrows():
                    emp_name = str(r.get(name_col, "")).strip()
                    picks = _split_multiselect(r.get(res_col, ""))  # your existing helper

                    norm_picks = []
                    for p in picks:
                        k = _norm_resource_opt(p)
                        if k:
                            norm_picks.append(k)

                    # count
                    for k in norm_picks:
                        counts[k] += 1

                    # Other: Name + text (if the other text column exists)
                    if "Other" in norm_picks and emp_name and other_text_col and other_text_col in emp_f.columns:
                        txt = str(r.get(other_text_col, "")).strip()
                        if txt:
                            other_entries.append(f'{emp_name}: "{txt}"')

                # Display counts as a table
                rows = []
                label_map = {
                    "Enrolling in the group's wellness sessions": "Enrolling in the group's wellness sessions",
                    "Seeking advice from HR": "Seeking advice from HR",
                    "Expressing complaints or miscellaneous ideas": "Expressing complaints or miscellaneous ideas",
                    "Other": "Other",
                    "Not Applicable": "Not Applicable",
                }
                for key in RES_ORDER:
                    if key in counts:
                        rows.append({"Company Resource": label_map.get(key, key), "Count": counts[key]})

                if rows:
                    df_resources = pd.DataFrame(rows)
                    st.dataframe(df_resources, use_container_width=True, hide_index=True)
                else:
                    st.caption("No responses found for Company Resources.")

                # Other section
                if other_entries:
                    divider()
                    st.subheader("Other")
                    for x in other_entries:
                        st.write(x)

                

        # --- Manager KPIs ---
        if view_choice == "Manager":
            divider()
            section_title("Manager KPIs")

            # --- Additional Manager KPIs (keyword-based) - FIRST ---
            with st.expander("Additional Manager KPIs (Yes %)", expanded=True):
                st.caption("Showing percentage of employees for whom managers answered YES to each question")
                
                cols = st.columns(4)
                
                # 1. Encountered changes in job requirements
                rate, col = _yes_rate_from_keywords(mgr_f, ["encountered", "changes", "job", "requirements"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                cols[0].metric(
                    "Encountered job changes",
                    f"{yes_count} ({yes_pct:.1f}%)" if col else "N/A",
                    help=f"Column: {col}" if col else "Column not detected"
                )
                
                # 2. At risk of low performance
                rate, col = _yes_rate_from_keywords(mgr_f, ["at risk", "low performance"])
                if not col:
                    rate, col = _yes_rate_from_keywords(mgr_f, ["risk", "performance"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                cols[1].metric(
                    "At risk of low performance",
                    f"{yes_count} ({yes_pct:.1f}%)" if col else "N/A",
                    help=f"Column: {col}" if col else "Column not detected"
                )
                
                # 3. Would perform better in another department
                rate, col = _yes_rate_from_keywords(mgr_f, ["perform better", "another department"])
                if not col:
                    rate, col = _yes_rate_from_keywords(mgr_f, ["better", "department"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                cols[2].metric(
                    "Better in another dept",
                    f"{yes_count} ({yes_pct:.1f}%)" if col else "N/A",
                    help=f"Column: {col}" if col else "Column not detected"
                )
                
                # 4. Ready for promotion today
                rate, col = _yes_rate_from_keywords(mgr_f, ["ready", "promotion"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                cols[3].metric(
                    "Ready for promotion",
                    f"{yes_count} ({yes_pct:.1f}%)" if col else "N/A",
                    help=f"Column: {col}" if col else "Column not detected"
                )
                
                # Second row
                cols2 = st.columns(4)
                
                # 5. Actively seek and consider team members' input
                rate, col = _yes_rate_from_keywords(mgr_f, ["actively seek", "team members", "input"])
                if not col:
                    rate, col = _yes_rate_from_keywords(mgr_f, ["seek", "consider", "input"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                cols2[0].metric(
                    "Seeks team input",
                    f"{yes_count} ({yes_pct:.1f}%)" if col else "N/A",
                    help=f"Column: {col}" if col else "Column not detected"
                )
                
                # 6. Fits in company culture
                rate, col = _yes_rate_from_keywords(mgr_f, ["fits", "company culture"])
                if not col:
                    rate, col = _yes_rate_from_keywords(mgr_f, ["person", "fits", "culture"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                cols2[1].metric(
                    "Fits in company culture",
                    f"{yes_count} ({yes_pct:.1f}%)" if col else "N/A",
                    help=f"Column: {col}" if col else "Column not detected"
                )

            divider()

            # Consolidated table matching spec: At Risk Yes, Reason, PIP, Reason if no
            with st.expander("At Risk of Low Performance", expanded=True):
                kpi = KPIService()
                risk_h = kpi.detect_manager_risk_header(mgr_f)
                name_col = _find_subordinate_name_col(mgr_f)
                pip_h = kpi.detect_pip_header(mgr_f)
                reason_col = _find_risk_reason_col(mgr_f)
                pip_reason_col = _find_pip_reason_no_col(mgr_f)

                # Quick detections info
                st.caption(
                    f"Detected — Risk: {risk_h or 'None'} • Name: {name_col or 'None'} • PIP: {pip_h or 'None'}"
                )

                if not risk_h or not name_col:
                    st.info("Could not detect the at-risk column or Subordinate Name in manager data.")
                else:
                    b = kpi.normalize_bool_series(mgr_f[risk_h])
                    flagged = mgr_f[b == True].copy()
                    if flagged.empty:
                        st.caption("No employees marked as at risk of low performance.")
                    else:
                        out = pd.DataFrame()
                        out["Employee Name"] = flagged[name_col].astype(str).fillna("").str.strip()
                        out["At Risk of Low Performance"] = "Yes"
                        # Reason
                        out["Reason"] = ""
                        if reason_col and reason_col in flagged.columns:
                            out.loc[:, "Reason"] = flagged[reason_col].fillna("").astype(str).str.strip()
                        # PIP Yes/No
                        if pip_h and pip_h in flagged.columns:
                            pip_b = kpi.normalize_bool_series(flagged[pip_h])
                            out["PIP"] = pip_b.map({True: "Yes", False: "No"}).astype(str).replace({"<NA>": "N/A"})
                        else:
                            out["PIP"] = ""
                        # Reason if no (only if PIP is No)
                        out["Reason if no"] = ""
                        if pip_reason_col and pip_reason_col in flagged.columns:
                            if "PIP" in out.columns:
                                no_mask = out["PIP"].str.lower().eq("no")
                                out.loc[no_mask, "Reason if no"] = flagged.loc[no_mask, pip_reason_col].fillna("").astype(str).str.strip()
                            else:
                                out["Reason if no"] = flagged[pip_reason_col].fillna("").astype(str).str.strip()

                        st.dataframe(out, use_container_width=True, hide_index=True)
                        st.download_button(
                            label="Download At-Risk details CSV",
                            data=out.to_csv(index=False).encode("utf-8"),
                            file_name="employee_performance_at_risk.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="dl_at_risk_table",
                        )

            with st.expander("Manager Stress about Employee", expanded=True):
                st.metric("Manager Stress rate", _pct(m["mgr_stress"]))

            # --- Manager insights — Promotion ---
            divider()
            section_title("Manager insights — Promotion")

            promo_col = _find_ready_for_promotion_col(mgr_f)
            name_col  = _find_employee_name_col(mgr_f)
            pos_col   = _find_promo_position_col(mgr_f)
            date_col  = _find_promo_review_date_col(mgr_f)
            sub_col   = _find_subordinate_name_col(mgr_f)

            if not promo_col or not sub_col:
                st.info("Could not detect promotion readiness and/or Subordinate Name column in the manager check-in.")
            else:
                # A) Ready for promotion (YES)
                yes_df = mgr_f[_yes_mask(mgr_f[promo_col])].copy()

                with st.expander("Ready for promotion (Yes)", expanded=False):
                    if yes_df.empty:
                        st.caption("No employees marked as ready for promotion (Yes).")
                    else:
                        out_yes = yes_df[[sub_col]].copy()
                        out_yes = out_yes.rename(columns={sub_col: "Employee Name"})

                        if pos_col and pos_col in yes_df.columns:
                            out_yes["Stated Position"] = yes_df[pos_col].fillna("").astype(str).str.strip()
                            out_yes.loc[out_yes["Stated Position"].eq(""), "Stated Position"] = "Not stated"
                        else:
                            out_yes["Stated Position"] = "Not stated"

                        # Filter + download for large lists
                        cfy1, cfy2 = st.columns([3, 1])
                        with cfy1:
                            q_yes = st.text_input("Filter by name", value="", key="promo_yes_filter").strip()
                        with cfy2:
                            csv_yes = out_yes.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="Download CSV",
                                data=csv_yes,
                                file_name="ready_for_promotion.csv",
                                mime="text/csv",
                                key="promo_yes_download",
                            )

                        if q_yes:
                            view_yes = out_yes[out_yes["Employee Name"].str.contains(q_yes, case=False, na=False)].reset_index(drop=True)
                        else:
                            view_yes = out_yes.reset_index(drop=True)

                        st.caption(f"Showing {len(view_yes)} of {len(out_yes)}")
                        st.dataframe(view_yes, use_container_width=True, hide_index=True)

                divider()

                # B) Not ready (NO) — Review schedule by month
                with st.expander("Not ready for promotion (No) — Suggested review dates", expanded=False):
                    if not date_col:
                        st.info("Could not find the 'suggest a date to review' column in the manager check-in.")
                    else:
                        no_df = mgr_f[_no_mask(mgr_f[promo_col])].copy()
                        # Use Subordinate Name exclusively for employee display
                        tmp = no_df[[date_col]].copy()
                        sub_vals = no_df[sub_col].astype(str).fillna("").str.strip()
                        tmp["Employee Name"] = sub_vals
                        tmp = tmp.rename(columns={date_col: "Review Date"})
                        tmp["Review Date"] = _parse_any_date(tmp["Review Date"]) 
                        tmp = tmp.sort_values("Review Date")

                        if tmp.empty:
                            st.caption("No valid review dates found for employees marked as not ready (No).")
                        else:
                            # Split the review date into two columns: Day and Month (text)
                            tmp["Review Month"] = tmp["Review Date"].dt.strftime("%B %Y")
                            # Fallback month label extracted from raw text when parsing fails
                            raw_month = _extract_month_year_label(no_df.loc[tmp.index, date_col])
                            tmp["Review Month"] = tmp["Review Month"].fillna(raw_month)
                            tmp["Review Month"] = tmp["Review Month"].fillna("Unknown")
                            tmp["Review Day"] = tmp["Review Date"].dt.day.astype("Int64")

                            # Filter + download for large lists
                            cfn1, cfn2 = st.columns([3, 1])
                            with cfn1:
                                q_no = st.text_input("Filter by name", value="", key="promo_no_filter").strip()
                            with cfn2:
                                csv_no = tmp[["Employee Name", "Review Day", "Review Month"]].to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="Download CSV",
                                    data=csv_no,
                                    file_name="not_ready_review_dates.csv",
                                    mime="text/csv",
                                    key="promo_no_download",
                                )

                            view_no = tmp
                            if q_no:
                                view_no = view_no[view_no["Employee Name"].str.contains(q_no, case=False, na=False)]

                            st.caption(f"Showing {len(view_no)} of {len(tmp)}")
                            # Summary counts to explain discrepancies (e.g., 7 No vs 5 with dates)
                            total_no = len(no_df)
                            with_dates = int(tmp["Review Date"].notna().sum())
                            missing_or_unparsed = max(total_no - with_dates, 0)
                            st.caption(f"Total 'No' responses: {total_no} • With valid dates: {with_dates} • Missing/unparsed dates: {missing_or_unparsed}")

                            # Present by month as structured expanders
                            months = view_no["Review Month"].unique().tolist()
                            months = [m for m in months if m and m != "Unknown"]
                            for mth in months:
                                g = view_no[view_no["Review Month"] == mth].copy()
                                with st.expander(f"{mth} — {len(g)} names", expanded=False):
                                    g_disp = g[["Employee Name", "Review Day", "Review Month"]].copy()
                                    st.dataframe(g_disp.reset_index(drop=True), use_container_width=True, hide_index=True)

                            # Note unknown month entries without rendering a separate section
                            unknown_count = int((view_no["Review Month"].fillna("") == "Unknown").sum())
                            if unknown_count > 0:
                                st.caption(f"Entries without a recognizable month: {unknown_count} (see Diagnostics below).")

                            # Diagnostics: Entries with missing or unparsed review dates
                            with st.expander("Diagnostics: missing or unparsed review dates", expanded=False):
                                diag = no_df[[date_col]].copy()
                                # Names: Subordinate only
                                diag_names = no_df[sub_col].astype(str).fillna("").str.strip()
                                diag["Employee Name"] = diag_names
                                diag["Raw Review Date"] = no_df[date_col]
                                parsed = _parse_any_date(diag["Raw Review Date"]) 
                                mask_missing = parsed.isna() | diag["Raw Review Date"].astype(str).str.strip().eq("")
                                issues = diag[mask_missing][["Employee Name", "Raw Review Date"]].copy()
                                if issues.empty:
                                    st.caption("No missing or unparsed review dates detected among 'No' responses.")
                                else:
                                    cdx1, cdx2 = st.columns([3, 1])
                                    with cdx1:
                                        st.caption(f"Entries with missing/unparsed dates: {len(issues)}")
                                    with cdx2:
                                        st.download_button(
                                            label="Download CSV",
                                            data=issues.to_csv(index=False).encode("utf-8"),
                                            file_name="not_ready_unparsed_dates.csv",
                                            mime="text/csv",
                                            key="promo_no_diag_download",
                                        )
                                    st.dataframe(issues.reset_index(drop=True), use_container_width=True, hide_index=True)

                            # Quick helper: find a specific name and show raw/parsed date
                            with st.expander("Find a specific name", expanded=False):
                                q_find = st.text_input("Name contains", value="", key="promo_no_find").strip()
                                if q_find:
                                    recon = no_df.copy()
                                    # Names: Subordinate only
                                    recon["Employee Name"] = recon[sub_col].astype(str).fillna("").str.strip()
                                    recon["Raw Review Date"] = recon[date_col]
                                    recon["Parsed Date"] = _parse_any_date(recon["Raw Review Date"]) 
                                    recon["Parsed Month"] = recon["Parsed Date"].dt.strftime("%B %Y")
                                    # fallback month label from raw if needed
                                    recon_raw_month = _extract_month_year_label(recon["Raw Review Date"])
                                    recon["Parsed Month"] = recon["Parsed Month"].fillna(recon_raw_month)

                                    filt = recon["Employee Name"].str.contains(q_find, case=False, na=False)
                                    show = recon.loc[filt, ["Employee Name", "Raw Review Date", "Parsed Date", "Parsed Month"]]
                                    if show.empty:
                                        st.info("No matching names found among 'No' responses.")
                                    else:
                                        st.dataframe(show.reset_index(drop=True), use_container_width=True, hide_index=True)

            
            
            divider()
            section_title("Employee – Company Culture Fit (Manager)")

            # Detect the Yes/No column
            culture_col = _find_col_by_keywords(mgr_f, ["fits", "company", "culture"]) \
                        or _find_col_by_keywords(mgr_f, ["person", "fits", "culture"])

            if not culture_col:
                st.info("Could not detect the culture-fit Yes/No question in manager data.")
            else:
                yes = _yes_mask(mgr_f[culture_col])
                no  = _no_mask(mgr_f[culture_col])

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Employees fit in the company’s culture", int(yes.sum()))
                with c2:
                    st.metric("Employees don’t fit in the company’s culture", int(no.sum()))


            divider()
            section_title("Employee Contribution to the Department Dynamics — Managers (Slide)")

            dyn_col = _find_mgr_dynamics_col_mgr(mgr_f)
            if not dyn_col:
                st.info("Could not detect the 'member contributes to enhance department dynamics' multi-select question in manager data.")
            else:
                counts = _mgr_dynamics_counts(mgr_f[dyn_col])

                # Horizontal bar chart like PPT
                labels = MGR_DYNAMICS_ORDER
                values = [int(counts.get(k, 0)) for k in labels]
                y = np.arange(len(labels))

                fig, ax = plt.subplots(figsize=(10.5, 4.8))
                ax.barh(y, values)
                ax.set_yticks(y)
                ax.set_yticklabels(labels)
                ax.invert_yaxis()
                ax.set_xlabel("Count")

                # value labels at end
                for i, v in enumerate(values):
                    ax.text(v + 0.1, i, str(int(v)), va="center")

                st.pyplot(fig, clear_figure=False)
                _download_fig_png(fig, "slide22_manager_dynamics.png", key="dl_slide22_mgr_dyn_png")


            divider()
            section_title("Stress Frequency — Managers (Slide 24)")

            freq_col = _find_mgr_stress_freq_col(mgr_f)
            reason_col = _find_mgr_stress_reasons_col(mgr_f)

            if not freq_col:
                st.info("Could not detect the manager stress frequency question in manager data.")
            else:
                tmp = mgr_f.copy()
                tmp["_stress_freq"] = tmp[freq_col].astype(str).map(_norm_stress_freq)  # reuse your employee normalizer

                # Counts per bucket
                counts = tmp["_stress_freq"].value_counts().reindex(STRESS_FREQ_ORDER, fill_value=0)

                # UI like slide (3 columns)
                c1, c2, c3 = st.columns(3)

                def top3_from_reason_series(s: pd.Series) -> list[str]:
                    if not reason_col or reason_col not in tmp.columns:
                        return []
                    vc = _stress_reason_counts(s)
                    top = vc[vc > 0].sort_values(ascending=False).head(3)
                    return top.index.tolist()

                for col_ui, label in zip([c1, c2, c3], STRESS_FREQ_ORDER):
                    with col_ui:
                        st.subheader(label.upper())
                        st.metric("Employees", int(counts[label]))

                        # Top reasons for this frequency bucket
                        if reason_col and reason_col in tmp.columns:
                            reasons_series = tmp.loc[tmp["_stress_freq"] == label, reason_col]
                            top3 = top3_from_reason_series(reasons_series)

                            if top3:
                                st.caption("\n".join(top3))
                            else:
                                st.caption("")
                        else:
                            st.caption("Reasons column not detected.")

                    

        elif view_choice == "Employee & Manager":

            def _find_adapt_scale_col_emp(df: pd.DataFrame) -> str | None:
                    # "During the year 2025, I was able to adapt to changes in my job requirements"
                    return _find_col_by_keywords(df, ["able", "adapt", "changes", "job", "requirements"])

            def _find_adapt_scale_col_mgr(df: pd.DataFrame) -> str | None:
                # "This person was directly able to adjust to changing job requirements"
                return (
                    _find_col_by_keywords(df, ["this", "person", "able", "adjust", "job", "requirements"]) or
                    _find_col_by_keywords(df, ["able", "adjust", "job", "requirements"]) or
                    _find_col_by_keywords(df, ["able", "adapt", "job", "requirements"])
                )

            def _find_adapt_reason_col(df: pd.DataFrame) -> str | None:
                # "If answered 1 or 2, please further elaborate..."
                # We use strong keywords to avoid picking random elaborations.
                return (
                    _find_col_by_keywords(df, ["if", "answered", "1", "2", "elaborate"]) or
                    _find_col_by_keywords(df, ["answered", "1", "2", "reasons"]) or
                    _find_col_by_keywords(df, ["not", "able", "adjust", "reasons"])
                )

            def _low_1_2_mask(series: pd.Series) -> pd.Series:
                s = pd.to_numeric(series, errors="coerce")
                return s.isin([1, 2])



            divider()
            section_title("Employee & Manager (combined view)")
            st.info("This section will be used later for cross-source comparisons.")

            divider()
            section_title("Changes in Job Responsibilities (Employee vs Manager)")

            emp_yn_col = _find_job_change_yn_col_emp(emp_f)
            mgr_yn_col = _find_job_change_yn_col_mgr(mgr_f)

            emp_types_col = _find_job_change_types_col(emp_f)
            mgr_types_col = _find_job_change_types_col(mgr_f)

            name_emp_col = _find_employee_name_col(emp_f)  # may or may not exist
            name_mgr_col = _find_subordinate_name_col(mgr_f)

            if not emp_yn_col or not mgr_yn_col:
                st.info("Could not detect the job-change Yes/No question in employee or manager data.")
            else:
                # Yes counts
                emp_yes = _yes_mask(emp_f[emp_yn_col])
                mgr_yes = _yes_mask(mgr_f[mgr_yn_col])

                left, right = st.columns(2)
                with left:
                    st.metric("Employees answered Yes", int(emp_yes.sum()))
                with right:
                    st.metric("Managers answered Yes", int(mgr_yes.sum()))

                # Multi-select breakdown for YES responders
                divider()
                section_title("Types of changes (Yes responders only) — Employee vs Manager")

                if not emp_types_col or not mgr_types_col:
                    st.info("Could not detect the change-type follow-up column in employee or manager data.")
                else:
                    emp_yes = _yes_mask(emp_f[emp_yn_col])
                    mgr_yes = _yes_mask(mgr_f[mgr_yn_col])

                    emp_counts = _change_counts(emp_f.loc[emp_yes, emp_types_col])
                    mgr_counts = _change_counts(mgr_f.loc[mgr_yes, mgr_types_col])

                    # Build plot
                    y = range(len(CHANGE_CATS))
                    fig, ax = plt.subplots(figsize=(8, 4.8))

                    ax.barh([i + 0.2 for i in y], emp_counts.values, height=0.35, label="Employees")
                    ax.barh([i - 0.2 for i in y], mgr_counts.values, height=0.35, label="Managers")

                    ax.set_yticks(list(y))
                    ax.set_yticklabels(CHANGE_CATS)
                    ax.invert_yaxis()  # top-to-bottom like slide
                    ax.set_xlabel("Count")
                    ax.legend()

                    # Add value labels at bar ends
                    for i, v in enumerate(emp_counts.values):
                        ax.text(v + 0.05, i + 0.2, str(int(v)), va="center")
                    for i, v in enumerate(mgr_counts.values):
                        ax.text(v + 0.05, i - 0.2, str(int(v)), va="center")

                    st.pyplot(fig, clear_figure=False)
                    _download_fig_png(fig, "slide11_job_changes_types.png", key="dl_slide11_job_changes_png")

                divider()
                section_title("Adapting to Change (Employee vs Manager)")

                emp_adapt_col = _find_adapt_scale_col_emp(emp_f)
                mgr_adapt_col = _find_adapt_scale_col_mgr(mgr_f)

                emp_reason_col = _find_adapt_reason_col(emp_f)
                mgr_reason_col = _find_adapt_reason_col(mgr_f)

                if not emp_adapt_col or not mgr_adapt_col:
                    st.info("Could not detect the 'adapt to change' scale question in employee or manager data.")
                else:
                    emp_low = _low_1_2_mask(emp_f[emp_adapt_col])
                    mgr_low = _low_1_2_mask(mgr_f[mgr_adapt_col])

                    emp_high = pd.to_numeric(emp_f[emp_adapt_col], errors="coerce").isin([4, 5])
                    mgr_high = pd.to_numeric(mgr_f[mgr_adapt_col], errors="coerce").isin([4, 5])

                    left, right = st.columns(2)
                    with left:
                        st.metric("Employees (4–5 = able to adapt)", int(emp_high.sum()))
                        st.caption(f"Employees (1–2 = not able): {int(emp_low.sum())}")
                    with right:
                        st.metric("Managers (rated employee 4–5)", int(mgr_high.sum()))
                        st.caption(f"Managers (rated employee 1–2): {int(mgr_low.sum())}")

                    # Slide-style bullets like your screenshot
                    bullets = []
                    if int(emp_low.sum()) == 0:
                        bullets.append("No employee mentioned they were not able to adapt to changes in their job requirements.")
                    if int(mgr_low.sum()) == 0:
                        bullets.append("No manager mentioned that their employee was not able to adjust to changes.")
                    if bullets:
                        for b in bullets:
                            st.write(f"• {b}")

                    # Reasons for low (1–2)
                    divider()
                    section_title("Reasons for not being able to adjust (only 1–2 scores)")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Employees — Reasons (1–2 only)")
                        if emp_reason_col:
                            texts = emp_f.loc[emp_low, emp_reason_col].dropna().astype(str).map(_clean_text)
                            texts = [t for t in texts if t.strip()]
                            if texts:
                                st.dataframe(_top_keywords(texts, top_n=20), use_container_width=True, hide_index=True)
                                with st.expander("Show raw reasons", expanded=False):
                                    st.dataframe(pd.DataFrame({"Reason": texts}), use_container_width=True, hide_index=True)
                            else:
                                st.caption("No reasons provided (or all empty).")
                        else:
                            st.caption("Reason column not detected in employee file.")

                    with c2:
                        st.subheader("Managers — Reasons (1–2 only)")
                        if mgr_reason_col:
                            texts = mgr_f.loc[mgr_low, mgr_reason_col].dropna().astype(str).map(_clean_text)
                            texts = [t for t in texts if t.strip()]
                            if texts:
                                st.dataframe(_top_keywords(texts, top_n=20), use_container_width=True, hide_index=True)
                                with st.expander("Show raw reasons", expanded=False):
                                    st.dataframe(pd.DataFrame({"Reason": texts}), use_container_width=True, hide_index=True)
                            else:
                                st.caption("No reasons provided (or all empty).")
                        else:
                            st.caption("Reason column not detected in manager file.")


                divider()
                section_title("Check-ins Meetings Frequency (Employee vs Manager)")

                emp_freq_col = _find_checkin_freq_col_emp(emp_f)
                mgr_freq_col = _find_checkin_freq_col_mgr(mgr_f)

                emp_name_col = _find_employee_name_col(emp_f)
                mgr_name_col = _find_subordinate_name_col(mgr_f)

                if not emp_freq_col or not mgr_freq_col:
                    st.info("Could not detect the check-in frequency question in employee or manager data.")
                else:
                    emp_freq = emp_f[emp_freq_col].astype(str).map(_norm_freq)
                    mgr_freq = mgr_f[mgr_freq_col].astype(str).map(_norm_freq)

                    def _pct_series(s: pd.Series) -> pd.Series:
                        base = s[s.isin(FREQ_CATS)]
                        if len(base) == 0:
                            return pd.Series({k: 0.0 for k in FREQ_CATS})
                        return (base.value_counts(normalize=True) * 100).reindex(FREQ_CATS, fill_value=0.0)

                    emp_pct = _pct_series(emp_freq)
                    mgr_pct = _pct_series(mgr_freq)

                    # Chart: grouped horizontal bars (Managers vs Employees) per category
    
                    y = np.arange(len(FREQ_CATS))
                    h = 0.35

                    fig, ax = plt.subplots(figsize=(9, 4.8))
                    ax.barh(y - h/2, mgr_pct.values, height=h, label="Managers’ perspective")
                    ax.barh(y + h/2, emp_pct.values, height=h, label="Employees’ perspective")
                    ax.set_yticks(y)
                    ax.set_yticklabels(FREQ_CATS)
                    ax.invert_yaxis()
                    ax.set_xlabel("% of responses")
                    ax.set_title("Check-ins meeting frequency (2025)")
                    ax.legend()
                    st.pyplot(fig, clear_figure=False)
                    _download_fig_png(fig, "checkins_meeting_frequency.png", key="dl_meeting_freq_png")

                    # Misalignments (needs names)
                    divider()
                    section_title("Misalignments")

                    if not emp_name_col or not mgr_name_col:
                        st.caption("Name column not detected in one of the datasets, so misalignments cannot be computed.")
                    else:
                        emp_map = dict(zip(emp_f[emp_name_col].astype(str).str.strip(), emp_freq))
                        mgr_map = dict(zip(mgr_f[mgr_name_col].astype(str).str.strip(), mgr_freq))

                        common = sorted(set(emp_map.keys()) & set(mgr_map.keys()))
                        rows = []
                        for nm in common:
                            e = emp_map.get(nm, "Unknown")
                            g = mgr_map.get(nm, "Unknown")
                            if e in FREQ_CATS and g in FREQ_CATS and e != g:
                                rows.append({"Employee Name": nm, "Employee": e, "Manager": g})

                        if not rows:
                            st.caption("No misalignments found among matched names.")
                        else:
                            df_mis = pd.DataFrame(rows)
                            st.warning(f"Misalignments found: {len(df_mis)}")
                            st.dataframe(df_mis, use_container_width=True, hide_index=True)

                            st.download_button(
                                "Download misalignments CSV",
                                df_mis.to_csv(index=False).encode("utf-8"),
                                file_name="checkins_frequency_misalignments.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key="dl_freq_misalign",      
                            )

                divider()
                section_title("Reward & Recognition (Employee vs Manager)")

                # Find the multi-select column in each dataset
                emp_recog_col = _find_col_by_keywords(emp_f, ["rewards", "recognizes", "contribution"]) \
                                or _find_col_by_keywords(emp_f, ["rewards", "recognition"]) \
                                or _find_col_by_keywords(emp_f, ["rewards", "recognizes"])

                mgr_recog_col = _find_col_by_keywords(mgr_f, ["recognize", "reward", "contributions"]) \
                                or _find_col_by_keywords(mgr_f, ["recognize", "reward"])

                if not emp_recog_col or not mgr_recog_col:
                    st.info("Could not detect the reward & recognition multi-select question in employee and/or manager data.")
                else:
                    emp_counts = _recog_counts(emp_f[emp_recog_col])
                    mgr_counts = _recog_counts(mgr_f[mgr_recog_col])

                    # Table (same structure as slide)
                    table_df = pd.DataFrame({
                        "Method": RECOG_ORDER,
                        "Employees’ Answers": [int(emp_counts.get(m, 0)) for m in RECOG_ORDER],
                        "Managers’ Answers":  [int(mgr_counts.get(m, 0)) for m in RECOG_ORDER],
                    })

                    st.dataframe(table_df, use_container_width=True, hide_index=True)
                    _download_df_csv(table_df, "reward_recognition_employee_vs_manager.csv", key="dl_reward_recognition_table")

                    # Top 3 bullets (employees)
                    st.markdown("**Based on employees’ feedback, the 3 most common used methods of recognition by managers are:**")
                    top_emp = table_df.sort_values("Employees’ Answers", ascending=False).head(3)
                    for m in top_emp["Method"].tolist():
                        st.write(f"• {m}")

                    divider()

                    # Top 3 bullets (managers)
                    st.markdown("**Based on managers’ feedback, the 3 most common used methods of recognition by managers are:**")
                    top_mgr = table_df.sort_values("Managers’ Answers", ascending=False).head(3)
                    for m in top_mgr["Method"].tolist():
                        st.write(f"• {m}")



                divider()
                section_title("Employee Input in the Department (Employee vs Manager)")

                emp_q = _find_input_seek_col_emp(emp_f)
                mgr_q = _find_input_seek_col_mgr(mgr_f)

                # ✅ Use employee names on BOTH sides
                emp_name_col = "Your Name"          # employee file
                mgr_name_col = "Subordinate Name"   # manager file

                if not emp_q or not mgr_q:
                    st.info("Could not detect the input-seeking Yes/No question in employee or manager data.")
                elif emp_name_col not in emp_f.columns:
                    st.info("Employee name column 'Your Name' not found in employee data.")
                elif mgr_name_col not in mgr_f.columns:
                    st.info("Employee name column 'Subordinate Name' not found in manager data.")
                else:
                    emp_yes = _yes_mask(emp_f[emp_q])
                    emp_no  = _no_mask(emp_f[emp_q])

                    mgr_yes = _yes_mask(mgr_f[mgr_q])
                    mgr_no  = _no_mask(mgr_f[mgr_q])

                    left, right = st.columns(2)

                    with left:
                        st.subheader("Employee")
                        st.metric("YES", int(emp_yes.sum()))
                        st.metric("NO", int(emp_no.sum()))

                        if int(emp_no.sum()) > 0:
                            names = emp_f.loc[emp_no, emp_name_col].astype(str).fillna("").str.strip()
                            names = [n for n in names.tolist() if n]
                            st.caption(", ".join(names) if names else "No employee names found for NO responses.")

                    with right:
                        st.subheader("Manager")
                        st.metric("YES", int(mgr_yes.sum()))
                        st.metric("NO", int(mgr_no.sum()))

                        if int(mgr_no.sum()) > 0:
                            names = mgr_f.loc[mgr_no, mgr_name_col].astype(str).fillna("").str.strip()
                            names = [n for n in names.tolist() if n]
                            st.caption(", ".join(names) if names else "No employee names found for NO responses.")


                divider()
                section_title("Ways to Address Mistakes (Employee vs Manager)")

                # Detect the multi-select columns
                emp_col = _find_col_by_keywords(emp_f, ["manager", "addresses", "mistakes"]) \
                        or _find_col_by_keywords(emp_f, ["addresses", "mistakes"])

                mgr_col = _find_col_by_keywords(mgr_f, ["address", "employee", "mistakes"]) \
                        or _find_col_by_keywords(mgr_f, ["address", "mistakes"])

                if not emp_col or not mgr_col:
                    st.info("Could not detect the 'address mistakes' multi-select question in employee and/or manager data.")
                else:
                    emp_counts = _mistake_counts(emp_f[emp_col])
                    mgr_counts = _mistake_counts(mgr_f[mgr_col])

                    # -----------------------
                    # A) Chart (like PPT)
                    # -----------------------
                    x = np.arange(len(MISTAKE_ORDER))
                    w = 0.35

                    fig, ax = plt.subplots(figsize=(11, 4.8))
                    ax.bar(x - w/2, mgr_counts.values, width=w, label="Managers")
                    ax.bar(x + w/2, emp_counts.values, width=w, label="Employees")
                    ax.set_xticks(x)
                    ax.set_xticklabels(MISTAKE_ORDER, rotation=45, ha="right")
                    ax.set_ylabel("Count")
                    ax.legend()

                    # Add value labels
                    for i, v in enumerate(mgr_counts.values):
                        ax.text(i - w/2, v + 0.1, str(int(v)), ha="center", va="bottom", fontsize=9)
                    for i, v in enumerate(emp_counts.values):
                        ax.text(i + w/2, v + 0.1, str(int(v)), ha="center", va="bottom", fontsize=9)

                    st.pyplot(fig, clear_figure=False)
                    _download_fig_png(fig, "slide17_mistakes_employee_vs_manager.png", key="dl_slide17_mistakes_png")


                    divider()

                    # -----------------------
                    # B) Managers adopting undesired approaches (like PPT left text)
                    # -----------------------
                    left, right = st.columns(2)

                    # Managers side: if they selected undesired methods
                    with left:
                        st.subheader("Managers adopting undesired approaches")
                        undesired_mgr_total = int(mgr_counts.get("Immediate Confrontation", 0) + mgr_counts.get("Blame Approach", 0))
                        st.markdown("**Managers**")
                        if undesired_mgr_total == 0:
                            st.write("• None")
                        else:
                            if int(mgr_counts.get("Immediate Confrontation", 0)) > 0:
                                st.write(f"• Immediate Confrontation: {int(mgr_counts['Immediate Confrontation'])}")
                            if int(mgr_counts.get("Blame Approach", 0)) > 0:
                                st.write(f"• Blame Approach: {int(mgr_counts['Blame Approach'])}")

                    # Employees side: list manager names mentioned by employees (mentioned by whom)
                    with right:
                        st.markdown("**Employees**")

                        # These must exist exactly as you told me in the previous slide:
                        emp_employee_name_col = "Your Name"

                        # Manager name column in employee check-in (if your cleaned file uses something else, replace it)
                        emp_manager_name_col = _find_manager_name_col_emp(emp_f)

                        if not emp_manager_name_col:
                            st.caption("Manager name column not detected in employee data, so we cannot list 'mentioned by' cases.")
                        elif emp_employee_name_col not in emp_f.columns:
                            st.caption("Employee name column 'Your Name' not found in employee data.")
                        else:
                            tmp = emp_f[[emp_employee_name_col, emp_manager_name_col, emp_col]].copy()

                            rows = []
                            for _, r in tmp.iterrows():
                                emp_name = str(r[emp_employee_name_col]).strip()
                                mgr_name = str(r[emp_manager_name_col]).strip()
                                picks = [_norm_mistake_opt(p) for p in _split_multiselect(r[emp_col])]
                                picks = [p for p in picks if p]

                                for u in UNDESIRED:
                                    if u in picks:
                                        rows.append({"Undesired": u, "Manager": mgr_name, "Mentioned by": emp_name})

                            if not rows:
                                st.write("• None")
                            else:
                                df_rows = pd.DataFrame(rows)

                                # Print like PPT: group by undesired type, then manager -> mentioned by
                                for u in ["Immediate Confrontation", "Blame Approach"]:
                                    sub = df_rows[df_rows["Undesired"] == u]
                                    if sub.empty:
                                        continue

                                    st.markdown(f"**Managers using {u}**")
                                    for mgr, g in sub.groupby("Manager"):
                                        names = ", ".join(sorted(set(g["Mentioned by"].tolist())))
                                        st.write(f"• {mgr} mentioned by {names}")

                divider()
                section_title("Employee Integration within the Team (Employee vs Manager)")

                emp_col = _find_team_integration_col_emp(emp_f)
                mgr_col = _find_team_integration_col_mgr(mgr_f)

                if not emp_col or not mgr_col:
                    st.info("Could not detect the team integration question in employee and/or manager data.")
                else:
                    emp_vals = emp_f[emp_col].astype(str).map(_norm_team_integration)
                    mgr_vals = mgr_f[mgr_col].astype(str).map(_norm_team_integration)

                    emp_total = int((emp_vals != "Unknown").sum())
                    mgr_total = int((mgr_vals != "Unknown").sum())

                    emp_well = int((emp_vals == "Well integrated with the team").sum())
                    mgr_well = int((mgr_vals == "Well integrated with the team").sum())

                    emp_pct = (emp_well / emp_total * 100.0) if emp_total else 0.0
                    mgr_pct = (mgr_well / mgr_total * 100.0) if mgr_total else 0.0

                    left, right = st.columns(2)

                    with left:
                        st.subheader("Employee")
                        st.metric("Well integrated", emp_well)
                        st.caption(f"{emp_pct:.0f}%")

                    with right:
                        st.subheader("Manager")
                        st.metric("Well integrated", mgr_well)
                        st.caption(f"{mgr_pct:.0f}%")


elif section == "Analysis":
    section_title("Analysis")
    divider()
    st.info("Analysis section is reserved for future use.")


# -------------------------
# SECTION: Compare 2 Years
# -------------------------
elif section == "Compare":
    divider()
    section_title("Year-over-Year Comparison")
    
    # Auto-detect and load check-in files from Data folder
    checkin_base = Path("Data/Check-ins 2024 2025")
    
    def _auto_load_checkin_files():
        """Auto-detect the latest two years of check-in files."""
        files_info = {}
        
        if checkin_base.exists():
            # Look for year subdirectories
            year_dirs = sorted([d for d in checkin_base.iterdir() if d.is_dir() and d.name.isdigit()], reverse=True)
            
            if len(year_dirs) >= 2:
                # Latest year (Year 2)
                y2_dir = year_dirs[0]
                y2 = int(y2_dir.name)
                
                # Previous year (Year 1)
                y1_dir = year_dirs[1]
                y1 = int(y1_dir.name)
                
                # Find employee and manager files for Year 2
                emp_y2 = None
                mgr_y2 = None
                for f in y2_dir.glob("*.xlsx"):
                    # Skip temporary Excel files (start with ~$)
                    if f.name.startswith("~$"):
                        continue
                    fname_lower = f.name.lower()
                    if "employee answers" in fname_lower or "employee questions" in fname_lower and "performance check-in" in fname_lower:
                        emp_y2 = f
                    elif "manager answers" in fname_lower or "manager questions" in fname_lower and "performance check-in" in fname_lower:
                        mgr_y2 = f
                
                # Find employee and manager files for Year 1
                emp_y1 = None
                mgr_y1 = None
                for f in y1_dir.glob("*.xlsx"):
                    # Skip temporary Excel files (start with ~$)
                    if f.name.startswith("~$"):
                        continue
                    fname_lower = f.name.lower()
                    if "employee answers" in fname_lower or "employee questions" in fname_lower and "performance check-in" in fname_lower:
                        emp_y1 = f
                    elif "manager answers" in fname_lower or "manager questions" in fname_lower and "performance check-in" in fname_lower:
                        mgr_y1 = f
                
                files_info = {
                    "y1": y1,
                    "y2": y2,
                    "emp_y1": emp_y1,
                    "mgr_y1": mgr_y1,
                    "emp_y2": emp_y2,
                    "mgr_y2": mgr_y2,
                }
        
        return files_info
    
    auto_files = _auto_load_checkin_files()
    
    # Show auto-detected files info
    if auto_files and all([auto_files.get("emp_y1"), auto_files.get("mgr_y1"), auto_files.get("emp_y2"), auto_files.get("mgr_y2")]):
        st.success(f"✅ Auto-detected files for {auto_files['y1']} vs {auto_files['y2']} comparison")
        with st.expander("📁 Auto-detected files", expanded=False):
            st.caption(f"**Year 1 ({auto_files['y1']}):**")
            st.caption(f"  • Employee: {auto_files['emp_y1'].name}")
            st.caption(f"  • Manager: {auto_files['mgr_y1'].name}")
            st.caption(f"**Year 2 ({auto_files['y2']}):**")
            st.caption(f"  • Employee: {auto_files['emp_y2'].name}")
            st.caption(f"  • Manager: {auto_files['mgr_y2'].name}")

    with st.expander("Upload Files for Comparison (Optional - overrides auto-detection)", expanded=not st.session_state.yoy_ready and not auto_files):
        st.markdown(
            """
            <div style="padding: 12px 14px; border: 1px solid #e5e7eb; border-radius: 10px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
              <div style="font-weight: 700; font-size: 15px; margin-bottom: 6px;">Manual Upload (Optional)</div>
              <div style="color: #475467;">Upload check-in files for two different years to compare KPIs and trends. Leave empty to use auto-detected files.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Year 1 (Earlier)")
            emp_file_y1 = st.file_uploader(
                "Employee Check-In (Year 1)",
                type=["xlsx", "xls", "csv"],
                key="compare_emp_y1",
            )
            mgr_file_y1 = st.file_uploader(
                "Manager Check-In (Year 1)",
                type=["xlsx", "xls", "csv"],
                key="compare_mgr_y1",
            )
        
        with col2:
            st.subheader("Year 2 (Later)")
            emp_file_y2 = st.file_uploader(
                "Employee Check-In (Year 2)",
                type=["xlsx", "xls", "csv"],
                key="compare_emp_y2",
            )
            mgr_file_y2 = st.file_uploader(
                "Manager Check-In (Year 2)",
                type=["xlsx", "xls", "csv"],
                key="compare_mgr_y2",
            )

    compare_btn = st.button("Run Comparison", type="primary", use_container_width=True)

    if compare_btn:
        st.session_state.yoy_ready = False
        st.session_state.yoy_payload = None

        df_mena = _load_static_mena()
        if df_mena.empty:
            st.error("Mena Report file not found in Data/ or root. Place 'Mena Report.xlsx' before comparing.")
        else:
            # Use auto-detected files if manual uploads are not provided
            use_emp_y1 = emp_file_y1 if emp_file_y1 else (auto_files.get("emp_y1") if auto_files else None)
            use_mgr_y1 = mgr_file_y1 if mgr_file_y1 else (auto_files.get("mgr_y1") if auto_files else None)
            use_emp_y2 = emp_file_y2 if emp_file_y2 else (auto_files.get("emp_y2") if auto_files else None)
            use_mgr_y2 = mgr_file_y2 if mgr_file_y2 else (auto_files.get("mgr_y2") if auto_files else None)
            
            if not (use_emp_y1 and use_mgr_y1 and use_emp_y2 and use_mgr_y2):
                st.error("Please upload all four files (Employee + Manager for both years) or ensure auto-detection found all files.")
            else:
                try:
                    cleaner = CheckInExcelCleaner()

                    # Load dataframes - handle both file paths and uploaded files
                    def _load_file(file_or_path):
                        if isinstance(file_or_path, Path):
                            return pd.read_excel(file_or_path) if file_or_path.suffix in ['.xlsx', '.xls'] else pd.read_csv(file_or_path)
                        else:
                            return _load_df(file_or_path)
                    
                    emp1 = _load_file(use_emp_y1)
                    mgr1 = _load_file(use_mgr_y1)
                    emp2 = _load_file(use_emp_y2)
                    mgr2 = _load_file(use_mgr_y2)

                    # Detect years from timestamp columns
                    emp1_w_year = _add_year_from_timestamp(emp1)
                    emp2_w_year = _add_year_from_timestamp(emp2)
                    
                    y1_emp = emp1_w_year["Year"].dropna().mode()[0] if "Year" in emp1_w_year.columns and not emp1_w_year["Year"].dropna().empty else "Year 1"
                    y2_emp = emp2_w_year["Year"].dropna().mode()[0] if "Year" in emp2_w_year.columns and not emp2_w_year["Year"].dropna().empty else "Year 2"
                    
                    y1 = int(y1_emp) if isinstance(y1_emp, (int, float)) else y1_emp
                    y2 = int(y2_emp) if isinstance(y2_emp, (int, float)) else y2_emp

                    st.info(f"Cleaning Year 1 ({y1})…")
                    emp1_c = cleaner.clean_employee_checkin(emp1, df_mena)
                    mgr1_c = cleaner.clean_manager_checkin(mgr1, df_mena)

                    st.info(f"Cleaning Year 2 ({y2})…")
                    emp2_c = cleaner.clean_employee_checkin(emp2, df_mena)
                    mgr2_c = cleaner.clean_manager_checkin(mgr2, df_mena)

                    m1 = _metrics(emp1_c, mgr1_c)
                    m2 = _metrics(emp2_c, mgr2_c)

                    kpi = KPIService()
                    risk_h1 = kpi.detect_manager_risk_header(mgr1_c)
                    risk_h2 = kpi.detect_manager_risk_header(mgr2_c)

                    def _at_risk_rows(mgr_df: pd.DataFrame, risk_col: str | None) -> pd.DataFrame:
                        if mgr_df is None or mgr_df.empty or not risk_col or risk_col not in mgr_df.columns:
                            return pd.DataFrame()
                        s = mgr_df[risk_col].astype(str).str.strip().str.lower()
                        mask = s.str.startswith(("yes", "y", "true", "1"))
                        return mgr_df[mask].copy()

                    risk_df_y1 = _at_risk_rows(mgr1_c, risk_h1)
                    risk_df_y2 = _at_risk_rows(mgr2_c, risk_h2)

                    st.session_state.yoy_ready = True
                    st.session_state.yoy_payload = {
                        "y1": y1, "y2": y2,
                        "m1": m1, "m2": m2,
                        "risk_y1": risk_df_y1, "risk_y2": risk_df_y2,
                        "emp1_c": emp1_c, "mgr1_c": mgr1_c,
                        "emp2_c": emp2_c, "mgr2_c": mgr2_c,
                    }
                    st.success("Comparison ready! ✅")

                except Exception as e:
                    st.error(f"Comparison failed: {e}")
                    st.exception(e)

    if st.session_state.yoy_ready and st.session_state.yoy_payload:
        y1 = st.session_state.yoy_payload["y1"]
        y2 = st.session_state.yoy_payload["y2"]
        m1 = st.session_state.yoy_payload["m1"]
        m2 = st.session_state.yoy_payload["m2"]
        risk_df_y1 = st.session_state.yoy_payload["risk_y1"]
        risk_df_y2 = st.session_state.yoy_payload["risk_y2"]
        
        # Store cleaned data for detailed comparisons
        emp1_c = st.session_state.yoy_payload.get("emp1_c")
        mgr1_c = st.session_state.yoy_payload.get("mgr1_c")
        emp2_c = st.session_state.yoy_payload.get("emp2_c")
        mgr2_c = st.session_state.yoy_payload.get("mgr2_c")

        # High-level summary cards
        divider()
        section_title("Quick Summary")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Employee Stress",
                _pct(m2["emp_stress"]),
                delta=_pct(m2["emp_stress"] - m1["emp_stress"]),
                delta_color="inverse"
            )
        with c2:
            st.metric(
                "At Risk Rate",
                _pct(m2["emp_at_risk"]),
                delta=_pct(m2["emp_at_risk"] - m1["emp_at_risk"]),
                delta_color="inverse"
            )
        with c3:
            st.metric(
                "Manager Stress",
                _pct(m2["mgr_stress"]),
                delta=_pct(m2["mgr_stress"] - m1["mgr_stress"]),
                delta_color="inverse"
            )
        with c4:
            st.metric(
                "HR Pulse Requests",
                _pct(m2["hr_pulse"]),
                delta=_pct(m2["hr_pulse"] - m1["hr_pulse"]),
                delta_color="inverse"
            )

        divider()
        
        # Detailed comparison tabs
        view_tab = st.radio(
            "Select comparison view",
            options=["� Employee Insights", "👔 Manager Insights", "🔄 Combined Analysis"],
            horizontal=True,
            key="yoy_view_tab",
        )

        # ========================================
        # TAB 1: EMPLOYEE INSIGHTS
        # ========================================
        if view_tab == "👥 Employee Insights":
            divider()
            section_title(f"Employee Insights: {y1} vs {y2}")
            
            if emp1_c is None or emp2_c is None:
                st.warning("Employee cleaned data not available. Please re-run the comparison.")
            else:
                # Additional Employee KPIs - FIRST
                with st.expander("📊 Additional Employee KPIs Comparison", expanded=True):
                    st.markdown(f"**Comparing additional employee KPIs: {y1} vs {y2}**")
                    
                    # Create a comparison table for all additional KPIs
                    kpi_data = []
                    
                    # 1. Alignment on Department Goals
                    col1 = _find_aligned_goals_col(emp1_c)
                    col2 = _find_aligned_goals_col(emp2_c)
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(emp1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(emp2_c, col2)
                        kpi_data.append(["Alignment on Dept Goals (Yes)", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    # 2. Discussed professional goals
                    rate1, col1 = _yes_rate_from_keywords(emp1_c, ["discuss", "professional", "goals"])
                    rate2, col2 = _yes_rate_from_keywords(emp2_c, ["discuss", "professional", "goals"])
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(emp1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(emp2_c, col2)
                        kpi_data.append(["Discussed Professional Goals", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    # 3. Job requirements changed
                    rate1, col1 = _yes_rate_from_keywords(emp1_c, ["encountered", "changes", "job", "requirements"])
                    rate2, col2 = _yes_rate_from_keywords(emp2_c, ["encountered", "changes", "job", "requirements"])
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(emp1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(emp2_c, col2)
                        kpi_data.append(["Job Requirements Changed", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    # 4. Adapted well (4-5)
                    rate1, col1 = _scale_good_rate_from_keywords(emp1_c, ["able", "adapt", "job", "requirements"], good_min=4)
                    rate2, col2 = _scale_good_rate_from_keywords(emp2_c, ["able", "adapt", "job", "requirements"], good_min=4)
                    if col1 and col2:
                        numeric1 = pd.to_numeric(emp1_c[col1], errors='coerce')
                        numeric2 = pd.to_numeric(emp2_c[col2], errors='coerce')
                        good1 = int((numeric1 >= 4).sum())
                        good2 = int((numeric2 >= 4).sum())
                        total1 = int(numeric1.notna().sum())
                        total2 = int(numeric2.notna().sum())
                        pct1 = (good1 / total1 * 100) if total1 > 0 else 0
                        pct2 = (good2 / total2 * 100) if total2 > 0 else 0
                        kpi_data.append(["Adapted Well (4-5 rating)", good1, f"{pct1:.1f}%", good2, f"{pct2:.1f}%", good2 - good1])
                    
                    # 5. Tasks aligned with growth
                    rate1, col1 = _yes_rate_from_keywords(emp1_c, ["tasks", "aligned", "growth"])
                    rate2, col2 = _yes_rate_from_keywords(emp2_c, ["tasks", "aligned", "growth"])
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(emp1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(emp2_c, col2)
                        kpi_data.append(["Tasks Aligned with Growth", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    # 6. Manager considers input
                    rate1, col1 = _yes_rate_from_keywords(emp1_c, ["seek", "consider", "input"])
                    rate2, col2 = _yes_rate_from_keywords(emp2_c, ["seek", "consider", "input"])
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(emp1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(emp2_c, col2)
                        kpi_data.append(["Manager Considers Input", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    # 7. Recommend company
                    rate1, col1 = _yes_rate_from_keywords(emp1_c, ["recommend", "company"])
                    rate2, col2 = _yes_rate_from_keywords(emp2_c, ["recommend", "company"])
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(emp1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(emp2_c, col2)
                        kpi_data.append(["Recommend Company", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    if kpi_data:
                        kpi_df = pd.DataFrame(kpi_data, columns=["KPI", f"{y1} Count", f"{y1} %", f"{y2} Count", f"{y2} %", "Change"])
                        st.dataframe(kpi_df, use_container_width=True, hide_index=True)
                        _download_df_csv(kpi_df, f"employee_kpis_comparison_{y1}_{y2}.csv", key="dl_yoy_emp_kpis")
                    else:
                        st.caption("Unable to detect additional employee KPI questions in both years")
                
                divider()
                
                # Stress comparison
                with st.expander("🔴 Stress Analysis", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"{y1}")
                        freq_col_1 = _find_stress_freq_col_emp(emp1_c)
                        if freq_col_1:
                            tmp1 = emp1_c.copy()
                            tmp1["_stress_freq"] = tmp1[freq_col_1].astype(str).map(_norm_stress_freq)
                            counts1 = tmp1["_stress_freq"].value_counts().reindex(STRESS_FREQ_ORDER, fill_value=0)
                            
                            for label in STRESS_FREQ_ORDER:
                                st.metric(label, int(counts1[label]))
                        else:
                            st.caption("Stress frequency column not detected")
                    
                    with col2:
                        st.subheader(f"{y2}")
                        freq_col_2 = _find_stress_freq_col_emp(emp2_c)
                        if freq_col_2:
                            tmp2 = emp2_c.copy()
                            tmp2["_stress_freq"] = tmp2[freq_col_2].astype(str).map(_norm_stress_freq)
                            counts2 = tmp2["_stress_freq"].value_counts().reindex(STRESS_FREQ_ORDER, fill_value=0)
                            
                            for label in STRESS_FREQ_ORDER:
                                delta = int(counts2[label] - counts1[label]) if freq_col_1 else 0
                                st.metric(label, int(counts2[label]), delta=delta)
                        else:
                            st.caption("Stress frequency column not detected")
                
                # Pulse-Check Meeting
                with st.expander("💬 Pulse-Check Meeting Requests", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    pulse_col_1 = _find_pulse_yn_col_emp(emp1_c)
                    pulse_col_2 = _find_pulse_yn_col_emp(emp2_c)
                    
                    if pulse_col_1 and pulse_col_2:
                        yes1 = int(_yes_mask(emp1_c[pulse_col_1]).sum())
                        yes2 = int(_yes_mask(emp2_c[pulse_col_2]).sum())
                        
                        with col1:
                            st.subheader(f"{y1}")
                            st.metric("Employees who said YES", yes1)
                        
                        with col2:
                            st.subheader(f"{y2}")
                            st.metric("Employees who said YES", yes2, delta=yes2 - yes1)
                    else:
                        st.caption("Pulse-Check question not detected in one or both years")
                
                # Work Environment
                with st.expander("🏢 Supportive Work Environment", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    env1_col = None
                    for c in emp1_c.columns:
                        h = _norm(c)
                        if ("collabor" in h) and ("support" in h) and ("environment" in h):
                            env1_col = c
                            break
                    
                    env2_col = None
                    for c in emp2_c.columns:
                        h = _norm(c)
                        if ("collabor" in h) and ("support" in h) and ("environment" in h):
                            env2_col = c
                            break
                    
                    if env1_col and env2_col:
                        yes1 = int(_yes_mask(emp1_c[env1_col]).sum())
                        no1 = int(_no_mask(emp1_c[env1_col]).sum())
                        yes2 = int(_yes_mask(emp2_c[env2_col]).sum())
                        no2 = int(_no_mask(emp2_c[env2_col]).sum())
                        
                        with col1:
                            st.subheader(f"{y1}")
                            st.metric("YES", yes1)
                            st.metric("NO", no1)
                        
                        with col2:
                            st.subheader(f"{y2}")
                            st.metric("YES", yes2, delta=yes2 - yes1)
                            st.metric("NO", no2, delta=no2 - no1, delta_color="inverse")
                    else:
                        st.caption("Supportive work environment question not detected")
                
                # Department Dynamics
                with st.expander("📈 Department Dynamics Enhancement", expanded=False):
                    dyn_col_1 = _find_dynamics_col_emp(emp1_c)
                    dyn_col_2 = _find_dynamics_col_emp(emp2_c)
                    
                    if dyn_col_1 and dyn_col_2:
                        counts1 = _dynamics_counts(emp1_c[dyn_col_1])
                        counts2 = _dynamics_counts(emp2_c[dyn_col_2])
                        
                        # Side-by-side bar chart
                        labels = DYNAMICS_ORDER
                        values1 = [int(counts1.get(k, 0)) for k in labels]
                        values2 = [int(counts2.get(k, 0)) for k in labels]
                        
                        y = np.arange(len(labels))
                        width = 0.35
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.barh([i - width/2 for i in y], values1, width, label=str(y1), alpha=0.8)
                        ax.barh([i + width/2 for i in y], values2, width, label=str(y2), alpha=0.8)
                        ax.set_yticks(y)
                        ax.set_yticklabels(labels)
                        ax.invert_yaxis()
                        ax.set_xlabel("Count")
                        ax.set_title(f"Department Dynamics: {y1} vs {y2}")
                        ax.legend()
                        
                        st.pyplot(fig, clear_figure=False)
                        _download_fig_png(fig, f"dynamics_comparison_{y1}_{y2}.png", key="dl_yoy_dynamics")
                    else:
                        st.caption("Department dynamics question not detected")

        # ========================================
        # TAB 3: MANAGER INSIGHTS
        # ========================================
        elif view_tab == "👔 Manager Insights":
            divider()
            section_title(f"Manager Insights: {y1} vs {y2}")
            
            if mgr1_c is None or mgr2_c is None:
                st.warning("Manager cleaned data not available. Please re-run the comparison.")
            else:
                # Additional Manager KPIs - FIRST
                with st.expander("📊 Additional Manager KPIs Comparison", expanded=True):
                    st.markdown(f"**Comparing additional manager KPIs: {y1} vs {y2}**")
                    st.caption("Showing percentage of employees for whom managers answered YES")
                    
                    # Create a comparison table for all 6 manager KPIs
                    kpi_data = []
                    
                    # 1. Encountered changes in job requirements
                    rate1, col1 = _yes_rate_from_keywords(mgr1_c, ["encountered", "changes", "job", "requirements"])
                    rate2, col2 = _yes_rate_from_keywords(mgr2_c, ["encountered", "changes", "job", "requirements"])
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(mgr1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(mgr2_c, col2)
                        kpi_data.append(["Encountered Job Changes", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    # 2. At risk of low performance
                    rate1, col1 = _yes_rate_from_keywords(mgr1_c, ["at risk", "low performance"])
                    if not col1:
                        rate1, col1 = _yes_rate_from_keywords(mgr1_c, ["risk", "performance"])
                    rate2, col2 = _yes_rate_from_keywords(mgr2_c, ["at risk", "low performance"])
                    if not col2:
                        rate2, col2 = _yes_rate_from_keywords(mgr2_c, ["risk", "performance"])
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(mgr1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(mgr2_c, col2)
                        kpi_data.append(["At Risk of Low Performance", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    # 3. Would perform better in another department
                    rate1, col1 = _yes_rate_from_keywords(mgr1_c, ["perform better", "another department"])
                    if not col1:
                        rate1, col1 = _yes_rate_from_keywords(mgr1_c, ["better", "department"])
                    rate2, col2 = _yes_rate_from_keywords(mgr2_c, ["perform better", "another department"])
                    if not col2:
                        rate2, col2 = _yes_rate_from_keywords(mgr2_c, ["better", "department"])
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(mgr1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(mgr2_c, col2)
                        kpi_data.append(["Better in Another Dept", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    # 4. Ready for promotion today
                    rate1, col1 = _yes_rate_from_keywords(mgr1_c, ["ready", "promotion"])
                    rate2, col2 = _yes_rate_from_keywords(mgr2_c, ["ready", "promotion"])
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(mgr1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(mgr2_c, col2)
                        kpi_data.append(["Ready for Promotion", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    # 5. Actively seek and consider team members' input
                    rate1, col1 = _yes_rate_from_keywords(mgr1_c, ["actively seek", "team members", "input"])
                    if not col1:
                        rate1, col1 = _yes_rate_from_keywords(mgr1_c, ["seek", "consider", "input"])
                    rate2, col2 = _yes_rate_from_keywords(mgr2_c, ["actively seek", "team members", "input"])
                    if not col2:
                        rate2, col2 = _yes_rate_from_keywords(mgr2_c, ["seek", "consider", "input"])
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(mgr1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(mgr2_c, col2)
                        kpi_data.append(["Seeks Team Input", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    # 6. Fits in company culture
                    rate1, col1 = _yes_rate_from_keywords(mgr1_c, ["fits", "company culture"])
                    if not col1:
                        rate1, col1 = _yes_rate_from_keywords(mgr1_c, ["person", "fits", "culture"])
                    rate2, col2 = _yes_rate_from_keywords(mgr2_c, ["fits", "company culture"])
                    if not col2:
                        rate2, col2 = _yes_rate_from_keywords(mgr2_c, ["person", "fits", "culture"])
                    if col1 and col2:
                        yes1, _, pct1, _ = _count_yes_no(mgr1_c, col1)
                        yes2, _, pct2, _ = _count_yes_no(mgr2_c, col2)
                        kpi_data.append(["Fits in Company Culture", yes1, f"{pct1:.1f}%", yes2, f"{pct2:.1f}%", yes2 - yes1])
                    
                    if kpi_data:
                        kpi_df = pd.DataFrame(kpi_data, columns=["KPI", f"{y1} Count", f"{y1} %", f"{y2} Count", f"{y2} %", "Change"])
                        st.dataframe(kpi_df, use_container_width=True, hide_index=True)
                        _download_df_csv(kpi_df, f"manager_kpis_comparison_{y1}_{y2}.csv", key="dl_yoy_mgr_kpis")
                    else:
                        st.caption("Unable to detect additional manager KPI questions in both years")
                
                divider()
                
                # At Risk comparison
                with st.expander("⚠️ At Risk of Low Performance", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"{y1}")
                        st.metric("Employees at risk", len(risk_df_y1))
                        if not risk_df_y1.empty:
                            st.dataframe(risk_df_y1[["Subordinate Name"]].head(10), use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.subheader(f"{y2}")
                        delta = len(risk_df_y2) - len(risk_df_y1)
                        st.metric("Employees at risk", len(risk_df_y2), delta=delta, delta_color="inverse")
                        if not risk_df_y2.empty:
                            st.dataframe(risk_df_y2[["Subordinate Name"]].head(10), use_container_width=True, hide_index=True)
                
                # Manager Stress
                with st.expander("😰 Manager Stress about Employees", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"{y1}")
                        st.metric("Manager Stress Rate", _pct(m1["mgr_stress"]))
                    
                    with col2:
                        st.subheader(f"{y2}")
                        st.metric("Manager Stress Rate", _pct(m2["mgr_stress"]), delta=_pct(m2["mgr_stress"] - m1["mgr_stress"]))
                
                # Promotion readiness
                with st.expander("🎯 Promotion Readiness", expanded=False):
                    promo_col_1 = _find_ready_for_promotion_col(mgr1_c)
                    promo_col_2 = _find_ready_for_promotion_col(mgr2_c)
                    
                    if promo_col_1 and promo_col_2:
                        yes1 = int(_yes_mask(mgr1_c[promo_col_1]).sum())
                        no1 = int(_no_mask(mgr1_c[promo_col_1]).sum())
                        yes2 = int(_yes_mask(mgr2_c[promo_col_2]).sum())
                        no2 = int(_no_mask(mgr2_c[promo_col_2]).sum())
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader(f"{y1}")
                            st.metric("Ready for promotion (Yes)", yes1)
                            st.metric("Not ready (No)", no1)
                        
                        with col2:
                            st.subheader(f"{y2}")
                            st.metric("Ready for promotion (Yes)", yes2, delta=yes2 - yes1)
                            st.metric("Not ready (No)", no2, delta=no2 - no1)
                    else:
                        st.caption("Promotion readiness question not detected")
                
                # Culture fit
                with st.expander("🎭 Company Culture Fit", expanded=False):
                    culture_col_1 = _find_col_by_keywords(mgr1_c, ["fits", "company", "culture"])
                    culture_col_2 = _find_col_by_keywords(mgr2_c, ["fits", "company", "culture"])
                    
                    if culture_col_1 and culture_col_2:
                        yes1 = int(_yes_mask(mgr1_c[culture_col_1]).sum())
                        no1 = int(_no_mask(mgr1_c[culture_col_1]).sum())
                        yes2 = int(_yes_mask(mgr2_c[culture_col_2]).sum())
                        no2 = int(_no_mask(mgr2_c[culture_col_2]).sum())
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader(f"{y1}")
                            st.metric("Employees fit culture", yes1)
                            st.metric("Don't fit", no1)
                        
                        with col2:
                            st.subheader(f"{y2}")
                            st.metric("Employees fit culture", yes2, delta=yes2 - yes1)
                            st.metric("Don't fit", no2, delta=no2 - no1, delta_color="inverse")
                    else:
                        st.caption("Culture fit question not detected")

        # ========================================
        # TAB 4: COMBINED ANALYSIS
        # ========================================
        elif view_tab == "🔄 Combined Analysis":
            divider()
            section_title(f"Combined Employee & Manager Analysis: {y1} vs {y2}")
            
            if emp1_c is None or emp2_c is None or mgr1_c is None or mgr2_c is None:
                st.warning("Complete cleaned data not available. Please re-run the comparison.")
            else:
                # Job changes
                with st.expander("💼 Changes in Job Responsibilities", expanded=True):
                    emp_yn_col_1 = _find_job_change_yn_col_emp(emp1_c)
                    mgr_yn_col_1 = _find_job_change_yn_col_mgr(mgr1_c)
                    emp_yn_col_2 = _find_job_change_yn_col_emp(emp2_c)
                    mgr_yn_col_2 = _find_job_change_yn_col_mgr(mgr2_c)
                    
                    if emp_yn_col_1 and mgr_yn_col_1 and emp_yn_col_2 and mgr_yn_col_2:
                        emp_yes_1 = int(_yes_mask(emp1_c[emp_yn_col_1]).sum())
                        mgr_yes_1 = int(_yes_mask(mgr1_c[mgr_yn_col_1]).sum())
                        emp_yes_2 = int(_yes_mask(emp2_c[emp_yn_col_2]).sum())
                        mgr_yes_2 = int(_yes_mask(mgr2_c[mgr_yn_col_2]).sum())
                        
                        st.markdown("**Yes responses comparison:**")
                        
                        comparison_df = pd.DataFrame({
                            "Perspective": ["Employees", "Managers"],
                            str(y1): [emp_yes_1, mgr_yes_1],
                            str(y2): [emp_yes_2, mgr_yes_2],
                            "Change": [emp_yes_2 - emp_yes_1, mgr_yes_2 - mgr_yes_1]
                        })
                        
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Job change questions not detected in both years")
                
                # Check-in meeting frequency
                with st.expander("📅 Check-in Meeting Frequency", expanded=False):
                    emp_freq_col_1 = _find_checkin_freq_col_emp(emp1_c)
                    mgr_freq_col_1 = _find_checkin_freq_col_mgr(mgr1_c)
                    emp_freq_col_2 = _find_checkin_freq_col_emp(emp2_c)
                    mgr_freq_col_2 = _find_checkin_freq_col_mgr(mgr2_c)
                    
                    if emp_freq_col_1 and mgr_freq_col_1 and emp_freq_col_2 and mgr_freq_col_2:
                        def _pct_series(s: pd.Series) -> pd.Series:
                            base = s[s.isin(FREQ_CATS)]
                            if len(base) == 0:
                                return pd.Series({k: 0.0 for k in FREQ_CATS})
                            return (base.value_counts(normalize=True) * 100).reindex(FREQ_CATS, fill_value=0.0)
                        
                        emp_freq_1 = emp1_c[emp_freq_col_1].astype(str).map(_norm_freq)
                        mgr_freq_1 = mgr1_c[mgr_freq_col_1].astype(str).map(_norm_freq)
                        emp_freq_2 = emp2_c[emp_freq_col_2].astype(str).map(_norm_freq)
                        mgr_freq_2 = mgr2_c[mgr_freq_col_2].astype(str).map(_norm_freq)
                        
                        emp_pct_1 = _pct_series(emp_freq_1)
                        mgr_pct_1 = _pct_series(mgr_freq_1)
                        emp_pct_2 = _pct_series(emp_freq_2)
                        mgr_pct_2 = _pct_series(mgr_freq_2)
                        
                        # Create comparison chart
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                        
                        y = np.arange(len(FREQ_CATS))
                        h = 0.35
                        
                        # Year 1
                        ax1.barh(y - h/2, mgr_pct_1.values, height=h, label="Managers", alpha=0.8)
                        ax1.barh(y + h/2, emp_pct_1.values, height=h, label="Employees", alpha=0.8)
                        ax1.set_yticks(y)
                        ax1.set_yticklabels(FREQ_CATS)
                        ax1.invert_yaxis()
                        ax1.set_xlabel("% of responses")
                        ax1.set_title(f"{y1}")
                        ax1.legend()
                        
                        # Year 2
                        ax2.barh(y - h/2, mgr_pct_2.values, height=h, label="Managers", alpha=0.8)
                        ax2.barh(y + h/2, emp_pct_2.values, height=h, label="Employees", alpha=0.8)
                        ax2.set_yticks(y)
                        ax2.set_yticklabels(FREQ_CATS)
                        ax2.invert_yaxis()
                        ax2.set_xlabel("% of responses")
                        ax2.set_title(f"{y2}")
                        ax2.legend()
                        
                        st.pyplot(fig, clear_figure=False)
                        _download_fig_png(fig, f"checkin_freq_comparison_{y1}_{y2}.png", key="dl_yoy_checkin_freq")
                    else:
                        st.caption("Check-in frequency questions not detected in both years")
                
                # Reward & Recognition
                with st.expander("🏆 Reward & Recognition Methods", expanded=False):
                    emp_recog_col_1 = _find_col_by_keywords(emp1_c, ["rewards", "recognizes", "contribution"])
                    mgr_recog_col_1 = _find_col_by_keywords(mgr1_c, ["recognize", "reward", "contributions"])
                    emp_recog_col_2 = _find_col_by_keywords(emp2_c, ["rewards", "recognizes", "contribution"])
                    mgr_recog_col_2 = _find_col_by_keywords(mgr2_c, ["recognize", "reward", "contributions"])
                    
                    if all([emp_recog_col_1, mgr_recog_col_1, emp_recog_col_2, mgr_recog_col_2]):
                        emp_counts_1 = _recog_counts(emp1_c[emp_recog_col_1])
                        mgr_counts_1 = _recog_counts(mgr1_c[mgr_recog_col_1])
                        emp_counts_2 = _recog_counts(emp2_c[emp_recog_col_2])
                        mgr_counts_2 = _recog_counts(mgr2_c[mgr_recog_col_2])
                        
                        comparison_df = pd.DataFrame({
                            "Method": RECOG_ORDER,
                            f"Emp {y1}": [int(emp_counts_1.get(m, 0)) for m in RECOG_ORDER],
                            f"Mgr {y1}": [int(mgr_counts_1.get(m, 0)) for m in RECOG_ORDER],
                            f"Emp {y2}": [int(emp_counts_2.get(m, 0)) for m in RECOG_ORDER],
                            f"Mgr {y2}": [int(mgr_counts_2.get(m, 0)) for m in RECOG_ORDER],
                        })
                        
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        _download_df_csv(comparison_df, f"recognition_comparison_{y1}_{y2}.csv", key="dl_yoy_recognition")
                    else:
                        st.caption("Recognition questions not detected in both years")


divider()
st.caption("© HR Analytics Cleaning Tool")
