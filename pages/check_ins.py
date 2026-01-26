import io
import os
import re
from collections import Counter
from pathlib import Path

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
page_header("Check-ins", "KPIs • Analysis • Compare two years")


# --------------------
# Sidebar navigation
# --------------------
st.sidebar.title("Check-ins")
section = st.sidebar.radio(
    "Sections",
    ["KPIs", "Analysis", "Compare 2 Years"],
    index=0
)


# ============================================================
# Keyword matching helpers (KPIs + Analysis)
# ============================================================
def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)          # collapse spaces/newlines
    s = re.sub(r"[^\w\s]", " ", s)      # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
        lc = str(c).strip().lower().replace("\\", "/")
        if lc == "company name / department" or lc == "company name/department":
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
        c = _find_dept_col(df)
        if c:
            vals |= set(df[c].dropna().astype(str).map(_norm_dept_value).unique().tolist())
    vals = {v for v in vals if v.strip()}
    return ["All departments"] + sorted(vals)


# --------------------
# KPI maps (keyword-based)
# --------------------
EMP_YN_KPIS = [
    ("Aligned on dept goals", ["aligned", "department", "goals"]),
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
def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)          # collapse spaces/newlines
    s = re.sub(r"[^\w\s]", " ", s)      # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

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

# --- Manager KPI helpers ---
def _find_ready_for_promotion_col(df: pd.DataFrame) -> str | None:
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
def _looks_like_scale_1_5(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    uniq = set(s.unique())
    return uniq.issubset({1,2,3,4,5})

def _detect_open_ended_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if _is_identity_col(c):
            continue
        s = df[c]
        if _looks_like_yesno(s) or _looks_like_scale_1_5(s):
            continue
        if s.dtype == object:
            vals = s.dropna().astype(str)
            if vals.empty:
                continue
            avg_len = vals.map(len).mean()
            uniq_ratio = vals.nunique() / max(len(vals), 1)
            if avg_len >= 20 and uniq_ratio >= 0.25:
                cols.append(c)
    return cols

def _clean_text(x: str) -> str:
    x = str(x)
    x = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[email]", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) >= 3 and t not in STOPWORDS]

def _top_keywords(texts: list[str], top_n: int = 25) -> pd.DataFrame:
    counter = Counter()
    for t in texts:
        counter.update(_tokenize(t))
    most = counter.most_common(top_n)
    return pd.DataFrame(most, columns=["Keyword", "Count"])

_analyzer = SentimentIntensityAnalyzer()

def _sentiment_score(text: str) -> float:
    return _analyzer.polarity_scores(text)["compound"]

def _sentiment_label(score: float) -> str:
    if score <= -0.05:
        return "Negative"
    if score >= 0.05:
        return "Positive"
    return "Neutral"

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
    st.switch_page("Home.py")  # change back to streamlit_app.py if you didn't rename


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
            scores_all = [_sentiment_score(t) for t in texts]
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

            col1, col2 = st.columns([1.2, 0.8])
            with col1:
                section_title("Recurring topics (with sentiment)")
                st.dataframe(_topic_summary_with_sentiment(texts), use_container_width=True, hide_index=True)

                divider()
                section_title("Action plan (Themes → Suggested actions)")

                plan_df = _reasons_action_plan(texts)

                if plan_df.empty:
                    st.info("No clear themes detected to generate an action plan.")
                else:
                    st.dataframe(plan_df, use_container_width=True, hide_index=True)

                    st.markdown("**Top priorities (quick view):**")
                    for _, r in plan_df.head(3).iterrows():
                        st.write(f"- **{r['Theme']}** ({r['% of No elaborations']}%): {r['Recommended actions']}")

            with col2:
                section_title("Top keywords")
                st.dataframe(_top_keywords(texts, top_n=25), use_container_width=True, hide_index=True)
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
        st.dataframe(_topic_summary_with_sentiment(row_texts), use_container_width=True, hide_index=True)
    with col2:
        section_title("Top keywords")
        st.dataframe(_top_keywords(row_texts, top_n=25), use_container_width=True, hide_index=True)


# ============================================================
# SECTION: KPIs
# ============================================================
if section == "KPIs":

    section_title("KPIs")
    if st.session_state.clean_ready:
        if st.button("🔄 Re-run"):
            st.session_state.clean_ready = False
            st.session_state.clean_error = None
            st.session_state.cleaned_emp = None
            st.session_state.cleaned_mgr = None
            st.session_state.combined = None
            st.rerun()

    divider()

    with st.expander("Upload / Data settings", expanded=not st.session_state.clean_ready):

        base_path = Path(os.getcwd())
        years, _ = _available_years_from_data(base_path)

        if years:
            selected_year_data = st.selectbox(
                "Select year (from Data folder)",
                options=years,
                index=len(years) - 1,
                key="kpi_year_from_data"
            )
            use_data = st.checkbox(
                "Use Data folder files (recommended)",
                value=True,
                key="use_data_kpis"
            )
        else:
            use_data = False
            st.warning("No complete year files detected in Data/ (need both employee + manager files with year in name).")

        col_left, col_right = st.columns(2)
        with col_left:
            emp_file = st.file_uploader(
                "Employee Check-In File (.xlsx/.xls/.csv)",
                type=["xlsx", "xls", "csv"],
                key="emp"
            )
        with col_right:
            mgr_file = st.file_uploader(
                "Manager Check-In File (.xlsx/.xls/.csv)",
                type=["xlsx", "xls", "csv"],
                key="mgr"
            )

        mena_df_preview = _load_static_mena()
        if mena_df_preview.empty:
            st.warning("No Mena Report file detected in Data/ or root. Place one before cleaning.")

        
    run_btn = st.button("Run Cleaning", type="primary", use_container_width=True)

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
        st.success("Cleaning complete ✅")
        st.caption(f"Employee rows: {len(st.session_state.cleaned_emp)} | Manager rows: {len(st.session_state.cleaned_mgr)}")

        cleaned_emp = st.session_state.cleaned_emp
        cleaned_mgr = st.session_state.cleaned_mgr

        # --- Department filter ---
        dept_choices = _dept_options(cleaned_emp, cleaned_mgr)
        selected_dept = st.selectbox(
            "Filter by department",
            options=dept_choices,
            index=0,
            key="kpi_dept_filter"
        )

        emp_dept_col = _find_dept_col(cleaned_emp)
        mgr_dept_col = _find_dept_col(cleaned_mgr)

        emp_f = _filter_by_dept(cleaned_emp, emp_dept_col, selected_dept)
        mgr_f = _filter_by_dept(cleaned_mgr, mgr_dept_col, selected_dept)

        divider()
        section_title("Completion Rate (manual inputs)")

        # Make keys unique per year+dept view so it doesn't mix inputs
        ctx = selected_dept if "selected_dept" in locals() else "All"
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
            options=["Employee", "Manager"],
            index=0,
            horizontal=True,
            key="kpi_view_choice",
        )

        # --- Employee KPIs ---
        if view_choice == "Employee":
            divider()
            section_title("Employee KPIs")

            c2, c3 = st.columns(2)
            c2.metric("Stress rate (Employees)", _pct(m["emp_stress"]))
            c3.metric("HR pulse request rate", _pct(m["hr_pulse"]))

            # ✅ Additional Employee KPIs MUST be inside here
            divider()
            section_title("Additional Employee KPIs (keyword-based)")

            cols = st.columns(4)
            for i, (label, keys) in enumerate(EMP_YN_KPIS):
                rate, col = _yes_rate_from_keywords(emp_f, keys)
                value = _pct(rate) if col else "N/A"
                cols[i % 4].metric(label, value)
                if (i % 4) == 3 and i != len(EMP_YN_KPIS) - 1:
                    cols = st.columns(4)

            for label, keys, good_min in EMP_SCALE_KPIS:
                rate, col = _scale_good_rate_from_keywords(emp_f, keys, good_min=good_min)
                st.metric(label, _pct(rate) if col else "N/A")

        # --- Manager KPIs ---
        if view_choice == "Manager":
            divider()
            section_title("Manager KPIs")

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

                section_title("Ready for promotion (Yes)")
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
                section_title("Not ready for promotion (No) — Suggested review dates")

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

                        # (Removed reconciliation section per request)
        
        # -------------------------
        # SECTION: Compare 2 Years
        # -------------------------
        if section == "Compare 2 Years":
            divider()
            section_title("Compare 2 Years")

            base_path = Path(os.getcwd())
            df_mena = _load_static_mena()

            if df_mena.empty:
                st.error("Static Mena Report file not found in root. Place 'Mena Report.xlsx' or CSV variant.")
            else:
                try:
                    paths = _find_data_candidates(base_path)
                    by_year = _classify_year_files(paths)
                    years = sorted(by_year.keys())

                    if len(years) < 2:
                        st.warning("Need at least two years of files in Data/ to compare.")
                    else:
                        y2, y1 = years[-1], years[-2]

                        missing = []
                        for y in [y1, y2]:
                            if "emp" not in by_year[y] or "mgr" not in by_year[y]:
                                missing.append(y)

                        if missing:
                            st.warning(f"Missing employee/manager files for years: {', '.join(map(str, missing))}.")
                        else:
                            cleaner = CheckInExcelCleaner()

                            def _read_any(p: Path) -> pd.DataFrame:
                                if p.suffix.lower() == ".csv":
                                    return pd.read_csv(str(p))
                                return pd.read_excel(str(p))

                            emp1 = _read_any(by_year[y1]["emp"])
                            mgr1 = _read_any(by_year[y1]["mgr"])
                            emp2 = _read_any(by_year[y2]["emp"])
                            mgr2 = _read_any(by_year[y2]["mgr"])

                            st.info(f"Cleaning {y1}…")
                            emp1_c = cleaner.clean_employee_checkin(emp1, df_mena)
                            mgr1_c = cleaner.clean_manager_checkin(mgr1, df_mena)

                            st.info(f"Cleaning {y2}…")
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
                            }

                except Exception as e:
                    st.error(f"Year-over-Year load failed: {e}")
                    st.exception(e)

            if st.session_state.yoy_ready and st.session_state.yoy_payload:
                y1 = st.session_state.yoy_payload["y1"]
                y2 = st.session_state.yoy_payload["y2"]
                m1 = st.session_state.yoy_payload["m1"]
                m2 = st.session_state.yoy_payload["m2"]
                risk_df_y1 = st.session_state.yoy_payload["risk_y1"]
                risk_df_y2 = st.session_state.yoy_payload["risk_y2"]

                section_title("Summary")
                col_y1, col_y2 = st.columns(2)

                with col_y1:
                    st.subheader(str(y1))
                    st.metric("Employee Completion", _pct(m1["emp_completion"]))
                    st.metric("Manager Completion", _pct(m1["mgr_completion"]))
                    st.metric("Employee Stress", _pct(m1["emp_stress"]))
                    st.metric("At Risk (Employees)", _pct(m1["emp_at_risk"]))

                    if not risk_df_y1.empty:
                        st.download_button(
                            f"Download at-risk employees ({y1})",
                            risk_df_y1.to_csv(index=False).encode("utf-8"),
                            file_name=f"at_risk_employees_{y1}.csv",
                            mime="text/csv",
                            key=f"dl_risk_{y1}",
                            use_container_width=True,
                        )
                    else:
                        st.caption("No at-risk employees detected (or risk column not found).")

                with col_y2:
                    st.subheader(str(y2))
                    st.metric("Employee Completion", _pct(m2["emp_completion"]), delta=_pct(m2["emp_completion"] - m1["emp_completion"]))
                    st.metric("Manager Completion", _pct(m2["mgr_completion"]), delta=_pct(m2["mgr_completion"] - m1["mgr_completion"]))
                    st.metric("Employee Stress", _pct(m2["emp_stress"]), delta=_pct(m2["emp_stress"] - m1["emp_stress"]))
                    st.metric("At Risk (Employees)", _pct(m2["emp_at_risk"]), delta=_pct(m2["emp_at_risk"] - m1["emp_at_risk"]))

                    if not risk_df_y2.empty:
                        st.download_button(
                            f"Download at-risk employees ({y2})",
                            risk_df_y2.to_csv(index=False).encode("utf-8"),
                            file_name=f"at_risk_employees_{y2}.csv",
                            mime="text/csv",
                            key=f"dl_risk_{y2}",
                            use_container_width=True,
                        )
                    else:
                        st.caption("No at-risk employees detected (or risk column not found).")


divider()
st.caption("© HR Analytics Cleaning Tool")
