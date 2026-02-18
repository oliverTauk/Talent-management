import io
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import streamlit as st

from hr_analytics.services.checkin_cleaner_excel import CheckInExcelCleaner
from hr_analytics.services.kpis import KPIService
from hr_analytics.ui.style import apply_global_style, page_header, section_title, divider
from hr_analytics.ui.downloads import _download_df_csv, _download_fig_png

# --- Column detection ---
from hr_analytics.services.column_detection import (
    _norm,
    _find_col_by_keywords,
    _find_aligned_goals_col,
    _find_if_no_elaborate_col,
    _yes_rate_from_keywords,
    _scale_good_rate_from_keywords,
    _find_mgr_behavior_question_col_emp,
    _find_manager_name_col_emp,
    _find_ready_for_promotion_col,
    _find_employee_name_col,
    _find_your_name_col,
    _find_subordinate_name_col,
    _find_promo_position_col,
    _find_promo_review_date_col,
    _find_risk_reason_col,
    _find_pip_reason_no_col,
    _find_checkin_freq_col_emp,
    _find_checkin_freq_col_mgr,
    _find_company_resources_col_emp,
    _find_resources_other_text_col_emp,
    _find_pulse_yn_col_emp,
    _find_pulse_reason_col_emp,
    _find_stress_freq_col_emp,
    _find_stress_reason_col_emp,
    _find_stress_reasons_col,
    _find_mgr_stress_freq_col,
    _find_mgr_dynamics_col_mgr,
    _find_dynamics_col_emp,
    _find_team_integration_col_emp,
    _find_team_integration_col_mgr,
    _find_input_seek_col_emp,
    _find_input_seek_col_mgr,
    _find_job_change_yn_col_emp,
    _find_job_change_yn_col_mgr,
    _find_job_change_types_col,
)

# --- Normalizers, constants, masks, helpers ---
from hr_analytics.services.normalizers import (
    _yes_mask,
    _no_mask,
    _count_yes_no,
    _split_multiselect,
    POS_TAGS,
    NEG_TAGS,
    _norm_behavior_opt,
    STRESS_FREQ_ORDER,
    _norm_stress_freq,
    STRESS_REASON_ORDER,
    _norm_stress_reason,
    _stress_reason_counts,
    MGR_DYNAMICS_ORDER,
    _mgr_dynamics_counts,
    DYNAMICS_ORDER,
    _dynamics_counts,
    _norm_team_integration,
    MISTAKE_ORDER,
    UNDESIRED,
    _norm_mistake_opt,
    _mistake_counts,
    RECOG_ORDER,
    _recog_counts,
    CHANGE_CATS,
    _change_counts,
    FREQ_CATS,
    _norm_freq,
    RES_ORDER,
    _norm_resource_opt,
    _pulse_reason_counts,
    _parse_any_date,
    _extract_month_year_label,
    _find_dept_col,
    _filter_by_dept,
    _dept_options,
    _add_year_from_timestamp,
    _metrics,
    _pct,
)

# --- Text analysis ---
from hr_analytics.services.text_analysis import (
    _sentiment_score,
    _sentiment_label,
    _top_keywords,
    _clean_text,
    _detect_open_ended_columns,
    _topic_summary_with_sentiment,
    _reasons_action_plan,
    _is_elaboration_header,
    _build_reasons_for_no_pairs,
)


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
    ["KPIs", "Compare"],
    index=0
)

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
        else:
            return pd.read_excel(str(p), engine="openpyxl")



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
    """Returns: (cleaned_emp, cleaned_mgr, combined) for the selected year."""
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
        return pd.read_excel(str(p), engine="openpyxl")

    emp_raw = _read_any(by_year[year]["emp"])
    mgr_raw = _read_any(by_year[year]["mgr"])

    cleaner = CheckInExcelCleaner()
    cleaned_emp = cleaner.clean_employee_checkin(emp_raw, df_mena)
    cleaned_mgr = cleaner.clean_manager_checkin(mgr_raw, df_mena)
    combined = cleaner.combine_cleaned(cleaned_emp, cleaned_mgr)

    return cleaned_emp, cleaned_mgr, combined


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
# Shared: Auto-detect check-in files for YoY comparison
# ============================================================
_CHECKIN_BASE = Path("Data/Check-ins 2024 2025")

def _auto_load_checkin_files() -> dict:
    """Auto-detect the latest two years of check-in files."""
    files_info: dict = {}

    if not _CHECKIN_BASE.exists():
        return files_info

    year_dirs = sorted(
        [d for d in _CHECKIN_BASE.iterdir() if d.is_dir() and d.name.isdigit()],
        reverse=True,
    )
    if len(year_dirs) < 2:
        return files_info

    y2_dir = year_dirs[0]
    y2 = int(y2_dir.name)
    y1_dir = year_dirs[1]
    y1 = int(y1_dir.name)

    def _find_files(directory):
        emp, mgr = None, None
        for f in directory.glob("*.xlsx"):
            if f.name.startswith("~$"):
                continue
            fname_lower = f.name.lower()
            if ("employee answers" in fname_lower or "employee questions" in fname_lower) and "performance check-in" in fname_lower:
                emp = f
            elif ("manager answers" in fname_lower or "manager questions" in fname_lower) and "performance check-in" in fname_lower:
                mgr = f
        return emp, mgr

    emp_y2, mgr_y2 = _find_files(y2_dir)
    emp_y1, mgr_y1 = _find_files(y1_dir)

    return {
        "y1": y1, "y2": y2,
        "emp_y1": emp_y1, "mgr_y1": mgr_y1,
        "emp_y2": emp_y2, "mgr_y2": mgr_y2,
    }


def _run_comparison(auto_files: dict | None = None) -> None:
    """Run the YoY comparison pipeline and store results in session state.
    Called by the KPI Run button so both sections update together.
    """
    st.session_state.yoy_ready = False
    st.session_state.yoy_payload = None

    if auto_files is None:
        auto_files = _auto_load_checkin_files()

    # Resolve files: prefer any manual uploads stored in session, else auto-detected
    use_emp_y1 = st.session_state.get("compare_emp_y1") or (auto_files.get("emp_y1") if auto_files else None)
    use_mgr_y1 = st.session_state.get("compare_mgr_y1") or (auto_files.get("mgr_y1") if auto_files else None)
    use_emp_y2 = st.session_state.get("compare_emp_y2") or (auto_files.get("emp_y2") if auto_files else None)
    use_mgr_y2 = st.session_state.get("compare_mgr_y2") or (auto_files.get("mgr_y2") if auto_files else None)

    if not (use_emp_y1 and use_mgr_y1 and use_emp_y2 and use_mgr_y2):
        # Not enough files — silently skip (no error when auto-detecting)
        return

    df_mena = _load_static_mena()
    if df_mena.empty:
        return

    try:
        cleaner = CheckInExcelCleaner()

        def _load_file(file_or_path):
            if isinstance(file_or_path, Path):
                return pd.read_excel(file_or_path) if file_or_path.suffix in ['.xlsx', '.xls'] else pd.read_csv(file_or_path)
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

        emp1_c = cleaner.clean_employee_checkin(emp1, df_mena)
        mgr1_c = cleaner.clean_manager_checkin(mgr1, df_mena)
        emp2_c = cleaner.clean_employee_checkin(emp2, df_mena)
        mgr2_c = cleaner.clean_manager_checkin(mgr2, df_mena)

        m1 = _metrics(emp1_c, mgr1_c)
        m2 = _metrics(emp2_c, mgr2_c)

        kpi = KPIService()
        risk_h1 = kpi.detect_manager_risk_header(mgr1_c)
        risk_h2 = kpi.detect_manager_risk_header(mgr2_c)

        def _at_risk_rows(mgr_df, risk_col):
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
    except Exception:
        pass  # Comparison silently skipped if it fails


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
                    name_col = _find_your_name_col(emp_f)
                    if name_col and name_col in emp_f.columns:
                        no_names = emp_f.loc[no_mask, name_col].astype(str).fillna("").str.strip()
                        no_names = [n for n in no_names.tolist() if n]
                        if no_names:
                            st.caption(", ".join(no_names))
                    else:
                        st.caption("Employee name column not found to list NO responders.")

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
            section_title("Employee Stress")
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
                    reason_col = _find_stress_reasons_col(emp_f)
                    name_col = _find_your_name_col(emp_f)

                    if not reason_col:
                        st.info("Could not detect the stress reasons multi-select question in employee data.")
                    elif not name_col or name_col not in emp_f.columns:
                        st.info("Employee name column not found in employee data.")
                    else:
                        reason_to_names: dict[str, set[str]] = {k: set() for k in STRESS_REASON_ORDER}

                        for _, r in emp_f[[name_col, reason_col]].iterrows():
                            emp_name = str(r[name_col]).strip()
                            if not emp_name:
                                continue

                            picks = _split_multiselect(r[reason_col])
                            for p in picks:
                                k = _norm_stress_reason(p)
                                if k:
                                    reason_to_names[k].add(emp_name)

                        rows = []
                        for reason in STRESS_REASON_ORDER:
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
            section_title("Pulse-Check Meeting")
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
            name_col = _find_your_name_col(emp_f)

            if not res_col:
                st.info("Could not detect the Company Resources question column in employee data.")
            elif not name_col or name_col not in emp_f.columns:
                st.info("Employee name column not found in employee data.")
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
            reason_col = _find_stress_reasons_col(mgr_f)

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
                    ax.set_title(f"Check-ins meeting frequency ({selected_year})")
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
                emp_name_col = _find_your_name_col(emp_f)
                mgr_name_col = _find_subordinate_name_col(mgr_f) or "Subordinate Name"

                if not emp_q or not mgr_q:
                    st.info("Could not detect the input-seeking Yes/No question in employee or manager data.")
                elif not emp_name_col or emp_name_col not in emp_f.columns:
                    st.info("Employee name column not found in employee data.")
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

                        emp_employee_name_col = _find_your_name_col(emp_f)

                        # Manager name column in employee check-in (if your cleaned file uses something else, replace it)
                        emp_manager_name_col = _find_manager_name_col_emp(emp_f)

                        if not emp_manager_name_col:
                            st.caption("Manager name column not detected in employee data, so we cannot list 'mentioned by' cases.")
                        elif not emp_employee_name_col or emp_employee_name_col not in emp_f.columns:
                            st.caption("Employee name column not found in employee data.")
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


# -------------------------
# SECTION: Compare 2 Years
# -------------------------
elif section == "Compare":
    divider()
    section_title("Year-over-Year Comparison")

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
            st.file_uploader(
                "Employee Check-In (Year 1)",
                type=["xlsx", "xls", "csv"],
                key="compare_emp_y1",
            )
            st.file_uploader(
                "Manager Check-In (Year 1)",
                type=["xlsx", "xls", "csv"],
                key="compare_mgr_y1",
            )

        with col2:
            st.subheader("Year 2 (Later)")
            st.file_uploader(
                "Employee Check-In (Year 2)",
                type=["xlsx", "xls", "csv"],
                key="compare_emp_y2",
            )
            st.file_uploader(
                "Manager Check-In (Year 2)",
                type=["xlsx", "xls", "csv"],
                key="compare_mgr_y2",
            )

    compare_run_btn = st.button("Run Comparison", type="primary", use_container_width=True, key="compare_run_btn")

    if compare_run_btn:
        st.session_state.yoy_ready = False
        st.session_state.yoy_payload = None
        _run_comparison(auto_files)

    if not st.session_state.yoy_ready:
        st.info("💡 Click **Run Comparison** above to load and compare year-over-year data.")

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
            options=["👥 Employee Insights", "👔 Manager Insights", "🔄 Combined Analysis"],
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
