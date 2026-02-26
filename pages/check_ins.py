import io
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import pandas as pd
import streamlit as st


from hr_analytics.services.checkin_cleaner_excel import CheckInExcelCleaner
from hr_analytics.services.kpis import KPIService
from hr_analytics.services.normalizers import (
    _add_year_from_timestamp,
    _change_counts,
    _count_yes_no,
    _dept_options,
    _dynamics_counts,
    _extract_month_year_label,
    _filter_by_dept,
    _find_dept_col,
    _metrics,
    _mgr_dynamics_counts,
    _mistake_counts,
    _no_mask,
    _norm_behavior_opt,
    _norm_freq,
    _norm_mistake_opt,
    _norm_resource_opt,
    _norm_stress_freq,
    _norm_stress_reason,
    _norm_team_integration,
    _parse_any_date,
    _pct,
    _pulse_reason_counts,
    _recog_counts,
    _split_multiselect,
    _stress_reason_counts,
    _yes_mask,
    CHANGE_CATS,
    DYNAMICS_ORDER,
    FREQ_CATS,
    MGR_DYNAMICS_ORDER,
    MISTAKE_ORDER,
    NEG_TAGS,
    POS_TAGS,
    RECOG_ORDER,
    RES_ORDER,
    STRESS_FREQ_ORDER,
    STRESS_REASON_ORDER,
    UNDESIRED,
)
from hr_analytics.services.column_detection import (
    _find_aligned_goals_col,
    _find_checkin_freq_col_emp,
    _find_discussed_professional_goals_col,
    _find_checkin_freq_col_mgr,
    _find_col_by_keywords,
    _find_company_resources_col_emp,
    _find_dynamics_col_emp,
    _find_employee_name_col,
    _find_if_no_elaborate_col,
    _find_input_seek_col_emp,
    _find_input_seek_col_mgr,
    _find_job_change_types_col,
    _find_job_change_yn_col_emp,
    _find_job_change_yn_col_mgr,
    _find_manager_name_col_emp,
    _find_mgr_behavior_question_col_emp,
    _find_mgr_dynamics_col_mgr,
    _find_mgr_stress_freq_col,
    _find_pip_reason_no_col,
    _find_promo_position_col,
    _find_promo_review_date_col,
    _find_pulse_reason_col_emp,
    _find_pulse_yn_col_emp,
    _find_ready_for_promotion_col,
    _find_resources_other_text_col_emp,
    _find_responsibility_growth_col,
    _find_risk_reason_col,
    _find_stress_freq_col_emp,
    _find_stress_reason_col_emp,
    _find_stress_reasons_col,
    _find_subordinate_name_col,
    _find_team_integration_col_emp,
    _find_team_integration_col_mgr,
    _find_your_name_col,
    _norm,
    _scale_good_rate_from_keywords,
    _yes_rate_from_keywords,
)
from hr_analytics.ui.downloads import _download_df_csv, _download_fig_png
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



    # ✅ Force this exact file name (check Data/ and subdirectories)
    forced = base / "Data" / "Mena Report - Copy Tech.xlsx"
    if forced.exists():
        return _read_any(forced)
    # Also check subdirectories
    forced_sub = base / "Data" / "Check-ins 2024 2025" / "Mena Report - Copy Tech.xlsx"
    if forced_sub.exists():
        return _read_any(forced_sub)

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



def _run_comparison() -> None:
    """Run the YoY comparison pipeline and store results in session state.
    Called by the KPI Run button so both sections update together.
    """
    st.session_state.yoy_ready = False
    st.session_state.yoy_payload = None

    use_emp_y1 = st.session_state.get("compare_emp_y1")
    use_mgr_y1 = st.session_state.get("compare_mgr_y1")
    use_emp_y2 = st.session_state.get("compare_emp_y2")
    use_mgr_y2 = st.session_state.get("compare_mgr_y2")

    if not (use_emp_y1 and use_mgr_y1 and use_emp_y2 and use_mgr_y2):
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
        mgr1_c = cleaner.clean_manager_checkin(mgr1, df_mena, df_emp_cleaned=emp1_c)
        emp2_c = cleaner.clean_employee_checkin(emp2, df_mena)
        mgr2_c = cleaner.clean_manager_checkin(mgr2, df_mena, df_emp_cleaned=emp2_c)

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

    with st.expander("Upload & Filters", expanded=not st.session_state.clean_ready):

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

        if emp_file and mgr_file:
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
                    cleaned_mgr = cleaner.clean_manager_checkin(df_mgr, df_mena, df_emp_cleaned=cleaned_emp)
                    combined = cleaner.combine_cleaned(cleaned_emp, cleaned_mgr)

                    st.session_state.cleaned_emp = cleaned_emp
                    st.session_state.cleaned_mgr = cleaned_mgr
                    st.session_state.combined = combined
                    st.session_state.clean_ready = True

        # Also run YoY comparison so the Compare tab is ready
        if st.session_state.clean_ready:
            _run_comparison()

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

            # ── Alignment on Goals with Managers ──
            with st.expander("Alignment on Goals with Managers", expanded=False):
                st.caption(
                    "The following insights are related to Employees' feedback with regards to "
                    "their alignment on the overall department goals and their own professional "
                    "goals with their managers across the department"
                )

                _left_col, _right_col = st.columns(2)

                # --- LEFT: Alignment on department goals ---
                aligned_col = _find_aligned_goals_col(emp_f)
                with _left_col:
                    if not aligned_col:
                        st.info("Could not detect the alignment question column.")
                    else:
                        a_yes, a_no, a_yes_pct, a_no_pct = _count_yes_no(emp_f, aligned_col)
                        n_resp = a_yes + a_no
                        st.markdown(
                            f'<div style="background:#333;color:white;padding:10px 18px;border-radius:30px;'
                            f'text-align:center;font-size:0.95rem;">'
                            f'The response of <span style="color:#E53935;font-weight:700;">{n_resp}</span> '
                            f'employees on their alignment on department goals with their managers</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("")
                        st.markdown(
                            f'<span style="color:#E53935;font-weight:700;font-size:1.3rem;">{a_yes}</span> '
                            f'employees **are aligned** on the department\'s goals with their managers',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f'**Whereas**')
                        st.markdown(
                            f'<span style="color:#E53935;font-weight:700;font-size:1.3rem;">{a_no}</span> '
                            f'employees **are not aligned** on the department\'s goals with their managers',
                            unsafe_allow_html=True,
                        )

                # --- RIGHT: Discussed professional goals ---
                discuss_col = _find_discussed_professional_goals_col(emp_f)
                with _right_col:
                    if not discuss_col:
                        st.info("Could not detect the discussed professional goals question column.")
                    else:
                        d_yes, d_no, d_yes_pct, d_no_pct = _count_yes_no(emp_f, discuss_col)
                        n_resp_d = d_yes + d_no
                        st.markdown(
                            f'<div style="background:#E53935;color:white;padding:10px 18px;border-radius:30px;'
                            f'text-align:center;font-size:0.95rem;">'
                            f'The response of <span style="font-weight:700;">{n_resp_d}</span> '
                            f'employees on discussing their professional goals with their managers</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("")
                        st.markdown(
                            f'<span style="color:#E53935;font-weight:700;font-size:1.3rem;">{d_yes}</span> '
                            f'employees **discussed** their professional goals for this year with their managers',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f'**Whereas**')
                        st.markdown(
                            f'<span style="color:#E53935;font-weight:700;font-size:1.3rem;">{d_no}</span> '
                            f'employees **did not** discuss their professional goals for this year with their managers',
                            unsafe_allow_html=True,
                        )

            # ── Alignment of Responsibility with Professional Growth ──
            with st.expander("Alignment of Responsibility with Professional Growth", expanded=False):
                st.caption(
                    "The following insights are related to Employees' feedback with regards to "
                    "their alignment of their tasks and responsibilities with their desired "
                    "professional growth across the organization"
                )

                resp_growth_col = _find_responsibility_growth_col(emp_f)
                elab_col = _find_if_no_elaborate_col(emp_f)
                name_col_rg = _find_your_name_col(emp_f)

                if not resp_growth_col:
                    st.info("Could not detect the tasks/responsibilities aligned with professional growth question.")
                else:
                    rg_yes, rg_no, rg_yes_pct, rg_no_pct = _count_yes_no(emp_f, resp_growth_col)

                    _col_left, _col_right = st.columns([1, 2])

                    with _col_left:
                        st.markdown(
                            f'<div style="text-align:center;">'
                            f'<span style="color:#E53935;font-weight:700;font-size:2.2rem;">{rg_yes}</span>'
                            f'<span style="font-size:1.1rem;"> employees</span><br>'
                            f'<span style="font-weight:700;font-size:1.2rem;">Aligned</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("")
                        st.markdown(
                            f'<div style="text-align:center;">'
                            f'<span style="color:#E53935;font-weight:700;font-size:2.2rem;">{rg_no}</span>'
                            f'<span style="font-size:1.1rem;"> employees</span><br>'
                            f'<span style="font-weight:700;font-size:1.2rem;">Not Aligned</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    with _col_right:
                        # Show names + elaboration of employees who answered No
                        if rg_no > 0 and name_col_rg:
                            no_rows = emp_f[
                                emp_f[resp_growth_col]
                                .astype(str).str.strip().str.lower()
                                .str.startswith(("no", "n"))
                            ]
                            for _, row in no_rows.iterrows():
                                emp_name = str(row.get(name_col_rg, "")).strip()
                                reason = ""
                                if elab_col and elab_col in row.index:
                                    reason = str(row.get(elab_col, "")).strip()
                                    if reason.lower() in ("nan", "", "none"):
                                        reason = ""
                                if emp_name and emp_name.lower() not in ("nan", "", "none"):
                                    if reason:
                                        st.markdown(
                                            f'• <span style="color:#E53935;font-weight:600;">{emp_name}</span>'
                                            f' — mentioned "{reason}"',
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        st.markdown(
                                            f'• <span style="color:#E53935;font-weight:600;">{emp_name}</span>',
                                            unsafe_allow_html=True,
                                        )
                        elif rg_no > 0:
                            st.info("Employee name column not detected — cannot list names.")
                        else:
                            st.success("All employees are aligned.")

            # ── Additional Employee KPIs ──
            with st.expander("Additional Employee KPIs (keyword-based)", expanded=False):

                # Remaining KPIs in specified order
                cols = st.columns(4)
                
                # Job requirements changed
                rate, col = _yes_rate_from_keywords(emp_f, ["encountered", "changes", "job", "requirements"])
                cols[0].metric("Job requirements changed", _pct(rate) if col else "N/A")
                
                # Adapted well (4–5)
                rate, col = _scale_good_rate_from_keywords(emp_f, ["able", "adapt", "job", "requirements"], good_min=4)
                cols[1].metric("Adapted well (4–5)", _pct(rate) if col else "N/A")
                
                # Tasks aligned with growth
                rate, col = _yes_rate_from_keywords(emp_f, ["tasks", "aligned", "growth"])
                cols[2].metric("Tasks aligned with growth", _pct(rate) if col else "N/A")
                
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


            with st.expander("Managers Behaviors", expanded=False):
                st.caption(
                    "The following insights are related to Employees' feedback "
                    "with regards to their managers' behaviors"
                )

                # This slide uses ONLY employee check-in data
                q_col = _find_mgr_behavior_question_col_emp(emp_f)
                # Use "Mena Name" (canonical corrected name from cleaned check-in)
                emp_name_col_beh = _find_your_name_col(emp_f)

                if not q_col:
                    st.info("Could not detect the 'true statements about your manager' question column in employee data.")
                else:
                    # --- classify each employee row ---
                    def _classify_row(cell: str) -> tuple[bool, bool]:
                        opts = _split_multiselect(cell)
                        tags = [_norm_behavior_opt(o) for o in opts]
                        pos = any(t in POS_TAGS for t in tags)
                        neg = any(t in NEG_TAGS for t in tags)
                        return pos, neg

                    _flags = emp_f[q_col].astype(str).apply(_classify_row)
                    _has_pos = _flags.apply(lambda x: x[0])
                    _has_neg = _flags.apply(lambda x: x[1])

                    pos_only_mask = _has_pos & ~_has_neg
                    neg_only_mask = ~_has_pos & _has_neg
                    both_mask     = _has_pos & _has_neg

                    total_valid = int((_has_pos | _has_neg).sum()) or 1
                    pos_pct  = round(int(pos_only_mask.sum()) / total_valid * 100)
                    neg_pct  = round(int(neg_only_mask.sum()) / total_valid * 100)
                    both_pct = round(int(both_mask.sum()) / total_valid * 100)

                    # --- Venn diagram with bullet text ---
                    fig, ax = plt.subplots(figsize=(12, 7.5))
                    fig.patch.set_facecolor("white")
                    ax.set_xlim(-1, 11)
                    ax.set_ylim(-1.8, 6.2)
                    ax.axis("off")

                    # Left circle (positive-only) — gray
                    ax.add_patch(Circle((3.8, 3.4), 2.0,
                                        facecolor="#555555", edgecolor="#333333",
                                        alpha=0.30, linewidth=1.5))
                    # Right circle (negative-only) — red
                    ax.add_patch(Circle((6.2, 3.4), 2.0,
                                        facecolor="#E53935", edgecolor="#C62828",
                                        alpha=0.30, linewidth=1.5))

                    # Percentage labels inside circles
                    ax.text(3.0, 3.5, f"{pos_pct}%", fontsize=20, fontweight="bold",
                            ha="center", va="center", color="#333333")
                    ax.text(5.0, 3.5, f"{both_pct}%", fontsize=18, fontweight="bold",
                            ha="center", va="center", color="#333333")
                    ax.text(7.0, 3.5, f"{neg_pct}%", fontsize=20, fontweight="bold",
                            ha="center", va="center", color="#E53935")

                    # Dots and lines connecting circles to bullet areas
                    ax.plot(2.0, 5.0, "o", color="#333333", markersize=6)
                    ax.plot([2.0, 3.0], [5.0, 4.4], color="#333333", linewidth=1)
                    ax.plot(8.0, 5.0, "o", color="#333333", markersize=6)
                    ax.plot([8.0, 7.0], [5.0, 4.4], color="#333333", linewidth=1)
                    ax.plot(5.0, 1.3, "o", color="#333333", markersize=6)
                    ax.plot([5.0, 5.0], [1.3, 1.8], color="#333333", linewidth=1)

                    # --- LEFT bullet text: positive (good) behaviors ---
                    ax.text(-0.8, 4.8, "Managers who", fontsize=10, fontweight="bold",
                            va="top", color="#333333")
                    pos_lines = [
                        "Provide employees with clear\n  guidance to fulfill their roles",
                        "Provide employees with continuous\n  feedback on their performance",
                        "Provide employees with recognition\n  for their work accomplishments",
                        "Offer employees opportunities\n  to develop and grow",
                    ]
                    y_pos = 4.2
                    for line in pos_lines:
                        ax.text(-0.8, y_pos, f"•  {line}", fontsize=8.5, va="top",
                                color="#333333", linespacing=1.3)
                        y_pos -= 0.95

                    # --- RIGHT bullet text: negative (bad) behaviors ---
                    ax.text(8.2, 4.8, "Managers who", fontsize=10, fontweight="bold",
                            va="top", color="#333333")
                    neg_lines = [
                        "Provide employees with insufficient\n  information to accomplish their duties",
                        "Provide employees with insufficient\n  feedback on their performance",
                        "Rarely give employees credit for\n  showing good performance",
                        "Rarely give employees chances\n  to advance and improve",
                    ]
                    y_neg = 4.2
                    for line in neg_lines:
                        ax.text(8.2, y_neg, f"•  {line}", fontsize=8.5, va="top",
                                color="#333333", linespacing=1.3)
                        y_neg -= 0.95

                    # --- MIDDLE: "In Between" label with names ---
                    if emp_name_col_beh and emp_name_col_beh in emp_f.columns:
                        _both_names_img = (
                            emp_f.loc[both_mask, emp_name_col_beh]
                            .astype(str).str.strip()
                            .loc[lambda s: s.str.lower().apply(
                                lambda v: v not in ("", "nan", "none")
                            )]
                            .tolist()
                        )
                        _both_names_img = sorted(set(_both_names_img))
                    else:
                        _both_names_img = []

                    ax.text(5.0, 0.9, "In Between", fontsize=11, fontweight="bold",
                            ha="center", va="top", color="#E53935")
                    if _both_names_img:
                        mid_y = 0.4
                        for nm in _both_names_img:
                            ax.text(5.0, mid_y, f"•  {nm}", fontsize=8.5,
                                    ha="center", va="top", color="#E53935")
                            mid_y -= 0.35

                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=False)
                    _download_fig_png(fig, "slide12_managers_behaviors.png", key="dl_slide12_behaviors_png")

                    # --- Cases to review ---
                    st.markdown("---")
                    st.markdown(
                        '<span style="color:#E53935;font-weight:700;font-size:1.1rem;">'
                        '⚠ Cases to Review</span>',
                        unsafe_allow_html=True,
                    )

                    if emp_name_col_beh and emp_name_col_beh in emp_f.columns:
                        _name_series = emp_f[emp_name_col_beh].astype(str).str.strip()
                        _valid = _name_series.str.lower().apply(
                            lambda v: v not in ("", "nan", "none")
                        )

                        # Employees who checked ONLY bad comments
                        neg_names = sorted(set(
                            _name_series.loc[neg_only_mask & _valid].tolist()
                        ))
                        # Employees who checked both bad and good comments
                        both_names_review = sorted(set(
                            _name_series.loc[both_mask & _valid].tolist()
                        ))

                        rev_left, rev_right = st.columns(2)

                        with rev_left:
                            st.markdown("**Employees who selected only negative statements**")
                            if neg_names:
                                for nm in neg_names:
                                    st.markdown(
                                        f'• <span style="color:#E53935;font-weight:600;">{nm}</span>',
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.caption("None")

                        with rev_right:
                            st.markdown("**Employees who selected both positive & negative statements**")
                            if both_names_review:
                                for nm in both_names_review:
                                    st.markdown(
                                        f'• <span style="color:#E53935;font-weight:600;">{nm}</span>',
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.caption("None")
                    else:
                        st.caption("Employee name column not detected — cannot list names.")


            with st.expander("Work Culture and Environment — Supportive Work Environment", expanded=False):
                left, right = st.columns(2)

                # --- Left: Supportive Work Environment Yes/No KPI ---
                with left:
                    st.markdown("<b>Q: Does the company’s culture foster a collaborative and supporting work environment?</b>", unsafe_allow_html=True)
                    env_col = None
                    for c in emp_f.columns:
                        h = _norm(c)
                        if ("foster" in h) and ("collabor" in h) and ("support" in h) and ("work environment" in h):
                            env_col = c
                            break
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
                        st.markdown(f"<div style='display:flex; gap:2em; margin-bottom:0.5em;'>"
                                    f"<div style='font-size:1.1rem;'><b>YES</b><br><span style='font-size:1.5rem;color:#388e3c'>{yes_count}</span></div>"
                                    f"<div style='font-size:1.1rem;'><b>NO</b><br><span style='font-size:1.5rem;color:#d32f2f'>{no_count}</span></div>"
                                    f"</div>", unsafe_allow_html=True)
                        # Names of employees who answered NO
                        if no_count > 0:
                            name_col = _find_your_name_col(emp_f)
                            if name_col and name_col in emp_f.columns:
                                no_names = emp_f.loc[no_mask, name_col].astype(str).fillna("").str.strip()
                                no_names = [n for n in no_names.tolist() if n]
                                if no_names:
                                    st.caption("**Employees who answered NO:** " + ", ".join(no_names))
                            else:
                                st.caption("Employee name column not found to list NO responders.")

                # --- Right: Work Culture Multi-select: Top 3 ---
                with right:
                    st.markdown("<b>Q: Check the three most relevant boxes that describe the company's work culture</b>", unsafe_allow_html=True)
                    culture_col = None
                    for c in emp_f.columns:
                        h = _norm(c)
                        if ("work culture" in h) and ("relevant" in h or "describe" in h):
                            culture_col = c
                            break
                    if culture_col:
                        # Parse all selected options (multi-select, comma/semicolon/pipe separated)
                        all_opts = []
                        for v in emp_f[culture_col].dropna().astype(str):
                            all_opts.extend(_split_multiselect(v))
                        # Count frequency
                        from collections import Counter
                        counts = Counter([o.strip() for o in all_opts if o.strip()])
                        top3 = counts.most_common(3)
                        if top3:
                            st.markdown("**Top 3 most selected work cultures:**")
                            for i, (opt, cnt) in enumerate(top3, 1):
                                st.markdown(f"{i}. <b>{opt}</b> <span style='color:#888'>(selected {cnt} times)</span>", unsafe_allow_html=True)
                        else:
                            st.caption("No work culture options selected.")
                    else:
                        st.caption("Could not detect the work culture multi-select question in employee data.")

            with st.expander("Enhancing Department Dynamics — Employees", expanded=False):
                dyn_col = _find_dynamics_col_emp(emp_f)
                if not dyn_col:
                    st.info("Could not detect the 'department dynamics need enhancement' multi-select question in employee data.")
                else:
                    counts = _dynamics_counts(emp_f[dyn_col])

                    labels = DYNAMICS_ORDER
                    values = [int(counts.get(k, 0)) for k in labels]
                    y = np.arange(len(labels))

                    # Modern color palette
                    palette = ["#90caf9", "#a5d6a7", "#ffb74d", "#f48fb1", "#ce93d8", "#fff176", "#80cbc4", "#b0bec5"]
                    colors = palette[:len(labels)]

                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    bars = ax.barh(y, values, color=colors, edgecolor="#eee", height=0.65)
                    ax.set_yticks(y)
                    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
                    ax.invert_yaxis()
                    ax.set_xlabel("Number of Employees", fontsize=12)
                    ax.set_title("Enhancing Department Dynamics — Employees", fontsize=14, fontweight="bold", pad=16)
                    ax.xaxis.grid(True, linestyle="--", color="#bbb", alpha=0.5)
                    ax.set_axisbelow(True)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_visible(False)
                    ax.spines["bottom"].set_color("#bbb")

                    # Value labels at end of bars
                    for i, v in enumerate(values):
                        ax.text(v + 0.3, y[i], str(int(v)), va="center", fontsize=13, fontweight="bold", color="#333")

                    st.pyplot(fig, clear_figure=False)
                    _download_fig_png(fig, "slide21_employee_dynamics.png", key="dl_slide21_emp_dyn_png")


            with st.expander("Employee Stress", expanded=False):
                st.caption(
                    "_Stress is the body's natural response to challenging situations, "
                    "characterized by physical, emotional, or mental strain.  \n"
                    "The employee could only choose one of the three options: "
                    "Extremely frequent · Frequent · Less frequent._"
                )

                tab1, tab2 = st.tabs(["Frequency", "Reasons (Who mentioned what)"])

                # -------------------------
                # Tab 1: Stress Frequency
                # -------------------------
                with tab1:
                    freq_col = _find_stress_freq_col_emp(emp_f)

                    if not freq_col:
                        st.info("Could not detect the employee stress frequency question in employee data.")
                    else:
                        tmp = emp_f.copy()
                        tmp["_stress_freq"] = tmp[freq_col].astype(str).map(_norm_stress_freq)

                        counts = tmp["_stress_freq"].value_counts().reindex(STRESS_FREQ_ORDER, fill_value=0)

                        c1, c2, c3 = st.columns(3)

                        _freq_colors = {
                            "Extremely frequent": "#d62728",
                            "Frequent": "#ff7f0e",
                            "Less frequent": "#2ca02c",
                        }

                        for col_ui, label in zip([c1, c2, c3], STRESS_FREQ_ORDER):
                            with col_ui:
                                _clr = _freq_colors.get(label, "#888")
                                st.markdown(
                                    f"<div style='text-align:center; border:1px solid {_clr}; "
                                    f"border-radius:8px; padding:12px; margin:4px 0;'>"
                                    f"<span style='font-size:0.85rem; font-weight:600; color:{_clr};'>{label}</span><br>"
                                    f"<span style='font-size:1.8rem; font-weight:700;'>{int(counts[label])}</span><br>"
                                    f"<span style='font-size:0.75rem; color:#aaa;'>employees</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                # -------------------------
                # Tab 2: Reasons table
                # -------------------------
                with tab2:
                    reason_col = _find_stress_reasons_col(emp_f)
                    name_col = _find_your_name_col(emp_f)
                    freq_col_t2 = _find_stress_freq_col_emp(emp_f)

                    if not reason_col:
                        st.info("Could not detect the stress reasons multi-select question in employee data.")
                    elif not name_col or name_col not in emp_f.columns:
                        st.info("Employee name column not found in employee data.")
                    elif not freq_col_t2:
                        st.info("Could not detect the stress frequency column.")
                    else:
                        # Only include employees who chose Extremely frequent or Frequent
                        tmp2 = emp_f.copy()
                        tmp2["_stress_freq"] = tmp2[freq_col_t2].astype(str).map(_norm_stress_freq)
                        stressed_mask = tmp2["_stress_freq"].isin(["Extremely frequent", "Frequent"])
                        stressed_df = tmp2[stressed_mask]

                        if stressed_df.empty:
                            st.caption("No employees chose Extremely frequent or Frequent.")
                        else:
                            reason_to_names: dict[str, list[tuple[str, str]]] = {k: [] for k in STRESS_REASON_ORDER}

                            for _, r in stressed_df.iterrows():
                                emp_name = str(r[name_col]).strip()
                                freq_label = str(r["_stress_freq"]).strip()
                                if not emp_name or emp_name == "nan":
                                    continue

                                picks = _split_multiselect(r[reason_col])
                                for p in picks:
                                    k = _norm_stress_reason(p)
                                    if k:
                                        reason_to_names[k].append((emp_name, freq_label))

                            rows = []
                            for reason in STRESS_REASON_ORDER:
                                entries = reason_to_names.get(reason, [])
                                if not entries:
                                    continue
                                # Deduplicate by name, keep frequency
                                seen: dict[str, str] = {}
                                for n, f in entries:
                                    seen[n] = f
                                name_strs = sorted(
                                    [f"{n} ({f})" for n, f in seen.items()]
                                )
                                rows.append({
                                    "Reason": reason,
                                    "Employees (frequency level)": ", ".join(name_strs),
                                })

                            if rows:
                                df_out = pd.DataFrame(rows)
                                st.dataframe(df_out, use_container_width=True, hide_index=True)
                                _download_df_csv(df_out, "stress_reasons_who_mentioned.csv", key="dl_stress_reasons_table")
                            else:
                                st.caption("No reasons were selected by stressed employees.")


            with st.expander("Pulse-Check Meeting", expanded=False):
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

            with st.expander("Company Resources", expanded=False):

                # Show the question text
                st.markdown("""
                <b>Q: What company resources or practices do I use to ease and facilitate my experience within the work culture?</b><br>
                <ul style='margin-top:0;margin-bottom:0;'>
                  <li>Enrolling in the group's wellness sessions (yoga, fitness, workshops, ...)</li>
                  <li>Seeking advice from HR (pulse check, one-on-one meetings, discussions, informal/formal requests, ...)</li>
                  <li>Expressing complaints or miscellaneous ideas (anonymous reporting, open door policy, ...)</li>
                  <li>Other</li>
                </ul>
                """, unsafe_allow_html=True)

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
                        picks = _split_multiselect(r.get(res_col, ""))
                        norm_picks = []
                        for p in picks:
                            k = _norm_resource_opt(p)
                            if k:
                                norm_picks.append(k)
                        for k in norm_picks:
                            counts[k] += 1
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

                def _mgr_kpi_card(container, title: str, count: int, pct: float, detected: bool, col_name: str | None = None):
                    """Render a compact KPI card: title on top, count left / pct right."""
                    if not detected:
                        container.markdown(
                            f"<div style='border:1px solid #555; border-radius:8px; padding:10px; margin:4px 0; text-align:center;'>"
                            f"<span style='font-size:0.78rem; font-weight:600;'>{title}</span><br>"
                            f"<span style='font-size:0.85rem; color:#aaa;'>N/A</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        return
                    container.markdown(
                        f"<div style='border:1px solid #555; border-radius:8px; padding:10px; margin:4px 0;'>"
                        f"<div style='text-align:center; font-size:0.78rem; font-weight:600; margin-bottom:6px;'>{title}</div>"
                        f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                        f"<span style='font-size:1.4rem; font-weight:700;'>{count}</span>"
                        f"<span style='font-size:1.1rem; color:#aaa;'>{pct:.1f}%</span>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                cols = st.columns(4)
                
                # 1. Encountered changes in job requirements
                rate, col = _yes_rate_from_keywords(mgr_f, ["encountered", "changes", "job", "requirements"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                _mgr_kpi_card(cols[0], "Encountered job changes", yes_count, yes_pct, col is not None, col)
                
                # 2. At risk of low performance
                rate, col = _yes_rate_from_keywords(mgr_f, ["at risk", "low performance"])
                if not col:
                    rate, col = _yes_rate_from_keywords(mgr_f, ["risk", "performance"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                _mgr_kpi_card(cols[1], "At risk of low performance", yes_count, yes_pct, col is not None, col)
                
                # 3. Would perform better in another department
                rate, col = _yes_rate_from_keywords(mgr_f, ["perform better", "another department"])
                if not col:
                    rate, col = _yes_rate_from_keywords(mgr_f, ["better", "department"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                _mgr_kpi_card(cols[2], "Better in another dept", yes_count, yes_pct, col is not None, col)
                
                # 4. Ready for promotion today
                rate, col = _yes_rate_from_keywords(mgr_f, ["ready", "promotion"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                _mgr_kpi_card(cols[3], "Ready for promotion", yes_count, yes_pct, col is not None, col)
                
                # Second row
                cols2 = st.columns(4)
                
                # 5. Actively seek and consider team members' input
                rate, col = _yes_rate_from_keywords(mgr_f, ["actively seek", "team members", "input"])
                if not col:
                    rate, col = _yes_rate_from_keywords(mgr_f, ["seek", "consider", "input"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                _mgr_kpi_card(cols2[0], "Seeks team input", yes_count, yes_pct, col is not None, col)
                
                # 6. Fits in company culture
                rate, col = _yes_rate_from_keywords(mgr_f, ["fits", "company culture"])
                if not col:
                    rate, col = _yes_rate_from_keywords(mgr_f, ["person", "fits", "culture"])
                yes_count, no_count, yes_pct, no_pct = _count_yes_no(mgr_f, col) if col else (0, 0, 0, 0)
                _mgr_kpi_card(cols2[1], "Fits in company culture", yes_count, yes_pct, col is not None, col)

                # 7. Manager Stress rate
                # Calculate: percent of managers who answered Frequent or Extremely frequent
                mgr_stress_col = _find_mgr_stress_freq_col(mgr_f)
                if mgr_stress_col:
                    normed = mgr_f[mgr_stress_col].astype(str).map(_norm_stress_freq)
                    mask = normed.isin(["Frequent", "Extremely frequent"])
                    mgr_stress_count = int(mask.sum())
                    mgr_stress_pct = 100.0 * mgr_stress_count / len(mgr_f) if len(mgr_f) > 0 else 0.0
                    _mgr_kpi_card(cols2[2], "Manager Stress rate", mgr_stress_count, mgr_stress_pct, True)
                else:
                    _mgr_kpi_card(cols2[2], "Manager Stress rate", 0, 0.0, False)

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
                            key="dl_at_risk_table")


            # --- Manager insights — Promotion ---
            with st.expander("Manager Insights — Promotion", expanded=False):
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

            
            
            with st.expander("Employee – Company Culture Fit (Manager)", expanded=False):
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


            with st.expander("Employee Contribution to the Department Dynamics — Managers", expanded=False):
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


            with st.expander("Stress Frequency — Managers", expanded=False):
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

            with st.expander("Changes in Job Responsibilities (Employee vs Manager)", expanded=False):

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

            with st.expander("Adapting to Change (Employee vs Manager)", expanded=False):

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

            with st.expander("Check-ins Meetings Between Managers and Employees", expanded=False):
                st.caption(
                    "The following percentages reflect the answers of both managers and employees "
                    "when asked about the frequency of their check-in meetings"
                )

                emp_freq_col = _find_checkin_freq_col_emp(emp_f)
                mgr_freq_col = _find_checkin_freq_col_mgr(mgr_f)

                emp_name_col_freq = _find_your_name_col(emp_f) or _find_employee_name_col(emp_f)
                mgr_sub_col_freq  = _find_subordinate_name_col(mgr_f)

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

                    # ---------- Vertical grouped bar chart ----------
                    x = np.arange(len(FREQ_CATS))
                    width = 0.32

                    fig, ax = plt.subplots(figsize=(8, 5))
                    fig.patch.set_facecolor("white")
                    ax.set_facecolor("white")

                    bars_mgr = ax.bar(
                        x - width / 2, mgr_pct.values, width,
                        label="Managers' Perspective", color="#E53935", edgecolor="white", linewidth=0.5,
                    )
                    bars_emp = ax.bar(
                        x + width / 2, emp_pct.values, width,
                        label="Employees' Perspective", color="#333333", edgecolor="white", linewidth=0.5,
                    )

                    # Value labels on top of each bar
                    for bar in bars_mgr:
                        h = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                                f"{int(round(h))}%", ha="center", va="bottom",
                                fontsize=9, fontweight="bold", color="#E53935")
                    for bar in bars_emp:
                        h = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                                f"{int(round(h))}%", ha="center", va="bottom",
                                fontsize=9, fontweight="bold", color="#333333")

                    ax.set_xticks(x)
                    ax.set_xticklabels(FREQ_CATS, fontsize=11, fontweight="bold")
                    ax.set_ylabel("% of responses", fontsize=10)
                    ax.set_ylim(0, max(max(mgr_pct.values), max(emp_pct.values)) + 12)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.legend(loc="upper right", frameon=False, fontsize=9)
                    fig.tight_layout()

                    st.pyplot(fig, clear_figure=False)
                    _download_fig_png(fig, "checkins_meeting_frequency.png", key="dl_meeting_freq_png")

                    # ---------- < One meeting per month ----------
                    divider()
                    st.markdown("**The following members stated that they have < One meeting per month**")

                    if emp_name_col_freq and mgr_sub_col_freq:
                        # Build lookups
                        emp_map = {}
                        for _idx, _row in emp_f.iterrows():
                            _nm = str(_row.get(emp_name_col_freq, "")).strip()
                            if _nm and _nm.lower() not in ("", "nan", "none"):
                                emp_map[_nm] = _norm_freq(str(_row.get(emp_freq_col, "")))

                        mgr_map = {}
                        for _idx, _row in mgr_f.iterrows():
                            _sub = str(_row.get(mgr_sub_col_freq, "")).strip()
                            if _sub and _sub.lower() not in ("", "nan", "none"):
                                mgr_map[_sub] = _norm_freq(str(_row.get(mgr_freq_col, "")))

                        common = sorted(set(emp_map.keys()) & set(mgr_map.keys()))
                        low_cats = {"Few", "Zero"}
                        low_both = []
                        for nm in common:
                            if emp_map.get(nm) in low_cats and mgr_map.get(nm) in low_cats:
                                low_both.append(nm)

                        if not low_both:
                            st.markdown(
                                '\u2022 <span style="color:#E53935;">No employee</span> and their manager '
                                'reported having less than a meeting per month.',
                                unsafe_allow_html=True,
                            )
                        else:
                            for nm in low_both:
                                st.markdown(
                                    f'\u2022 <span style="color:#E53935;font-weight:600;">{nm}</span> '
                                    f'and their manager both reported having less than a meeting per month '
                                    f'(Employee: **{emp_map[nm]}**, Manager: **{mgr_map[nm]}**).',
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.caption("Name column not detected \u2014 cannot list members.")

            with st.expander("Reward & Recognition (Employee vs Manager)", expanded=False):

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



            with st.expander("Employee Input in the Department (Employee vs Manager)", expanded=False):

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


            with st.expander("Ways to Address Mistakes (Employee vs Manager)", expanded=False):

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

            with st.expander("Employee Integration within the Team (Employee vs Manager)", expanded=False):

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

    with st.expander("Upload", expanded=not st.session_state.yoy_ready):

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

    if not st.session_state.yoy_ready:
        st.info("💡 Go to the **KPIs** section and click **Run** to load both KPIs and the year-over-year comparison.")

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
