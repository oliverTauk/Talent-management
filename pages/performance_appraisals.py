import io
import pandas as pd
import streamlit as st

from hr_analytics.ui.style import apply_global_style, page_header, section_title, divider

apply_global_style()
page_header("Check-ins", "KPIs • Analysis • Compare two years")


st.set_page_config(page_title="Performance Appraisals", layout="wide")

if st.button("⬅ Back to Home"):
    st.switch_page("Home.py")


st.title("Performance Appraisals (PA)")
st.caption("Upload PA export (Excel/CSV). Key = Employee Code. Year is derived from Appraisal Date.")

pa_file = st.file_uploader("Upload PA file (.xlsx/.xls/.csv)", type=["xlsx", "xls", "csv"])

def load_any(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = file.name.lower()
    data = file.read()
    file.seek(0)
    bio = io.BytesIO(data)
    if name.endswith(".csv"):
        return pd.read_csv(bio)
    return pd.read_excel(bio)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]  # remove trailing spaces
    return out

def detect_col(df: pd.DataFrame, candidates: list[str], contains: list[str] | None = None) -> str | None:
    cols = list(df.columns)
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand.lower().strip() in low:
            return low[cand.lower().strip()]
    if contains:
        for c in cols:
            cl = c.lower()
            if all(tok in cl for tok in contains):
                return c
    return None

def parse_year(df: pd.DataFrame, date_col: str) -> pd.Series:
    # dayfirst=True because your dates look like 28/10/2025
    dt = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    return dt.dt.year

def score_band(score: float) -> str:
    # 0–40 Unsatisfactory; >40–55 Needs Improvement; >55–70 Meets; >70–85 Exceeds; >85–100 Outstanding
    if pd.isna(score):
        return "Unknown"
    try:
        s = float(score)
    except Exception:
        return "Unknown"
    if s <= 40:
        return "Unsatisfactory (0–40)"
    if s <= 55:
        return "Needs Improvement (40–55]"
    if s <= 70:
        return "Meets Expectations (55–70]"
    if s <= 85:
        return "Exceeds Expectations (70–85]"
    return "Outstanding (85–100]"

if pa_file is None:
    st.info("Upload a PA file to begin.")
    st.stop()

df_raw = normalize_columns(load_any(pa_file))
if df_raw.empty:
    st.error("File looks empty or unreadable.")
    st.stop()

# Detect required columns (adapt if HR changes headers slightly)
emp_code_col = detect_col(df_raw, ["Employee Code"], contains=["employee", "code"])
date_col = detect_col(df_raw, ["Appraisal Date"], contains=["appraisal", "date"])
score_col = detect_col(df_raw, ["Final Score"], contains=["final", "score"])

missing = [k for k, v in {
    "Employee Code": emp_code_col,
    "Appraisal Date": date_col,
    "Final Score": score_col,
}.items() if v is None]

if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}. Please check the PA export headers.")
    st.write("Detected columns:", list(df_raw.columns))
    st.stop()

df = df_raw.copy()
df["Employee Code"] = df[emp_code_col].astype(str).str.strip()
df["Year"] = parse_year(df, date_col)
df["Final Score"] = pd.to_numeric(df[score_col], errors="coerce")
df["Score Band"] = df["Final Score"].apply(score_band)

st.subheader("Raw Preview")
st.dataframe(df_raw.head(25), use_container_width=True)

# Deduplicate to 1 row per employee-year (your PA has multiple objective rows per employee)
dedup = (
    df.sort_values(by=["Year"], kind="mergesort")
      .drop_duplicates(subset=["Employee Code", "Year"], keep="last")
      .reset_index(drop=True)
)

st.subheader("Deduped (1 row per Employee Code + Year)")
st.write(f"Rows: {len(dedup)} | Unique Employee Codes: {dedup['Employee Code'].nunique()}")
st.dataframe(dedup.head(50), use_container_width=True)

st.download_button(
    "Download Deduped PA CSV",
    dedup.to_csv(index=False).encode("utf-8"),
    file_name="pa_deduped.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("Distribution by Year")
dist = (
    dedup.groupby(["Year", "Score Band"], dropna=False)
         .size()
         .reset_index(name="Count")
         .sort_values(["Year", "Count"], ascending=[False, False])
)
st.dataframe(dist, use_container_width=True)

st.markdown("---")
st.subheader("Year-over-Year (latest two years)")

years = sorted([y for y in dedup["Year"].dropna().unique()])
if len(years) < 2:
    st.warning("Need at least two years in the PA file to compute YoY.")
    st.stop()

y_prev, y_curr = years[-2], years[-1]
a = dedup[dedup["Year"] == y_prev][["Employee Code", "Final Score", "Score Band"]].rename(
    columns={"Final Score": f"Score {y_prev}", "Score Band": f"Band {y_prev}"}
)
b = dedup[dedup["Year"] == y_curr][["Employee Code", "Final Score", "Score Band"]].rename(
    columns={"Final Score": f"Score {y_curr}", "Score Band": f"Band {y_curr}"}
)

yoy = b.merge(a, on="Employee Code", how="left")
yoy["Δ Score"] = yoy[f"Score {y_curr}"] - yoy.get(f"Score {y_prev}")
st.write(f"Comparing {y_prev} → {y_curr}")
st.dataframe(yoy.sort_values("Δ Score", ascending=False), use_container_width=True)

st.download_button(
    "Download YoY CSV",
    yoy.to_csv(index=False).encode("utf-8"),
    file_name=f"pa_yoy_{y_prev}_{y_curr}.csv",
    mime="text/csv",
)
