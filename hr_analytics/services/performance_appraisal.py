from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd



@dataclass
class PADedupeWarning:
    employee_code: str
    year: int
    final_score_min: float
    final_score_max: float
    n_rows: int


class PerformanceAppraisalService:
    """
    Performance Appraisal (PA) helpers for this HR Analytics workspace.

    Important repo conventions (per user requirements):
    - PA is analyzed STANDALONE (no joining to check-ins).
    - Primary key for PA analytics: Employee Code (+ Year derived from Appraisal Date).
    - Year-over-year (YoY) comparisons happen within PA only.
    """

    # ----------------------------
    # Column detection / cleaning
    # ----------------------------
    @staticmethod
    def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        return out

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: List[str], *, contains: Optional[List[str]] = None) -> Optional[str]:
        """
        Find a column by exact match (case-insensitive) after stripping,
        optionally falling back to "contains" tokens.
        """
        cols = list(df.columns)
        norm = {str(c).strip().lower(): c for c in cols}

        for cand in candidates:
            key = cand.strip().lower()
            if key in norm:
                return norm[key]

        if contains:
            for c in cols:
                cl = str(c).strip().lower()
                if all(tok in cl for tok in contains):
                    return c
        return None

    # ----------------------------
    # Public API
    # ----------------------------
    def clean_pa_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean PA export (from HR) without changing the meaning:
        - Strip header whitespace
        - Normalize Employee Code to string
        - Parse Appraisal Date and add Year
        - Coerce Final Score to numeric if present

        Returns a NEW DataFrame.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df = self._strip_columns(df)

        emp_code_col = self._find_col(df, ["Employee Code"], contains=["employee", "code"])
        if not emp_code_col:
            raise ValueError("PA file must contain an 'Employee Code' column (or a close variant).")

        # Date
        date_col = self._find_col(df, ["Appraisal Date"], contains=["appraisal", "date"])
        if not date_col:
            raise ValueError("PA file must contain an 'Appraisal Date' column (or a close variant).")

        out = df.copy()

        # Employee Code: keep as string for stable grouping
        out[emp_code_col] = (
            out[emp_code_col]
            .astype(str)
            .str.strip()
            .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
        )

        # Parse dates (your sample is dd/mm/yyyy)
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce", dayfirst=True)
        out["Year"] = out[date_col].dt.year

        # Final score (percentage 0-100)
        score_col = self._find_col(df, ["Final Score"], contains=["final", "score"])
        if score_col and score_col in out.columns:
            out[score_col] = pd.to_numeric(out[score_col], errors="coerce")

        return out

    def score_band(self, score: Any) -> str:
        """
        Score bands per user definition:
          0 → 40: Unsatisfactory Performance
          > 40 → 55: Needs Improvement
          > 55 → 70: Meets Expectations
          > 70 → 85: Exceeds Expectations
          > 85 → 100: Outstanding
        """
        try:
            x = float(score)
        except Exception:
            return "Unknown"

        if pd.isna(x):
            return "Unknown"

        if x <= 40:
            return "Unsatisfactory"
        if 40 < x <= 55:
            return "Needs Improvement"
        if 55 < x <= 70:
            return "Meets Expectations"
        if 70 < x <= 85:
            return "Exceeds Expectations"
        if 85 < x <= 100:
            return "Outstanding"
        return "Out of Range"

    def add_score_band_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        score_col = self._find_col(df, ["Final Score"], contains=["final", "score"])
        if not score_col or score_col not in df.columns:
            df["Score Band"] = "Unknown"
            return df
        df["Score Band"] = df[score_col].apply(self.score_band)
        return df

    def dedupe_employee_year(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[PADedupeWarning]]:
        """
        Many PA exports are at the Objective/KPI line level, meaning employees repeat many rows.
        For dashboards / YoY, we need one row per (Employee Code, Year).

        Aggregation strategy:
        - For most fields: take first non-null
        - For Final Score: validate it's stable; if not, take mean and emit warning

        Returns (deduped_df, warnings)
        """
        if df is None or df.empty:
            return pd.DataFrame(), []

        df = self._strip_columns(df)

        emp_code_col = self._find_col(df, ["Employee Code"], contains=["employee", "code"])
        date_year_col = "Year" if "Year" in df.columns else None
        if not emp_code_col or not date_year_col:
            raise ValueError("PA df must include Employee Code and Year columns. Run clean_pa_export first.")

        score_col = self._find_col(df, ["Final Score"], contains=["final", "score"])

        # Helper: first non-null
        def first_non_null(s: pd.Series) -> Any:
            s2 = s.dropna()
            return s2.iloc[0] if len(s2) else pd.NA

        agg: Dict[str, Any] = {c: first_non_null for c in df.columns if c not in (emp_code_col, date_year_col)}
        # We'll handle score separately (still via agg but with validation)
        if score_col and score_col in agg:
            agg.pop(score_col, None)

        grouped = df.groupby([emp_code_col, date_year_col], dropna=False)

        warnings: List[PADedupeWarning] = []

        base = grouped.agg(agg).reset_index()

        if score_col and score_col in df.columns:
            score_stats = grouped[score_col].agg(["min", "max", "mean", "count"]).reset_index()
            # If min != max (within tolerance), warn and use mean
            tol = 1e-9
            score_stats["Final Score"] = score_stats["mean"]
            for _, r in score_stats.iterrows():
                mn, mx = r["min"], r["max"]
                if pd.notna(mn) and pd.notna(mx) and abs(float(mn) - float(mx)) > tol:
                    warnings.append(
                        PADedupeWarning(
                            employee_code=str(r[emp_code_col]),
                            year=int(r[date_year_col]) if pd.notna(r[date_year_col]) else -1,
                            final_score_min=float(mn),
                            final_score_max=float(mx),
                            n_rows=int(r["count"]),
                        )
                    )
            base = base.merge(score_stats[[emp_code_col, date_year_col, "Final Score"]], on=[emp_code_col, date_year_col], how="left")
            # Put the score column name back as original score column if it differs
            if score_col != "Final Score":
                base = base.rename(columns={"Final Score": score_col})

        # Add Score Band
        base = self.add_score_band_column(base)

        return base, warnings

    # ----------------------------
    # Analytics helpers
    # ----------------------------
    def year_distribution(self, deduped: pd.DataFrame) -> pd.DataFrame:
        """Count employees by Score Band per year."""
        if deduped is None or deduped.empty:
            return pd.DataFrame()
        if "Year" not in deduped.columns:
            return pd.DataFrame()
        if "Score Band" not in deduped.columns:
            deduped = self.add_score_band_column(deduped)

        pivot = (
            deduped
            .pivot_table(index="Year", columns="Score Band", values=deduped.columns[0], aggfunc="count", fill_value=0)
            .reset_index()
        )
        # nicer ordering if present
        ordered = ["Unsatisfactory", "Needs Improvement", "Meets Expectations", "Exceeds Expectations", "Outstanding", "Unknown", "Out of Range"]
        cols = ["Year"] + [c for c in ordered if c in pivot.columns] + [c for c in pivot.columns if c not in {"Year", *ordered}]
        return pivot[cols]

    def yoy_employee_delta(self, deduped: pd.DataFrame, year_prev: int, year_curr: int) -> pd.DataFrame:
        """
        Return a per-employee YoY comparison table between two years.
        """
        if deduped is None or deduped.empty:
            return pd.DataFrame()

        deduped = self._strip_columns(deduped)
        emp_code_col = self._find_col(deduped, ["Employee Code"], contains=["employee", "code"])
        score_col = self._find_col(deduped, ["Final Score"], contains=["final", "score"])

        if not emp_code_col or not score_col or "Year" not in deduped.columns:
            return pd.DataFrame()

        a = deduped[deduped["Year"] == year_prev][[emp_code_col, score_col, "Score Band"]].rename(
            columns={score_col: f"Score {year_prev}", "Score Band": f"Band {year_prev}"}
        )
        b = deduped[deduped["Year"] == year_curr][[emp_code_col, score_col, "Score Band"]].rename(
            columns={score_col: f"Score {year_curr}", "Score Band": f"Band {year_curr}"}
        )
        m = a.merge(b, on=emp_code_col, how="outer")

        # Delta
        m["Delta"] = pd.to_numeric(m.get(f"Score {year_curr}"), errors="coerce") - pd.to_numeric(m.get(f"Score {year_prev}"), errors="coerce")
        def direction(x):
            if pd.isna(x): return "Unknown"
            if x > 0: return "Improved"
            if x < 0: return "Declined"
            return "No Change"
        m["Direction"] = m["Delta"].apply(direction)
        return m.sort_values(by=["Direction", emp_code_col], ascending=[True, True])

    def manager_bias_summary(self, deduped: pd.DataFrame) -> pd.DataFrame:
        """
        Simple, explainable indicators (not accusations):
        - avg final score per appraiser
        - count of employees
        Requires an 'Appraiser' column.
        """
        if deduped is None or deduped.empty:
            return pd.DataFrame()

        deduped = self._strip_columns(deduped)
        score_col = self._find_col(deduped, ["Final Score"], contains=["final", "score"])
        app_col = self._find_col(deduped, ["Appraiser"], contains=["appraiser"])
        if not score_col or not app_col:
            return pd.DataFrame()

        g = deduped.groupby(app_col, dropna=False)[score_col].agg(["count", "mean", "min", "max"]).reset_index()
        g = g.rename(columns={"count": "Employees", "mean": "Avg Score", "min": "Min Score", "max": "Max Score"})
        return g.sort_values(by="Avg Score", ascending=False)

    def department_bias_summary(self, deduped: pd.DataFrame) -> pd.DataFrame:
        """
        Dept-level summary if a Department column exists.
        """
        if deduped is None or deduped.empty:
            return pd.DataFrame()

        deduped = self._strip_columns(deduped)
        score_col = self._find_col(deduped, ["Final Score"], contains=["final", "score"])
        dept_col = self._find_col(deduped, ["Department"], contains=["department"])
        if not score_col or not dept_col:
            return pd.DataFrame()

        g = deduped.groupby(dept_col, dropna=False)[score_col].agg(["count", "mean", "min", "max"]).reset_index()
        g = g.rename(columns={"count": "Employees", "mean": "Avg Score", "min": "Min Score", "max": "Max Score"})
        return g.sort_values(by="Avg Score", ascending=False)
