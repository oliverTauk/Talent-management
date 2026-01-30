from __future__ import annotations
from typing import Iterable, Dict, Any, Optional, Tuple

import pandas as pd


class KPIService:
    """Compute core KPIs: completion rate, stress %, at-risk %.

    Conventions: boolean-like fields are matched on {"yes","true","1"} case-insensitively.
    Also provides DataFrame helpers for deduplication by Timestamp and completion counts.
    """

    # --------------------
    # Generic boolean KPIs
    # --------------------
    def _is_true(self, v: Any) -> bool:
        return str(v).strip().lower() in {"yes", "true", "1"}

    def completion_rate(self, rows: Iterable[Dict[str, Any]], field: str) -> float:
        rows = list(rows or [])
        if not rows:
            return 0.0
        return sum(1 for r in rows if self._is_true(r.get(field))) / len(rows)

    def percentage(self, rows: Iterable[Dict[str, Any]], field: str) -> float:
        return self.completion_rate(rows, field)

    def at_risk_rate(self, rows: Iterable[Dict[str, Any]], field: str = "at_risk") -> float:
        return self.completion_rate(rows, field)

    # --------------------
    # DataFrame-based KPIs
    # --------------------
    def _detect_email_column(self, df: pd.DataFrame, email_col: Optional[str] = None) -> Optional[str]:
        """Detect the best email column following repo conventions.

        Priority: 'Your Work Email Address' > 'Work Email' > 'Work E-mail' > first header containing 'email'.
        Case-insensitive. Returns None if no candidate found.
        """
        if df is None or df.empty:
            return None
        if email_col and email_col in df.columns:
            return email_col
        lowers = {str(c).strip().lower(): c for c in df.columns}
        for cand in ("your work email address", "work email", "work e-mail"):
            if cand in lowers:
                return lowers[cand]
        for c in df.columns:
            if "email" in str(c).lower():
                return c
        return None

    def dedupe_by_latest(self, df: pd.DataFrame, email_col: Optional[str] = None, ts_col: str = "Timestamp") -> pd.DataFrame:
        """Deduplicate by email, keeping the latest row by Timestamp.

        - Detects email column if not provided.
        - Parses Timestamp with pandas; errors coerced to NaT.
        - Sorts by Timestamp ascending and keeps the last occurrence per email.
        - Returns a new DataFrame (does not modify input).
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=[])
        email = self._detect_email_column(df, email_col)
        if not email or email not in df.columns:
            return pd.DataFrame(columns=df.columns)
        work = df.copy()
        if ts_col in work.columns:
            work[ts_col] = pd.to_datetime(work[ts_col], errors="coerce")
            work = work.sort_values(by=[ts_col], kind="mergesort")
        # Normalize email to lower/strip for duplicate detection
        norm_col = "__norm_email__"
        work[norm_col] = work[email].astype(str).str.strip().str.lower()
        deduped = work.drop_duplicates(subset=[norm_col], keep="last").drop(columns=[norm_col])
        return deduped

    def unique_submitter_count(self, df: pd.DataFrame, email_col: Optional[str] = None, ts_col: str = "Timestamp") -> int:
        """Count unique submitters after deduping by latest Timestamp per email."""
        d = self.dedupe_by_latest(df, email_col=email_col, ts_col=ts_col)
        email = self._detect_email_column(d, email_col)
        if d is None or d.empty or not email:
            return 0
        return int(d[email].astype(str).str.strip().str.lower().replace({"": pd.NA}).dropna().nunique())

    def completion_rate_df(self, df: pd.DataFrame, expected_total: int, email_col: Optional[str] = None, ts_col: str = "Timestamp") -> float:
        """Completion rate as unique submitters / expected_total from a DataFrame."""
        if expected_total <= 0:
            return 0.0
        count = self.unique_submitter_count(df, email_col=email_col, ts_col=ts_col)
        return float(count) / float(expected_total)

    def employee_completion_rate_df(self, df: pd.DataFrame, expected_total: int = 500, email_col: Optional[str] = None, ts_col: str = "Timestamp") -> float:
        return self.completion_rate_df(df, expected_total=expected_total, email_col=email_col, ts_col=ts_col)

    def manager_completion_rate_df(self, df: pd.DataFrame, expected_total: int = 30, email_col: Optional[str] = None, ts_col: str = "Timestamp") -> float:
        return self.completion_rate_df(df, expected_total=expected_total, email_col=email_col, ts_col=ts_col)

    # --------------------
    # Stress KPI (scale → boolean)
    # --------------------
    def stressed_flag_from_scale(self, s: pd.Series) -> pd.Series:
        """Convert a stress frequency scale into a boolean 'stressed' flag.

        True for values equal to 'Frequent' or 'Extremely frequent' (case-insensitive).
        False for 'Never', 'Rarely', 'Sometimes'. Others → NaN.
        """
        if s is None or s.empty:
            return pd.Series([pd.NA] * 0, dtype="boolean")
        v = s.astype(str).str.strip().str.lower()
        true_vals = {"frequent", "extremely frequent"}
        false_vals = {"never", "rarely", "sometimes", "less frequent"}
        out = pd.Series(pd.NA, index=s.index, dtype="object")
        out[v.isin(true_vals)] = True
        out[v.isin(false_vals)] = False
        return out.astype("boolean")

    def stress_rate(self, df: pd.DataFrame, header: str) -> float:
        """Percentage (0..1) of respondents flagged as stressed from the given scale header."""
        if df is None or df.empty or header not in df.columns:
            return 0.0
        flagged = self.stressed_flag_from_scale(df[header])
        if flagged.isna().all():
            return 0.0
        return float((flagged == True).mean())

    def detect_stress_header(self, df: pd.DataFrame) -> Optional[str]:
        """Detect the stress scale column.

        Priority: exact known header, else the first column whose lowercase name
        contains both 'stress' and 'frequency'.
        """
        if df is None or df.empty:
            return None
        exact = (
            "How would you rate your stress levels at work in terms of frequency?\n"
            "Stress is the body's natural response to challenging situations, characterized by physical, emotional, or mental strain"
        )
        if exact in df.columns:
            return exact
        for c in df.columns:
            lc = str(c).lower()
            if "stress" in lc and "frequency" in lc:
                return c
        return None

    # --------------------
    # HR Pulse (Yes/No)
    # --------------------
    def normalize_bool_series(self, s: pd.Series) -> pd.Series:
        if s is None or s.empty:
            return pd.Series([pd.NA] * 0, dtype="boolean")
        m = {
            "yes": True, "y": True, "true": True, "1": True, "t": True,
            "no": False, "n": False, "false": False, "0": False, "f": False
        }
        out = s.astype(str).str.strip().str.lower().map(m)
        return out.astype("boolean")

    def rate_from_boolean(self, df: pd.DataFrame, header: str) -> float:
        if df is None or df.empty or header not in df.columns:
            return 0.0
        b = self.normalize_bool_series(df[header])
        if b.isna().all():
            return 0.0
        return float((b == True).mean())

    def detect_hr_pulse_header(self, df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty:
            return None
        exact = (
            "Do you feel it’s necessary to have more pulse check meetings with the HR Department?\n"
            "Pulse Check meeting is a meeting between the employee and HR department to assess the employee's wellbeing, work environment, and other relevant factors"
        )
        if exact in df.columns:
            return exact
        for c in df.columns:
            lc = str(c).lower()
            if "pulse" in lc and ("hr" in lc or "pulse check" in lc):
                return c
        return None

    def hr_pulse_rate(self, df: pd.DataFrame, header: Optional[str] = None) -> float:
        if header is None:
            header = self.detect_hr_pulse_header(df)
        if not header:
            return 0.0
        return self.rate_from_boolean(df, header)

    # --------------------
    # Manager: At Risk of Low Performance (Yes/No)
    # --------------------
    def detect_manager_risk_header(self, df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty:
            return None
        # Exact phrasing confirmed in requirements
        exact = "This person is at risk of low performance."
        if exact in df.columns:
            return exact
        for c in df.columns:
            lc = str(c).lower()
            if ("risk" in lc and "performance" in lc) or ("at risk" in lc and "performance" in lc):
                return c
        return None

    def manager_risk_rate(self, df: pd.DataFrame, header: Optional[str] = None) -> float:
        if df is None or df.empty:
            return 0.0
        if header is None:
            header = self.detect_manager_risk_header(df)
        if not header:
            return 0.0
        return self.rate_from_boolean(df, header)

    # --------------------
    # Manager: Stress about Employee (scale → boolean)
    # --------------------
    def detect_manager_stress_header(self, df: pd.DataFrame, exact_header: Optional[str] = None) -> Optional[str]:
        """Detects a manager question about employee stress frequency.

        - If exact_header provided and present, return it.
        - Else, first column whose name contains 'stress' and a frequency token.
        Accepted frequency tokens: 'frequency', 'frequent', 'frequently'.
        Also supports the confirmed phrasing: 'How frequently does this employee reflect stress at work?'
        """
        if df is None or df.empty:
            return None
        if exact_header and exact_header in df.columns:
            return exact_header
        # Direct known phrasing
        known_exact = "How frequently does this employee reflect stress at work?"
        if known_exact in df.columns:
            return known_exact
        freq_tokens = ["frequency", "frequent", "frequently"]
        for c in df.columns:
            lc = str(c).lower()
            if "stress" in lc and any(tok in lc for tok in freq_tokens):
                return c
        return None

    def manager_stress_rate(self, df: pd.DataFrame, header: Optional[str] = None) -> float:
        if df is None or df.empty:
            return 0.0
        if header is None:
            header = self.detect_manager_stress_header(df)
        if not header:
            return 0.0
        flagged = self.stressed_flag_from_scale(df[header])
        if flagged.isna().all():
            return 0.0
        return float((flagged == True).mean())

    def manager_stress_extraction(
        self,
        df: pd.DataFrame,
        header: Optional[str] = None,
        employee_col: str = "Subordinate Name",
        manager_col: str = "Name on Mena",
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if header is None:
            header = self.detect_manager_stress_header(df)
        if not header or header not in df.columns:
            return pd.DataFrame()
        flagged = self.stressed_flag_from_scale(df[header])
        out = df[flagged == True].copy()
        cols = []
        for c in [employee_col, manager_col, "Company Name / Department", header]:
            if c in out.columns and c not in cols:
                cols.append(c)
        return out[cols] if cols else out

    # --------------------
    # Manager: Performance Improvement Plan (PIP)
    # --------------------
    def detect_pip_header(self, df: pd.DataFrame) -> Optional[str]:
        """Detects a column indicating PIP recommendation/assignment (Yes/No).

        Prefers the confirmed phrasing from the manager sheet; falls back to headers
        containing 'pip' or 'performance improvement plan'. Case-insensitive.
        """
        if df is None or df.empty:
            return None
        exacts = [
            "The PIP serves as a structured tool and a planned approach used by HR and Direct Managers to aid the employee in enhancing his/her job performance for a period of 3 months.",
            "Given that this person is at risk of low performance, I would like to enroll him/her in the Performance Improvement Plan (PIP)",
        ]
        for exact in exacts:
            if exact in df.columns:
                return exact
        for c in df.columns:
            lc = str(c).lower()
            if ("pip" in lc) or ("performance improvement plan" in lc):
                return c
        return None

    def pip_rate(self, df: pd.DataFrame, header: Optional[str] = None) -> float:
        """Percentage (0..1) of employees with PIP=True."""
        if df is None or df.empty:
            return 0.0
        if header is None:
            header = self.detect_pip_header(df)
        if not header:
            return 0.0
        return self.rate_from_boolean(df, header)

    def pip_extraction(
        self,
        df: pd.DataFrame,
        header: Optional[str] = None,
        employee_col: str = "Subordinate Name",
        manager_col: str = "Name on Mena",
        include_cols: Optional[list] = None,
    ) -> pd.DataFrame:
        """Return rows where PIP=True with common columns and optional extras."""
        if df is None or df.empty:
            return pd.DataFrame()
        if header is None:
            header = self.detect_pip_header(df)
        if not header or header not in df.columns:
            return pd.DataFrame()
        b = self.normalize_bool_series(df[header])
        flagged = df[b == True].copy()
        if flagged.empty:
            return pd.DataFrame()
        cols = []
        for c in [employee_col, manager_col, "Company Name / Department", header]:
            if c in flagged.columns and c not in cols:
                cols.append(c)
        if include_cols:
            for c in include_cols:
                if c in flagged.columns and c not in cols:
                    cols.append(c)
        return flagged[cols] if cols else flagged

    # --------------------
    # Employee-level view from Manager 'At Risk'
    # --------------------
    def unique_employees_at_risk_count(
        self,
        df: pd.DataFrame,
        risk_header: Optional[str] = None,
        employee_col: str = "Subordinate Name",
    ) -> int:
        if df is None or df.empty or employee_col not in df.columns:
            return 0
        if risk_header is None:
            risk_header = self.detect_manager_risk_header(df)
        if not risk_header or risk_header not in df.columns:
            return 0
        b = self.normalize_bool_series(df[risk_header])
        flagged = df[b == True]
        if flagged.empty:
            return 0
        return int(
            flagged[employee_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": pd.NA})
            .dropna()
            .nunique()
        )

    def unique_employees_with_manager_submission(
        self,
        df: pd.DataFrame,
        employee_col: str = "Subordinate Name",
    ) -> int:
        if df is None or df.empty or employee_col not in df.columns:
            return 0
        return int(
            df[employee_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": pd.NA})
            .dropna()
            .nunique()
        )

    def at_risk_employee_rate(
        self,
        df: pd.DataFrame,
        denominator: str = "fixed",
        total_employees: int = 500,
        risk_header: Optional[str] = None,
        employee_col: str = "Subordinate Name",
    ) -> float:
        """Unique employees at risk / denominator.

        denominator:
        - "fixed": use `total_employees` (default 500)
        - "observed": use unique employees with any manager submission
        """
        num = self.unique_employees_at_risk_count(df, risk_header=risk_header, employee_col=employee_col)
        if denominator == "observed":
            den = self.unique_employees_with_manager_submission(df, employee_col=employee_col)
        else:
            den = total_employees
        if den <= 0:
            return 0.0
        return float(num) / float(den)

    def at_risk_extraction(
        self,
        df: pd.DataFrame,
        risk_header: Optional[str] = None,
        employee_col: str = "Subordinate Name",
        manager_col: str = "Name on Mena",
        include_cols: Optional[list] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if risk_header is None:
            risk_header = self.detect_manager_risk_header(df)
        if not risk_header or risk_header not in df.columns:
            return pd.DataFrame()
        b = self.normalize_bool_series(df[risk_header])
        flagged = df[b == True].copy()
        if flagged.empty:
            return pd.DataFrame()
        cols = []
        for c in [employee_col, manager_col, "Company Name / Department", risk_header]:
            if c in flagged.columns and c not in cols:
                cols.append(c)
        if include_cols:
            for c in include_cols:
                if c in flagged.columns and c not in cols:
                    cols.append(c)
        return flagged[cols] if cols else flagged
