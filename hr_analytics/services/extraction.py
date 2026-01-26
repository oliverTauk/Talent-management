from __future__ import annotations
from typing import Iterable, Dict, Any, List


class ExtractionService:
    def filter_by(self, rows: Iterable[Dict[str, Any]], **equals) -> List[Dict[str, Any]]:
        out = []
        for r in rows or []:
            if all(str(r.get(k)) == str(v) for k, v in equals.items()):
                out.append(r)
        return out

    def project(self, rows: Iterable[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
        return [{f: r.get(f) for f in fields} for r in rows or []]

    def extract_at_risk_from_df(self, df: Any) -> List[Dict[str, Any]]:
        """
        Extract employees flagged at risk of low performance from a cleaned Manager Check-In DataFrame.

        Expects canonical columns per repo conventions:
        - "Subordinate Name" (original respondent-provided name)
        - "Name on Mena" (canonical manager name)
        - "Company Name / Department" (segmentation field)
        - A boolean-like column indicating risk, detected via tokens ("risk" & "performance" or "at risk").

        Returns list of dicts with keys: Subordinate Name, Name on Mena, Company Name / Department, RiskFlagColumn, RiskFlag.
        """
        if df is None:
            return []
        try:
            import pandas as pd  # type: ignore
        except Exception:
            # Fallback if pandas not available; attempt minimal iteration
            rows = []
            return rows

        cols = [str(c) for c in getattr(df, 'columns', [])]
        def has_tokens(c: str, required: List[str]) -> bool:
            c_low = c.lower()
            return all(tok in c_low for tok in required)

        # Detect risk flag column
        risk_cols = [c for c in cols if has_tokens(c, ["risk"]) and (has_tokens(c, ["performance"]) or has_tokens(c, ["at", "risk"]))]
        risk_col = risk_cols[0] if risk_cols else None

        key_cols = {
            'subordinate': next((c for c in cols if c.lower().strip() == 'subordinate name'), None),
            'mena_manager': next((c for c in cols if c.lower().strip() == 'name on mena'), None),
            'segment': next((c for c in cols if c.lower().strip() == 'company name / department'), None),
        }

        if not (risk_col and key_cols['subordinate'] and key_cols['mena_manager'] and key_cols['segment']):
            return []

        def is_yes(val: Any) -> bool:
            s = str(val).strip().lower()
            return s in {'yes', 'true', '1', 'y'}

        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            risk = is_yes(row.get(risk_col))
            if risk:
                records.append({
                    'Subordinate Name': row.get(key_cols['subordinate']),
                    'Name on Mena': row.get(key_cols['mena_manager']),
                    'Company Name / Department': row.get(key_cols['segment']),
                    'RiskFlagColumn': risk_col,
                    'RiskFlag': True,
                })

        return records

    
