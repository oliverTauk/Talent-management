# AI Agent Instructions for This Repo

These rules make an AI assistant productive immediately in this codebase. Keep changes minimal and aligned with current patterns.

## Big Picture
- Purpose: Clean HR check-in spreadsheets using the Mena Report as source of truth; offer a Streamlit UI for uploads.
- Core: `hr_analytics/services/checkin_cleaner_excel.py` with class `CheckInExcelCleaner`.
- Runners:
  - Script: `run_cleaning.py` reads three Excel files at repo root and writes cleaned outputs to `outputs/`.
  - UI: `streamlit_app.py` loads Mena automatically from root; user uploads two check-in files.

## Data Flow & Canonicalization
- Mena Report is authoritative for names. `_prepare_mena(df)` normalizes:
  - Column detection by loose matching: maps to `Email`, `Employee Name`, optional `Manager Name`; adds `EmailKey` (local-part before `@`).
- Employee Check-In cleaning (`clean_employee_checkin`):
  - Find the email column by exact or contains match on "email".
  - Map to Mena via exact email; fallback to `EmailKey` (local-part) for domain mismatches.
  - Add `Mena Name` and `Match Source` ∈ {`ExactEmail`, `LocalPart`, `Unmatched`}.
  - Drop temp columns: `__email__`, `Email`, `EmailKey`, `__email_key__`, `__Mena_Employee_Name__`.
- Manager Check-In cleaning (`clean_manager_checkin`):
  - Same email mapping; sets `Name on Mena` for the manager.
  - If Mena has `Manager Name`, canonicalize `Subordinate Name` by mapping to the canonical `Employee Name` under that manager.
  - Adds `Match Source`; drops the same temp columns.
- Combining data: `combine_cleaned(emp_df, mgr_df)` adds `Record Type` and column-aligns before concat.

## Project Conventions
- Column discovery is forgiving: prefer exact known labels, else first column containing token, e.g., any header containing "email".
- Always avoid column collisions by renaming Mena `Employee Name` to a temp (`__Mena_Employee_Name__`) before merges.
- Normalize emails: lower + strip; for fuzzy matching use `EmailKey` (regex `^([^@]+)`).
- Keep original survey name fields; add canonical fields (`Mena Name`, `Name on Mena`) instead of overwriting.
- Do not split the combined field `Company Name / Department`; keep it as a single segmentation field.

## Confirmed 2025 KPI Field Mappings
- Stress (Employee): From the exact header `How would you rate your stress levels at work in terms of frequency?\nStress is the body's natural response to challenging situations, characterized by physical, emotional, or mental strain`. Map to boolean `Stressed` where values are `Frequent` or `Extremely frequent`; treat `Less frequent` as not stressed. Token-based fallback may detect columns containing both `stress` and `frequency` if exact header differs.
- HR Pulse (Employee): From the exact header `Do you feel it’s necessary to have more pulse check meetings with the HR Department?\nPulse Check meeting is a meeting between the employee and HR department to assess the employee's wellbeing, work environment, and other relevant factors`. Treat as boolean (Yes/No). Token-based fallback may detect columns containing `pulse` and `hr` or `pulse check`.
- At Risk (Manager): From the exact header `This person is at risk of low performance.` Treat as boolean (Yes/No). Token-based fallback may detect columns containing `risk` and `performance` or `at risk`.
 - Manager Stress about Employee: Detect a frequency scale column in Manager Check-In (prefers exact header if provided; else token-based detection on both `stress` and `frequency`). Map to `Stressed by Manager` where values are `Frequent` or `Extremely frequent`; treat `Less frequent` as not stressed.

## Employee-Level At-Risk (from Manager Sheet)
- Unique employee rate: count unique canonical employees flagged at risk at least once.
- Denominator options:
  - Fixed: 500 (total employees) — default in app.
  - Observed: unique employees with any manager submission.
- Extraction should include: canonical `Subordinate Name`, manager `Name on Mena`, `Company Name / Department`, and the risk flag column.

## Developer Workflows
- Dependencies: `pandas`, `openpyxl`, `streamlit` (see `requirements.txt`). Use the same interpreter to install and run.
- Script run:
  - Inputs at repo root: `Performance Check-In.xlsx`, `Employee Performance Check-In.xlsx`, `Mena Report.xlsx` (edit paths in `run_cleaning.py` if different).
  - Outputs: `outputs/Performance_CheckIn_CLEAN.xlsx`, `outputs/Employee_Performance_CheckIn_CLEAN.xlsx`.
- Streamlit run: `streamlit run ./streamlit_app.py`.
  - Auto-detects Mena file by names: `Mena Report.xlsx|_|.csv` (case-variant).
  - Uploads can be `.xlsx/.xls/.csv`; previews + CSV downloads provided.

## Extension Points
- Adding new cleaners: follow `_prepare_mena` + email exact/fallback pattern; preserve audit column `Match Source` and drop temp columns.
- KPI/Extraction stubs:
  - `hr_analytics/services/kpis.py` – boolean-like parsing for completion/percent rates (`yes/true/1`).
  - `hr_analytics/services/extraction.py` – simple equality filters and projection helpers.
- UI beta metrics in `streamlit_app.py` auto-detect flag columns containing tokens like `stress`, `risk`, `at risk`.

## Example Usage (Programmatic)
```python
from hr_analytics.services.checkin_cleaner_excel import CheckInExcelCleaner
import pandas as pd
mena = pd.read_excel("Mena Report.xlsx")
emp  = pd.read_excel("Performance Check-In.xlsx")
mgr  = pd.read_excel("Employee Performance Check-In.xlsx")
cleaner = CheckInExcelCleaner()
emp_clean = cleaner.clean_employee_checkin(emp, mena)
mgr_clean = cleaner.clean_manager_checkin(mgr, mena)
combined  = cleaner.combine_cleaned(emp_clean, mgr_clean)
```

## Guardrails for Agents
- Do not remove or rename canonical fields: `Mena Name`, `Name on Mena`, `Match Source`.
- Do not overwrite original respondent name columns; only add canonical columns.
- When merging, always isolate Mena columns under temporary names to prevent collisions.
- Keep logic domain-agnostic: rely on column detection rather than hardcoding specific workbook schemas.
