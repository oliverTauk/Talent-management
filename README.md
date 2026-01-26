# HR Analytics – Cleaning Runner

This workspace includes tools to clean Employee and Manager check-in files using the Mena Report as the source of truth, plus a Streamlit web interface for HR uploads.

## Files

- `run_cleaning.py` – batch script reading three Excel files at the workspace root and writing cleaned Excel outputs to `outputs/`.
- `hr_analytics/services/checkin_cleaner_excel.py` – core cleaning logic (email normalization, fallback local-part matching, subordinate name canonicalization).
- `streamlit_app.py` – interactive web UI (upload Mena + check-ins; preview & download cleaned CSVs).
- Expected inputs at root for the runner (adjust paths inside `run_cleaning.py` if names differ):
  - `Performance Check-In.xlsx`
  - `Employee Performance Check-In.xlsx`
  - `Mena Report.xlsx`

## Quick start – Script (Windows PowerShell)

1) Ensure Python packages are installed in the SAME interpreter you will use to run the script:

```powershell
# Option A: If you know your Python path (e.g., a conda env):
& "C:\\Path\\To\\Your\\Python.exe" -m pip install --upgrade pip
& "C:\\Path\\To\\Your\\Python.exe" -m pip install pandas openpyxl

# Option B: If VS Code shows a selected interpreter, you can run:
python -m pip install --upgrade pip
python -m pip install pandas openpyxl
```

2) Run the cleaner:

```powershell
# Using the same Python you installed packages into
& "C:\\Path\\To\\Your\\Python.exe" ".\\run_cleaning.py"

# Or if your PATH/interpreter is set correctly in this terminal
python .\run_cleaning.py
```

Outputs will be written to `outputs/Performance_CheckIn_CLEAN.xlsx` and `outputs/Employee_Performance_CheckIn_CLEAN.xlsx`.

## Quick start – Streamlit Web App

1) Place a current Mena Report file in the workspace root using one of the supported names (searched in order):
  - `Mena Report.xlsx`, `Mena_Report.xlsx`, `mena_report.xlsx`
  - or CSV variants: `Mena Report.csv`, `Mena_Report.csv`, `mena_report.csv`

2) Install dependencies (same environment you will run Streamlit):

```powershell
conda activate itg  # if using conda
pip install -r requirements.txt
```

3) Launch the app:

```powershell
streamlit run .\streamlit_app.py
```

4) In the browser:
  - Upload Employee Check-In and Manager Check-In only (Mena loaded automatically)
  - Click "Run Cleaning"
  - Download cleaned CSV outputs (includes `Mena Name` / `Name on Mena` and `Match Source`).

### Match Source Column

Each cleaned file adds `Match Source` indicating how the canonical Mena name was matched:
- `ExactEmail` – direct email match.
- `LocalPart` – matched by the local-part (before `@`) due to domain mismatch.
- `Unmatched` – no Mena record found.

## Troubleshooting

- Error: `No module named pandas` or editor warning: `Import "pandas" could not be resolved`
  - Install `pandas` and `openpyxl` into the exact Python you use to execute `run_cleaning.py` (see step 1).
  - In VS Code, check the bottom-right interpreter selection to confirm the active environment.

- Error: `Import "streamlit" could not be resolved`
  - Run `pip install -r requirements.txt` in the same environment; verify with `python -c "import streamlit; print(streamlit.__version__)"`.

- Streamlit page not opening
  - Ensure terminal shows a local URL (e.g., `http://localhost:8501`). If blocked, check firewall prompts or use `streamlit run --server.port 8502 .\streamlit_app.py`.

- Error about missing input files
  - Ensure the three Excel files are at the workspace root with the expected names, or edit the paths at the top of `run_cleaning.py`.

## Environment / Interpreter Stability

- Prefer a dedicated conda env (e.g., `itg`).
- If VS Code reverts interpreter, set workspace `.vscode/settings.json`:
  ```json
  {
    "python.defaultInterpreterPath": "C:/Users/OliverTauk/AppData/Local/miniconda3/envs/itg/python.exe"
  }
  ```
- Remove conflicting user-level interpreter settings if they override the workspace.

## Updating Dependencies

Add new packages to `requirements.txt` then run:

```powershell
pip install -r requirements.txt
``` 

