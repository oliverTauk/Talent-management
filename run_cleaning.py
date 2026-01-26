from __future__ import annotations
from pathlib import Path
import sys
from typing import Optional, Tuple, List


def _find_candidates(base: Path) -> List[Path]:
    data_dir = base / "Data"
    exts = {".xlsx", ".xls", ".csv"}
    paths: List[Path] = []
    # Search Data/ recursively first
    if data_dir.exists():
        for p in data_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                if p.name.startswith("~$"):
                    continue
                paths.append(p)
    # Fallback: root
    for p in base.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            if p.name.startswith("~$"):
                continue
            paths.append(p)
    return paths


def _detect_files(base: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    def name_tokens(p: Path) -> List[str]:
        s = p.name.lower()
        # Normalize hyphens and underscores
        for ch in ["-", "_"]:
            s = s.replace(ch, " ")
        return s.split()

    candidates = _find_candidates(base)
    mena: List[Path] = []
    emp: List[Path] = []
    mgr: List[Path] = []

    for p in candidates:
        toks = name_tokens(p)
        has = lambda *kw: all(any(kw_i in t for t in toks) for kw_i in kw)
        any_has = lambda *kw: any(any(kw_i in t for t in toks) for kw_i in kw)

        # Mena detection: contains "mena" and "report"
        if has("mena", "report"):
            mena.append(p)
            continue

        # Manager check-in: prioritize explicit manager tag even if 'employee' also appears
        if any_has("manager") and any_has("check", "checkin", "check-in"):
            mgr.append(p)
            continue

        # Employee check-in: contains "employee" and a variant of check-in
        if any_has("employee") and any_has("check", "checkin", "check-in"):
            emp.append(p)
            continue

        # Performance check-in (manager) without explicit manager but no 'employee'
        if has("performance") and any_has("check", "checkin", "check-in") and not any_has("employee"):
            mgr.append(p)
            continue

    # Choose latest by modified time if multiple
    def latest(files: List[Path]) -> Optional[Path]:
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    return latest(mena), latest(emp), latest(mgr)


def main() -> int:
    try:
        from hr_analytics.services.checkin_cleaner_excel import CheckInExcelCleaner
    except Exception as e:
        print("Failed to import cleaner. Ensure the package path is correct and dependencies are installed (pandas, openpyxl).", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 1

    base = Path(__file__).parent
    # Try auto-detect in Data/ and root
    mena_path, epc_path, pc_path = _detect_files(base)

    if not all([mena_path, epc_path, pc_path]):
        # Fallback to legacy expected names at root
        legacy_pc = base / "Performance Check-In.xlsx"
        legacy_epc = base / "Employee Performance Check-In.xlsx"
        legacy_mena = base / "Mena Report.xlsx"
        if all([legacy_pc.exists(), legacy_epc.exists(), legacy_mena.exists()]):
            pc_path = legacy_pc
            epc_path = legacy_epc
            mena_path = legacy_mena
        else:
            print("Could not auto-detect required files.", file=sys.stderr)
            print("Searched in Data/ and workspace root for Mena Report, Employee Check-In, and Manager/Performance Check-In.", file=sys.stderr)
            print("Place files at root with legacy names or ensure Data/ contains discoverable filenames.", file=sys.stderr)
            return 2

    cleaner = CheckInExcelCleaner()
 
    try:
        cleaned = cleaner.clean_from_paths({
            'pc': str(pc_path),
            'epc': str(epc_path),
            'mena': str(mena_path),
        })
    except Exception as e:
        print("Cleaning failed:", file=sys.stderr)
        print(e, file=sys.stderr)
        return 3

    out_dir = base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pc = out_dir / "Performance_CheckIn_CLEAN.xlsx"
    out_epc = out_dir / "Employee_Performance_CheckIn_CLEAN.xlsx"

    try:
        cleaner.save_to_paths(cleaned, str(out_pc), str(out_epc))
    except Exception as e:
        print("Saving outputs failed:", file=sys.stderr)
        print(e, file=sys.stderr)
        return 4

    # Friendly summary
    try:
        import pandas as pd
        pc_rows = len(cleaned['pc_clean'])
        epc_rows = len(cleaned['epc_clean'])
    except Exception:
        pc_rows = epc_rows = -1

    print("Done. Cleaned files saved:")
    print(f"- {out_pc} (rows: {pc_rows if pc_rows >= 0 else 'n/a'})")
    print(f"- {out_epc} (rows: {epc_rows if epc_rows >= 0 else 'n/a'})")
    print("Inputs used:")
    print(f"- Mena: {mena_path}")
    print(f"- Employee Check-In: {epc_path}")
    print(f"- Manager/Performance Check-In: {pc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
