from __future__ import annotations
import pandas as pd


class CheckInExcelCleaner:
    """Clean Employee and Manager check-ins using Mena Report as authority.

    - Employee Check-In (Performance Check-In.xlsx):
      * Map by email to Mena 'Employee Name'
      * Replace 'Your Name' with Mena name
      * Add 'Mena Name' column
    - Employee Performance Check-In (manager -> employee):
      * Map manager by email to Mena 'Employee Name' => 'Name on Mena'
      * Replace 'Your Full Name' with that value
      * Correct 'Subordinate Name' by matching manager's 'Name on Mena' to Mena 'Manager Name'
        and using the canonical 'Employee Name' under that manager.
    """

    def _norm_email(self, s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.lower()

    def _email_key(self, s: pd.Series) -> pd.Series:
        """Return the local-part of the email (before @) in lowercase for fuzzy matching."""
        return self._norm_email(s).str.extract(r"^([^@]+)", expand=False)

    def _prepare_mena(self, df_mena: pd.DataFrame) -> pd.DataFrame:
        df = df_mena.copy()
        # Map of lower -> original
        cols_lower = {c.lower().strip(): c for c in df.columns}

        def find_exact_or_contains(candidates: list[str], fallback_contains: str | None = None) -> str | None:
            for cand in candidates:
                if cand in cols_lower:
                    return cols_lower[cand]
            if fallback_contains:
                # pick the first column that contains the token (e.g., 'email')
                for k, orig in cols_lower.items():
                    if fallback_contains in k:
                        return orig
            return None

        email_c = find_exact_or_contains(
            ['email', 'email address', 'work email', 'work e-mail', 'corporate email', 'company email'],
            fallback_contains='email',
        )
        emp_name_c = find_exact_or_contains(
            ['employee name', 'employe name', 'name', 'employee']
        )
        mgr_name_c = find_exact_or_contains(
            ['manager name', 'manager']
        )

        if not (email_c and emp_name_c):
            raise ValueError("Mena Report must contain an email column and an 'Employee Name' column")

        rename_map = {email_c: 'Email', emp_name_c: 'Employee Name'}
        if mgr_name_c and mgr_name_c != 'Manager Name':
            rename_map[mgr_name_c] = 'Manager Name'
        df = df.rename(columns=rename_map)
        df['Email'] = self._norm_email(df['Email'])
        df['EmailKey'] = self._email_key(df['Email'])
        return df

    def clean_employee_checkin(self, df_pc: pd.DataFrame, df_mena: pd.DataFrame) -> pd.DataFrame:
        pc = df_pc.copy()
        mena = self._prepare_mena(df_mena)
        # Find an email-like column in pc
        def find_pc_email(cols: list[str]) -> str | None:
            norm_map = {c.strip().lower(): c for c in cols}
            # Prefer explicit work email fields when available
            for pref in ['your work email address', 'work email', 'work e-mail']:
                if pref in norm_map:
                    return norm_map[pref]
            # Fallbacks: common exacts, then any column containing 'email'
            norm = [c.strip().lower() for c in cols]
            for i, c in enumerate(norm):
                if c in {'email', 'email address', 'work email', 'work e-mail'}:
                    return cols[i]
            for i, c in enumerate(norm):
                if 'email' in c:
                    return cols[i]
            return None

        email_col = find_pc_email(list(pc.columns))
        if not email_col:
            raise ValueError("Employee check-in must contain an email column (e.g., 'Email' or 'Email Address')")

        pc['__email__'] = self._norm_email(pc[email_col])
        name_col = next((c for c in pc.columns if c.strip().lower() in {'your name', 'name'}), None) or 'Your Name'
        if name_col not in pc.columns:
            pc[name_col] = None

        # Avoid column collisions by renaming Mena 'Employee Name' before merge
        mena_small = mena[['Email', 'EmailKey', 'Employee Name']].rename(columns={'Employee Name': '__Mena_Employee_Name__'})
        # First pass: exact email match
        mapped = pc.merge(mena_small[['Email', 'EmailKey', '__Mena_Employee_Name__']], left_on='__email__', right_on='Email', how='left')
        # Audit: initialize match source
        mapped['Match Source'] = 'Unmatched'
        mapped.loc[mapped['__Mena_Employee_Name__'].notna(), 'Match Source'] = 'ExactEmail'

        # Fallback: if missing, try matching on email local-part (handles domain differences)
        need_fallback = mapped['__Mena_Employee_Name__'].isna()
        if need_fallback.any():
            mapped['__email_key__'] = self._email_key(mapped['__email__'])
            fb_df = mapped[need_fallback].merge(
                mena_small[['EmailKey', '__Mena_Employee_Name__']].drop_duplicates('EmailKey'),
                left_on='__email_key__', right_on='EmailKey', how='left', suffixes=('_l', '_r')
            )
            fallback = fb_df['__Mena_Employee_Name___r'] if '__Mena_Employee_Name___r' in fb_df.columns else fb_df['__Mena_Employee_Name__']
            mapped.loc[need_fallback, '__Mena_Employee_Name__'] = fallback.values
            # Update match source for those newly matched
            newly = need_fallback & mapped['__Mena_Employee_Name__'].notna()
            mapped.loc[newly, 'Match Source'] = 'LocalPart'

        mapped['Mena Name'] = mapped['__Mena_Employee_Name__']
        out = mapped.drop(columns=[c for c in ['__email__', 'Email', 'EmailKey', '__email_key__', '__Mena_Employee_Name__'] if c in mapped.columns])
        return out

    def clean_manager_checkin(self, df_epc: pd.DataFrame, df_mena: pd.DataFrame) -> pd.DataFrame:
        epc = df_epc.copy()
        mena = self._prepare_mena(df_mena)

        # Find an email-like column in epc
        def find_epc_email(cols: list[str]) -> str | None:
            norm_map = {c.strip().lower(): c for c in cols}
            # Prefer explicit work email fields when available
            for pref in ['your work email address', 'work email', 'work e-mail']:
                if pref in norm_map:
                    return norm_map[pref]
            # Fallbacks: common exacts, then any column containing 'email'
            norm = [c.strip().lower() for c in cols]
            for i, c in enumerate(norm):
                if c in {'email', 'email address', 'work email', 'work e-mail'}:
                    return cols[i]
            for i, c in enumerate(norm):
                if 'email' in c:
                    return cols[i]
            return None

        email_col = find_epc_email(list(epc.columns))
        if not email_col:
            raise ValueError("Manager check-in must contain an email column (e.g., 'Email' or 'Email Address')")

        epc['__email__'] = self._norm_email(epc[email_col])

        # Avoid collisions: bring in Mena name under a unique temporary column
        mena_small = mena[['Email', 'EmailKey', 'Employee Name']].rename(columns={'Employee Name': '__Mena_Employee_Name__'})
        # First pass: exact email match
        epc = epc.merge(mena_small[['Email', 'EmailKey', '__Mena_Employee_Name__']], left_on='__email__', right_on='Email', how='left')
        epc['Match Source'] = 'Unmatched'
        epc.loc[epc['__Mena_Employee_Name__'].notna(), 'Match Source'] = 'ExactEmail'

        # Fallback: email local-part if exact match missing
        need_fallback = epc['__Mena_Employee_Name__'].isna()
        if need_fallback.any():
            epc['__email_key__'] = self._email_key(epc['__email__'])
            fb_df = epc[need_fallback].merge(
                mena_small[['EmailKey', '__Mena_Employee_Name__']].drop_duplicates('EmailKey'),
                left_on='__email_key__', right_on='EmailKey', how='left', suffixes=('_l', '_r')
            )
            fallback = fb_df['__Mena_Employee_Name___r'] if '__Mena_Employee_Name___r' in fb_df.columns else fb_df['__Mena_Employee_Name__']
            epc.loc[need_fallback, '__Mena_Employee_Name__'] = fallback.values
            newly = need_fallback & epc['__Mena_Employee_Name__'].notna()
            epc.loc[newly, 'Match Source'] = 'LocalPart'

        epc['Name on Mena'] = epc['__Mena_Employee_Name__']

        sub_col = next((c for c in epc.columns if c.strip().lower() in {'subordinate name', 'employee name'}), None) or 'Subordinate Name'
        if sub_col not in epc.columns:
            epc[sub_col] = None
        if 'Manager Name' in mena.columns:
            def norm_name(x: str) -> str:
                return ' '.join(str(x or '').strip().lower().split())
            mena_lookup = {}
            for mgr, grp in mena.groupby('Manager Name'):
                canon = list(grp['Employee Name'].dropna().astype(str))
                mena_lookup[norm_name(mgr)] = {norm_name(n): n for n in canon}

            nm_col = 'Name on Mena'
            if nm_col in epc.columns:
                def fix_row(row):
                    mgr_key = norm_name(row.get(nm_col))
                    sub_val = row.get(sub_col)
                    if not sub_val or mgr_key not in mena_lookup:
                        return row
                    sub_key = norm_name(sub_val)
                    canon_map = mena_lookup[mgr_key]
                    if sub_key in canon_map:
                        row[sub_col] = canon_map[sub_key]
                    return row
                epc = epc.apply(fix_row, axis=1)

        cleaned = epc.drop(columns=[c for c in ['__email__', 'Email', 'EmailKey', '__email_key__', '__Mena_Employee_Name__'] if c in epc.columns])
        return cleaned

    def clean_from_paths(self, paths: dict[str, str]) -> dict[str, pd.DataFrame]:
        pc = pd.read_excel(paths['pc'])
        epc = pd.read_excel(paths['epc'])
        mena = pd.read_excel(paths['mena'])
        return {
            'pc_clean': self.clean_employee_checkin(pc, mena),
            'epc_clean': self.clean_manager_checkin(epc, mena),
        }

    def save_to_paths(self, cleaned: dict[str, pd.DataFrame], out_pc: str, out_epc: str) -> None:
        cleaned['pc_clean'].to_excel(out_pc, index=False)
        cleaned['epc_clean'].to_excel(out_epc, index=False)

    def combine_cleaned(self, emp_df: pd.DataFrame, mgr_df: pd.DataFrame) -> pd.DataFrame:
        emp = emp_df.copy()
        mgr = mgr_df.copy()
        emp['Record Type'] = 'Employee'
        mgr['Record Type'] = 'Manager'
        all_cols = sorted(set(emp.columns) | set(mgr.columns))
        emp = emp.reindex(columns=all_cols)
        mgr = mgr.reindex(columns=all_cols)
        return pd.concat([emp, mgr], ignore_index=True)
