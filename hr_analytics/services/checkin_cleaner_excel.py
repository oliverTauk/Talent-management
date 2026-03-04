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
        [
            'email', 'e mail', 'e-mail',
            'email address', 'work email', 'work e-mail',
            'corporate email', 'company email'
        ],
        fallback_contains='mail',   # ✅ catches "e mail"
    )

        emp_name_c = find_exact_or_contains(
            ['employee name', 'employe name', 'name', 'employee']
        )
        mgr_name_c = find_exact_or_contains(
            ['manager name', 'manager']
        )

        if not (email_c and emp_name_c):
            raise ValueError(
                f"Mena Report must contain an email column and an 'Employee Name' column."
            )

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

        # Fallback 2: try "Email Address" column if it exists and differs from primary email col
        alt_email_col = next((c for c in pc.columns if c.strip().lower() == 'email address'), None)
        if alt_email_col and alt_email_col != email_col:
            still_missing = mapped['__Mena_Employee_Name__'].isna()
            if still_missing.any():
                mapped['__alt_email__'] = self._norm_email(mapped[alt_email_col])
                # Exact match on alt email
                fb2 = mapped[still_missing].merge(
                    mena_small[['Email', '__Mena_Employee_Name__']].drop_duplicates('Email'),
                    left_on='__alt_email__', right_on='Email', how='left', suffixes=('_orig', '_alt')
                )
                alt_name_col = '__Mena_Employee_Name___alt' if '__Mena_Employee_Name___alt' in fb2.columns else '__Mena_Employee_Name__'
                mapped.loc[still_missing, '__Mena_Employee_Name__'] = fb2[alt_name_col].values
                newly2 = still_missing & mapped['__Mena_Employee_Name__'].notna()
                mapped.loc[newly2, 'Match Source'] = 'AltEmail'
                # If still missing, try local-part of alt email
                still_missing2 = mapped['__Mena_Employee_Name__'].isna()
                if still_missing2.any():
                    mapped['__alt_email_key__'] = self._email_key(mapped['__alt_email__'])
                    fb3 = mapped[still_missing2].merge(
                        mena_small[['EmailKey', '__Mena_Employee_Name__']].drop_duplicates('EmailKey'),
                        left_on='__alt_email_key__', right_on='EmailKey', how='left', suffixes=('_orig2', '_alt2')
                    )
                    alt2_col = '__Mena_Employee_Name___alt2' if '__Mena_Employee_Name___alt2' in fb3.columns else '__Mena_Employee_Name__'
                    mapped.loc[still_missing2, '__Mena_Employee_Name__'] = fb3[alt2_col].values
                    newly3 = still_missing2 & mapped['__Mena_Employee_Name__'].notna()
                    mapped.loc[newly3, 'Match Source'] = 'AltEmailLocalPart'
                mapped = mapped.drop(columns=[c for c in ['__alt_email__', '__alt_email_key__'] if c in mapped.columns])

        mapped['Mena Name'] = mapped['__Mena_Employee_Name__']
        # Coalesce: fall back to original name when Mena Name is NaN (unmatched)
        mapped['Mena Name'] = mapped['Mena Name'].fillna(mapped[name_col])
        out = mapped.drop(columns=[c for c in ['__email__', 'Email', 'EmailKey', '__email_key__', '__Mena_Employee_Name__'] if c in mapped.columns])

        # ── Canonicalize manager name ────────────────────────
        # Join back to Mena: employee's Mena Name → Mena Employee Name
        # to get the canonical Manager Name from the Mena report.
        mgr_col_in_checkin = next(
            (c for c in out.columns if c.strip().lower() in {
                "your manager's name", "your managers name",
                "manager name", "manager's name",
            }),
            None,
        )
        if mgr_col_in_checkin and 'Manager Name' in mena.columns:
            mena_mgr = (
                mena[['Employee Name', 'Manager Name']]
                .drop_duplicates('Employee Name')
                .rename(columns={'Manager Name': '__Mena_Manager_Name__'})
            )
            out = out.merge(
                mena_mgr,
                left_on='Mena Name',
                right_on='Employee Name',
                how='left',
                suffixes=('', '__mgr_dup__'),
            )
            # Replace the self-reported manager name with the canonical one
            has_canonical = out['__Mena_Manager_Name__'].notna()
            out.loc[has_canonical, mgr_col_in_checkin] = out.loc[has_canonical, '__Mena_Manager_Name__']
            # Drop temp columns
            out = out.drop(
                columns=[c for c in out.columns if c.startswith('__Mena_Manager_') or c.endswith('__mgr_dup__')],
            )
            # Also drop the duplicate 'Employee Name' if it appeared from the merge
            if 'Employee Name' in out.columns and 'Mena Name' in out.columns:
                out = out.drop(columns=['Employee Name'], errors='ignore')

        return out

    def clean_manager_checkin(
        self,
        df_epc: pd.DataFrame,
        df_mena: pd.DataFrame,
        df_emp_cleaned: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Clean manager check-in using Mena Report + cleaned employee check-in.

        1. Fix Subordinate Name: match to employee check-in's Mena Name
           (canonical full name), with Mena direct lookup as fallback.
        2. Add Name on Mena: manager's own canonical name from Mena
           (join manager's Your Work Email Address → Mena Email → Employee Name).
        """
        epc = df_epc.copy()
        mena = self._prepare_mena(df_mena)

        # --- helper: find email column ---
        def find_epc_email(cols: list[str]) -> str | None:
            norm_map = {c.strip().lower(): c for c in cols}
            for pref in ['your work email address', 'work email', 'work e-mail']:
                if pref in norm_map:
                    return norm_map[pref]
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
            raise ValueError("Manager check-in must contain an email column")

        epc['__email__'] = self._norm_email(epc[email_col])

        # ---- Step 1: Fix Subordinate Name ----
        sub_col = next(
            (c for c in epc.columns if c.strip().lower() == 'subordinate name'),
            None,
        ) or 'Subordinate Name'
        if sub_col not in epc.columns:
            epc[sub_col] = None

        def _norm_name(x) -> str:
            return ' '.join(str(x or '').strip().lower().split())

        # Build canonical name lookup
        name_lookup: dict[str, str] = {}  # normalized name → canonical name

        # (a) From Mena directly: Employee Name variants
        for _, row in mena.iterrows():
            canon = str(row['Employee Name']).strip()
            if not canon or canon.lower() in ('nan', 'none', ''):
                continue
            key = _norm_name(canon)
            name_lookup[key] = canon
            # Also register first + last name shortcut
            words = key.split()
            if len(words) >= 2:
                name_lookup[f"{words[0]} {words[-1]}"] = canon

        # (b) From cleaned employee check-in: Your Name → Mena Name
        if df_emp_cleaned is not None:
            yn_col = next(
                (c for c in df_emp_cleaned.columns
                 if c.strip().lower() in {'your name', 'name'}),
                None,
            )
            mn_col = 'Mena Name' if 'Mena Name' in df_emp_cleaned.columns else None
            if yn_col and mn_col:
                for _, row in df_emp_cleaned.iterrows():
                    yn = str(row[yn_col]).strip()
                    mn = str(row[mn_col]).strip()
                    if mn and mn.lower() not in ('nan', 'none', ''):
                        name_lookup[_norm_name(yn)] = mn

        def _find_canonical(sub_name):
            if pd.isna(sub_name):
                return sub_name
            key = _norm_name(sub_name)
            if not key or key in ('nan', 'none', ''):
                return sub_name
            # Exact normalized match
            if key in name_lookup:
                return name_lookup[key]
            # Fuzzy: find the lookup entry sharing the most words (≥ 2)
            sub_words = set(key.split())
            best, best_score = None, 0
            for lk, canon in name_lookup.items():
                common = sub_words & set(lk.split())
                if len(common) >= 2 and len(common) > best_score:
                    best, best_score = canon, len(common)
            return best if best else sub_name

        epc[sub_col] = epc[sub_col].apply(_find_canonical)

        # ---- Step 2: Add Name on Mena (manager's canonical name) ----
        mena_small = mena[['Email', 'EmailKey', 'Employee Name']].rename(
            columns={'Employee Name': '__Mena_Employee_Name__'}
        )
        epc = epc.merge(
            mena_small[['Email', 'EmailKey', '__Mena_Employee_Name__']],
            left_on='__email__', right_on='Email', how='left',
        )
        epc['Match Source'] = 'Unmatched'
        epc.loc[epc['__Mena_Employee_Name__'].notna(), 'Match Source'] = 'ExactEmail'

        # Fallback: email local-part
        need_fallback = epc['__Mena_Employee_Name__'].isna()
        if need_fallback.any():
            epc['__email_key__'] = self._email_key(epc['__email__'])
            fb_df = epc[need_fallback].merge(
                mena_small[['EmailKey', '__Mena_Employee_Name__']].drop_duplicates('EmailKey'),
                left_on='__email_key__', right_on='EmailKey',
                how='left', suffixes=('_l', '_r'),
            )
            fallback = (
                fb_df['__Mena_Employee_Name___r']
                if '__Mena_Employee_Name___r' in fb_df.columns
                else fb_df['__Mena_Employee_Name__']
            )
            epc.loc[need_fallback, '__Mena_Employee_Name__'] = fallback.values
            newly = need_fallback & epc['__Mena_Employee_Name__'].notna()
            epc.loc[newly, 'Match Source'] = 'LocalPart'

        epc['Name on Mena'] = epc['__Mena_Employee_Name__']

        # Drop temp columns
        cleaned = epc.drop(
            columns=[c for c in [
                '__email__', 'Email', 'EmailKey',
                '__email_key__', '__Mena_Employee_Name__',
            ] if c in epc.columns]
        )
        return cleaned

    def clean_from_paths(self, paths: dict[str, str]) -> dict[str, pd.DataFrame]:
        pc = pd.read_excel(paths['pc'])
        epc = pd.read_excel(paths['epc'])
        mena = pd.read_excel(paths['mena'])
        emp_cleaned = self.clean_employee_checkin(pc, mena)
        return {
            'pc_clean': emp_cleaned,
            'epc_clean': self.clean_manager_checkin(epc, mena, df_emp_cleaned=emp_cleaned),
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
