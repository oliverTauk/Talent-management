"""
Streamlit download helpers for DataFrames (CSV) and matplotlib figures (PNG).
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st


def _download_df_csv(df: pd.DataFrame, filename: str, key: str) -> None:
    """Render a Streamlit download button for *df* as a CSV file."""
    if df is None or df.empty:
        st.caption("No data to download.")
        return
    st.download_button(
        label="⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
        key=key,
    )


def _download_fig_png(fig, filename: str, key: str) -> None:
    """Render a Streamlit download button for a matplotlib *fig* as PNG."""
    if fig is None:
        st.caption("No chart to download.")
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="⬇️ Download Chart (PNG)",
        data=buf,
        file_name=filename,
        mime="image/png",
        use_container_width=True,
        key=key,
    )
