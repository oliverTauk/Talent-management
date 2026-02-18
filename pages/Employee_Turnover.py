import streamlit as st
from hr_analytics.ui.style import apply_global_style, page_header, divider

st.set_page_config(page_title="Employee Turnover Prediction", layout="wide")

apply_global_style()
page_header("⏳ Employee Turnover Prediction", "Predictive analytics for workforce retention")

divider()

st.warning("⏳ This section is currently under development")
st.info("""
**Coming Soon:**
- Turnover risk prediction models
- Historical turnover analysis
- Key retention indicators
- Actionable insights for workforce planning

This module will be available in a future release.
""")

divider()

if st.button("⬅ Back to Home", use_container_width=True):
    st.switch_page("Home.py")
