import streamlit as st

from hr_analytics.ui.style import apply_global_style, page_header, card, divider

# MUST be first Streamlit call
st.set_page_config(page_title="Talent Management", layout="wide", initial_sidebar_state="collapsed")

apply_global_style()
page_header("Talent Management", "Choose a module")

divider()

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    card(
        "✅ Check-ins",
        "Cleaning, KPIs, extraction, and year-over-year comparison for Employee & Manager check-ins."
    )
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.button("Open Check-ins", key="btn_checkins", use_container_width=True):
        st.switch_page("pages/check_ins.py")

with c2:
    card(
        "📊 Performance Appraisals (PA)",
        "PA cleaning, score bands, bias indicators, and year-over-year comparison."
    )
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.button("🔒 Closed", key="btn_pa", use_container_width=True, disabled=True)

with c3:
    card(
        "⏳ Employee Turnover Prediction",
        "Predictive analytics for workforce retention (Coming Soon)."
    )
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.button("🔒 Closed", key="btn_turnover", use_container_width=True, disabled=True)

divider()
st.caption("© HR Analytics Cleaning Tool")
