import streamlit as st

def apply_global_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif !important;
        }

        .block-container {
            padding-top: 2.2rem !important;
            padding-bottom: 2.0rem !important;
            max-width: 1200px;
        }

        .tm-header {
            display:flex;
            align-items:flex-end;
            justify-content:space-between;
            margin: 0.25rem 0 1.25rem 0;
        }
        .tm-title {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #0F172A;
            line-height: 1.2;
        }
        .tm-subtitle {
            margin-top: 0.25rem;
            font-size: 0.98rem;
            color: #475569;
            font-weight: 500;
        }

        .tm-section {
            margin-top: 1rem;
            margin-bottom: 0.6rem;
            font-size: 1.15rem;
            font-weight: 700;
            color: #0F172A;
        }
        .tm-divider {
            height: 1px;
            background: #E5E7EB;
            margin: 1.0rem 0;
        }

        .tm-card {
            border: 1px solid #E5E7EB;
            border-radius: 16px;
            padding: 16px 16px;
            background: #FFFFFF;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        }
        .tm-card-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin: 0 0 6px 0;
            color:#0F172A;
        }
        .tm-card-desc {
            margin: 0;
            color: #475569;
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.35;
        }

        .stButton > button {
            width: 100%;
            border-radius: 12px !important;
            padding: 0.65rem 0.9rem !important;
            font-weight: 700 !important;
            border: 1px solid #E5E7EB !important;
        }
        .stButton > button:hover {
            border-color: #CBD5E1 !important;
        }

        div[data-testid="stMetric"]{
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            padding: 14px 14px;
            border-radius: 16px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetric"] label {
            color: #475569 !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricValue"]{
            color:#0F172A !important;
            font-weight: 800 !important;
        }

        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E5E7EB;
        }

        .stTextInput input, .stSelectbox div, .stFileUploader, .stNumberInput input {
            border-radius: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def page_header(title: str, subtitle: str | None = None):
    if subtitle:
        st.markdown(
            f"""
            <div class="tm-header">
              <div>
                <div class="tm-title">{title}</div>
                <div class="tm-subtitle">{subtitle}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="tm-header">
              <div class="tm-title">{title}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def section_title(text: str):
    st.markdown(f"<div class='tm-section'>{text}</div>", unsafe_allow_html=True)

def divider():
    st.markdown("<div class='tm-divider'></div>", unsafe_allow_html=True)

def card(title: str, desc: str):
    st.markdown(
        f"""
        <div class="tm-card">
          <div class="tm-card-title">{title}</div>
          <p class="tm-card-desc">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
