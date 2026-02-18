import streamlit as st

# -------------------------------------------------------------------
# Global Design System (one place for whole app)
# -------------------------------------------------------------------

def apply_global_style():
    st.markdown(
        """
        <style>
        /* ---------------------------
           App background + typography
        --------------------------- */
        html, body, [class*="css"]  {
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
        }

        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 40%, #ffffff 100%);
        }

        /* ---------------------------
           Sidebar
        --------------------------- */
        section[data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {
            color: #e5e7eb !important;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] select,
        section[data-testid="stSidebar"] textarea {
            background: rgba(255,255,255,0.06) !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            border-radius: 12px !important;
            color: #e5e7eb !important;
        }

        /* ---------------------------
           Page spacing
        --------------------------- */
        .block-container {
            padding-top: 3.5rem;  
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* ---------------------------
           Cards (we use these in helpers)
        --------------------------- */
        .hr-card {
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 16px 16px 14px 16px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            height: 140px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .hr-card + .hr-card {
            margin-top: 12px;
        }

        /* ---------------------------
           Section titles
        --------------------------- */
        .hr-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 6px 0;
        }
        .hr-subtitle {
            font-size: 0.95rem;
            color: rgba(15,23,42,0.70);
            margin: 0;
        }
        .hr-section {
            margin-top: 18px;
            margin-bottom: 8px;
            font-weight: 800;
            font-size: 1.05rem;
            color: #0f172a;
        }

        .hr-subsection {
            margin-top: 10px;
            margin-bottom: 6px;
            font-weight: 800;
            font-size: 0.92rem;      /* ✅ smaller */
            color: rgba(15,23,42,0.80);
        }

        /* ---------------------------
           Divider
        --------------------------- */
        .hr-divider {
            height: 1px;
            background: rgba(15,23,42,0.10);
            margin: 18px 0;
        }

        /* ---------------------------
           Buttons
        --------------------------- */
        .stButton button {
            border-radius: 14px;
            padding: 10px 14px;
            border: 1px solid rgba(15,23,42,0.12);
            background: #0f172a;
            color: #ffffff;
            font-weight: 700;
        }
        .stButton button:hover {
            border-color: rgba(15,23,42,0.18);
            filter: brightness(1.05);
        }
        .stButton button:active {
            transform: translateY(1px);
        }

        /* Primary button style (Streamlit type="primary") */
        button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
            border: none !important;
        }

        /* ---------------------------
           Metrics
        --------------------------- */
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 12px 14px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        [data-testid="stMetricLabel"] {
            color: rgba(15,23,42,0.75);
            font-weight: 700;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 900;
        }

        /* ---------------------------
           Expanders
        --------------------------- */
        details {
            background: rgba(255,255,255,0.70);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            padding: 8px 10px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        details summary {
            font-weight: 800;
            color: #0f172a;
        }

        /* ---------------------------
           Dataframes / tables
        --------------------------- */
        .stDataFrame {
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            padding: 6px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }

        /* ---------------------------
           Captions
        --------------------------- */
        .hr-note {
            color: rgba(15,23,42,0.65);
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str | None = None):
    """Top header card."""
    st.markdown(
        f"""
        <div class="hr-card">
            <div class="hr-title">{title}</div>
            {"<div class='hr-subtitle'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str):
    """Consistent section title."""
    st.markdown(f"<div class='hr-section'>{text}</div>", unsafe_allow_html=True)


def divider():
    """Soft divider."""
    st.markdown("<div class='hr-divider'></div>", unsafe_allow_html=True)


def card(title: str | None = None, body: str | None = None):
    """Simple card container used by Home.py."""
    st.markdown(
        f"""
        <div class="hr-card">
            {f"<div class='hr-title' style='font-size:1.05rem'>{title}</div>" if title else ""}
            {f"<div class='hr-subtitle'>{body}</div>" if body else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

