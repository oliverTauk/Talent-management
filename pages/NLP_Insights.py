import streamlit as st
import pandas as pd
import re
import os
import io
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Import existing project services/style
from hr_analytics.services.checkin_cleaner_excel import CheckInExcelCleaner
from hr_analytics.ui.style import apply_global_style, page_header, section_title, divider
from hr_analytics.services.nlp_service_2025 import NLPService2025

try:
    import plotly.express as px
except ImportError:
    px = None

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Sentiment Analyzer
_analyzer = SentimentIntensityAnalyzer()

# ============================================================
# Helpers
# ============================================================

def _clean_text(x: str) -> str:
    x = str(x)
    x = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[email]", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _find_dept_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "department" in str(c).lower() or "dept" in str(c).lower():
            return c
    return None

def _filter_by_dept(df: pd.DataFrame, dept_col: str | None, selected_dept: str) -> pd.DataFrame:
    if not dept_col or not selected_dept or selected_dept == "All departments":
        return df
    return df[df[dept_col].astype(str) == selected_dept]

def _detect_open_ended_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns likely to contain open-ended feedback/text."""
    cols = []
    # Metadata/PII to exclude
    blacklist = [
        "name", "email", "id", "timestamp", "date", "dept", "department", 
        "manager", "subordinate", "status", "year", "month", "code", 
        "position", "title", "location", "gender", "age", "tenure", "hr", "pulse",
        "match source", "mena name", "name on mena"
    ]
    # Keywords that indicate qualitative feedback even if short (for follow-ups)
    qualitative_keywords = ["elaborate", "why", "describe", "comment", "feedback", "goal", "accomplish", "barrier", "support", "development"]
    
    for c in df.columns:
        c_low = str(c).lower()
        if df[c].dtype == object and not any(k in c_low for k in blacklist):
            vals = df[c].dropna().astype(str)
            if vals.empty:
                continue
            
            # If it's an "elaborate" type question, include it regardless of length
            is_elaborate = any(k in c_low for k in ["elaborate", "specify", "why", "example"])
            avg_len = vals.map(len).mean()
            
            # Threshold: Include if it has qualitative keywords OR if it's long feedback
            if is_elaborate or any(k in c_low for k in qualitative_keywords) or avg_len >= 15:
                cols.append(c)
    return cols

# ============================================================
# Main Side-Page Logic
# ============================================================

st.set_page_config(page_title="NLP Insights", layout="wide")
apply_global_style()
page_header("✨ Advanced NLP Insights", "AI-powered analysis of check-in sentiments and alignment")

if st.button("⬅ Back to Dashboard"):
    st.switch_page("pages/check_ins.py")

# Check if data is ready
if not st.session_state.get("clean_ready"):
    st.warning("No cleaned data found. Please run the cleaning process on the 'Check-ins' page first.")
    st.info("The NLP insights require the cleaned and processed data stored in session memory.")
else:
    # Sidebar setup
    st.sidebar.title("Analysis Options")
    
    # --- Emergency Environment Fix ---
    with st.sidebar.expander(" Environment Fix", expanded=False):
        st.write("If visualizations are missing, try this:")
        if st.button("Fix Plotly & Dependencies"):
            with st.spinner("Installing..."):
                import subprocess
                import sys
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "scikit-learn", "rapidfuzz", "vaderSentiment"])
                    st.success("Installation successful! Please refresh this page.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to install: {e}")
    
    who_choice = st.sidebar.radio("Select source dataset", ["Employee", "Manager"])
    
    df = st.session_state.cleaned_emp if who_choice == "Employee" else st.session_state.cleaned_mgr
    
    # Department Filter
    dept_col = _find_dept_col(df)
    dept_options = ["All departments"] + (sorted(df[dept_col].dropna().unique().tolist()) if dept_col else [])
    selected_dept = st.sidebar.selectbox("Filter by department", dept_options)
    
    df_f = _filter_by_dept(df, dept_col, selected_dept)
    st.sidebar.caption(f"Loaded {len(df_f)} records for analysis.")
    
    nlp = NLPService2025()
    
    # ------------------------------------------------------------
    #  1. Goal Alignment Analysis
    # ------------------------------------------------------------
    section_title(" Goal Alignment Analysis")
    st.markdown("Comparing goal descriptions between employees and their managers to find gaps.")
    
    combined = st.session_state.get("combined")
    if combined is not None and not combined.empty:
        comb_f = _filter_by_dept(combined, _find_dept_col(combined), selected_dept)
        gaps_df = nlp.analyze_goal_gaps(comb_f)
        
        if gaps_df.empty:
            st.info("No comparative goal data detected in the current dataset.")
        else:
            col_t1, col_t2 = st.columns([3, 1])
            with col_t2:
                st.download_button(
                    "📥 Export Gaps Table",
                    data=gaps_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"goal_gaps_{who_choice.lower()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            def color_score(val):
                color = 'red' if val < 50 else 'orange' if val < 80 else 'green'
                return f'color: {color}; font-weight: bold'
            
            st.dataframe(
                gaps_df.style.applymap(color_score, subset=['Alignment Score']),
                use_container_width=True, 
                hide_index=True
            )
    else:
        st.info("Please load both Employee and Manager datasets to enable alignment analysis.")

    divider()

    # ------------------------------------------------------------
    #  Analysis Selection (Professional Layout)
    # ------------------------------------------------------------
    section_title(" Deep Analysis Configuration")
    open_cols = _detect_open_ended_columns(df_f)
    
    if not open_cols:
        st.warning("No open-ended feedback columns were automatically detected.")
        with st.expander("Manual Column Override"):
            selected_cols = st.multiselect("Select columns manually:", df_f.columns, key="manual_nlp_cols")
    else:
        # Professional Dropdown with "All" Option
        options = ["All open ended questions"] + open_cols
        choice = st.multiselect(
            "Select specific check-in questions to analyze:", 
            options, 
            default=["All open ended questions"],
            help="Metadata is excluded. Follow-up questions automatically include their parent context (e.g., the 'Yes/No' answer).",
            key="theme_cols_nlp"
        )
        
        if "All open ended questions" in choice:
            selected_cols = open_cols
        else:
            selected_cols = choice

    if not selected_cols:
        st.info("Please select one or more questions above to view visualizations.")
    else:
        # Context-Aware Text Extraction
        processed_texts = []
        all_cols_list = list(df_f.columns)
        
        for col in selected_cols:
            is_followup = any(k in str(col).lower() for k in ["elaborate", "specify", "why", "example"])
            parent_col = None
            
            if is_followup:
                try:
                    idx = all_cols_list.index(col)
                    if idx > 0:
                        parent_col = all_cols_list[idx - 1]
                except ValueError:
                    pass
            
            # Extract data with context if applicable
            for i, row in df_f.iterrows():
                val = str(row[col]).strip()
                if not val or val.lower() == "nan":
                    continue
                
                if parent_col:
                    p_val = str(row[parent_col]).strip()
                    if p_val and p_val.lower() != "nan":
                        val = f"[{p_val}]: {val}"
                
                processed_texts.append(val)

        if not processed_texts:
            st.info("No text data found in the selected columns for the current filter.")
        else:
            texts = processed_texts # Re-assigned for downstream use
            all_text_raw = " ".join([_clean_text(t) for t in texts])

            # ------------------------------------------------------------
            #  2. Charts & Visualization
            # ------------------------------------------------------------
            col_viz_l, col_viz_r = st.columns([1, 1])
            
            with col_viz_l:
                section_title(" Core Themes")
                topic_counts = Counter()
                for t in texts:
                    topic_counts.update(nlp.extract_topics(t))
                
                if topic_counts:
                    t_df = pd.DataFrame(topic_counts.items(), columns=["Theme", "Mentions"]).sort_values("Mentions", ascending=True)
                    st.bar_chart(t_df.set_index("Theme"), use_container_width=True, color="#4F46E5")
                    
                    st.download_button(
                        " Download Theme CSV",
                        data=t_df.to_csv(index=False).encode('utf-8'),
                        file_name="theme_distribution.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No specific 2025 themes detected in this selection.")

            with col_viz_r:
                section_title(" Sentiment Analysis")
                sent_data = [nlp.get_sentiment(t)["label"] for t in texts]
                s_counts = pd.Series(sent_data).value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
                
                st.bar_chart(s_counts, color="#29b5e8", use_container_width=True)
                
                s_df = s_counts.reset_index()
                s_df.columns = ["Label", "Count"]
                st.download_button(
                    " Download Sentiment CSV",
                    data=s_df.to_csv(index=False).encode('utf-8'),
                    file_name="sentiment_breakdown.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                pos_pct = (s_counts["Positive"] / len(texts) * 100).round(1)
                st.write(f"💡 **Tip:** {pos_pct}% of feedback is positive. Detailed context: {len(texts)} entries analyzed.")

            divider()

            # ------------------------------------------------------------
            #  3. Word Cloud & Extraction
            # ------------------------------------------------------------
            col_low_l, col_low_r = st.columns([1, 1])
            
            with col_low_l:
                section_title("☁️ Insight Word Cloud")
                if WordCloud and all_text_raw.strip():
                    wc = WordCloud(
                        width=800, height=500, 
                        background_color="white",
                        colormap="viridis",
                        max_words=100
                    ).generate(all_text_raw)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                    
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format="png", bbox_inches='tight', pad_inches=0)
                    st.download_button(
                        label="📥 Download Cloud Image",
                        data=img_buffer.getvalue(),
                        file_name=f"word_cloud_{who_choice.lower()}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                else:
                    st.info("Insufficient data for word cloud.")

            with col_low_r:
                section_title(" Priority Resources")
                resources = nlp.detect_resource_requests(all_text_raw)
                if resources:
                    st.write("Identified tool/training requests:")
                    for r in resources[:12]:
                        st.markdown(f"• `{r.capitalize()}`")
                    
                    res_df = pd.DataFrame(resources, columns=["Item"])
                    st.download_button(
                        "📥 Download List",
                        data=res_df.to_csv(index=False).encode('utf-8'),
                        file_name="resource_extraction.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No specific resource requests detected.")

            divider()

            section_title(" Deep Semantic Analysis (SBERT + Advanced ML)")
            st.markdown("Automated grouping using **Sentence-BERT** embeddings for state-of-the-art semantic understanding.")
            
            with st.spinner("Loading SBERT model and analyzing... (first run may download ~80MB)"):
                try:
                    if px is None:
                        import sys
                        st.error(f"Plotly is not installed in the current environment: `{sys.executable}`")
                        st.info("Please run: `pip install plotly scikit-learn umap-learn hdbscan` in your terminal.")
                        
                        with st.expander("Diagnostic Info"):
                            st.write(f"Python version: {sys.version}")
                            st.write(f"Python path: {sys.path}")
                            try:
                                import pkg_resources
                                installed_packages = [f"{d.project_name}=={d.version}" for d in pkg_resources.working_set]
                                st.write(f"Installed packages: {sorted(installed_packages)}")
                            except:
                                st.write("Could not list packages.")
                        
                        st.info("Falling back to table-only view.")
                    
                    # Use K-Means for balanced cluster detection
                    cluster_df = nlp.perform_sbert_clustering(texts, n_clusters=4)
                    
                    # Calculate cluster quality metrics
                    n_clusters_found = cluster_df['Cluster'].nunique()
                    noise_count = len(cluster_df[cluster_df['Cluster'] == -1]) if -1 in cluster_df['Cluster'].values else 0
                    
                    col_clus_l, col_clus_r = st.columns([2, 1])
                    
                    with col_clus_l:
                        st.subheader("Semantic Theme Map (K-Means Clustering)")
                        st.caption(f"Found **{n_clusters_found}** optimal clusters from {len(texts)} responses. Closer dots = more similar meaning.")
                        
                        if px:
                            fig = px.scatter(
                                cluster_df, x="X", y="Y",
                                color="Cluster Name",
                                hover_data={"Text": True, "X": False, "Y": False},
                                template="plotly_white",
                                color_discrete_sequence=px.colors.qualitative.Prism
                            )
                            fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
                            fig.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=30))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download options for the graph
                            col_dl1, col_dl2 = st.columns(2)
                            with col_dl1:
                                # Download as interactive HTML
                                html_buffer = fig.to_html(include_plotlyjs='cdn')
                                st.download_button(
                                    label=" Download Interactive Map (HTML)",
                                    data=html_buffer,
                                    file_name=f"semantic_map_{who_choice.lower()}.html",
                                    mime="text/html",
                                    use_container_width=True
                                )
                            with col_dl2:
                                # Download as static image
                                try:
                                    img_bytes = fig.to_image(format="png", width=1200, height=800)
                                    st.download_button(
                                        label=" Download Map Image (PNG)",
                                        data=img_bytes,
                                        file_name=f"semantic_map_{who_choice.lower()}.png",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                                except Exception:
                                    # Fallback if kaleido not installed
                                    st.caption("💡 Install kaleido for PNG export: `pip install kaleido`")
                        else:
                            st.warning("Visual map unavailable (missing plotly). Viewing table instead.")
                            st.dataframe(cluster_df[["Text", "Cluster Name"]], use_container_width=True)
                    
                    with col_clus_r:
                        st.subheader(" Cluster Insights")
                        st.metric("Clusters Found", n_clusters_found, help="Automatically detected optimal number")
                        if noise_count > 0:
                            st.metric("Outliers", noise_count, help="Unique responses that don't fit major themes")
                        
                        summary_data = []
                        
                        # Helper to check if text is mostly English
                        import re
                        def is_mostly_english(text):
                            english_chars = len(re.findall(r'[a-zA-Z]', str(text)))
                            total_letters = len(re.findall(r'\w', str(text)))
                            return english_chars > (total_letters * 0.5) if total_letters > 0 else False
                        
                        for cid in sorted(cluster_df["Cluster"].unique()):
                            sub = cluster_df[cluster_df["Cluster"] == cid]
                            
                            # Find representative quote (English only)
                            rep_quote = "(Feedback in non-English)"
                            for _, row in sub.iterrows():
                                if is_mostly_english(row["Text"]):
                                    rep_quote = row["Text"]
                                    break
                            
                            if len(rep_quote) > 120:
                                rep_quote = rep_quote[:117] + "..."
                                
                            summary_data.append({
                                "Theme": sub.iloc[0]["Cluster Name"],
                                "Volume": len(sub),
                                "Typical Quote": rep_quote
                            })
                        
                        sum_df = pd.DataFrame(summary_data)
                        st.dataframe(sum_df, use_container_width=True, hide_index=True)
                        
                        st.download_button(
                            "📥 Export Clustered Data",
                            data=cluster_df.to_csv(index=False).encode('utf-8'),
                            file_name=f"semantic_clusters_{who_choice.lower()}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Clustering error: {e}")
                    import traceback
                    st.expander("Show Technical Details").code(traceback.format_exc())

divider()
st.caption("© HR Analytics Insights • Professional Edition 2025")
