import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rapidfuzz import fuzz

class NLPService:
    """
    Advanced NLP service for  Check-in analysis.
    Provides sentiment analysis, goal alignment gap detection, and resource extraction.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # Specialized 2025 lexicon for topic modeling
        self.topic_lexicon = {
            "Technology & Tools": ["azure", "ai", "python", "automation", "software", "access", "tools", "laptop"],
            "Remote & Coordination": ["remote", "zoom", "teams", "coordination", "traditional", "meeting", "internet"],
            "Hierarchy & Promotions": ["promotion", "hierarchy", "level", "grade", "title", "career"],
            "Wellbeing & Pressure": ["stress", "burnout", "pressure", "workload", "deadlines", "personal"],
            "Collaboration & Culture": ["silo", "communication", "transparency", "accountability", "integration", "team"]
        }

    def get_sentiment(self, text: str) -> dict:
        if not text or not isinstance(text, str):
            return {"compound": 0, "label": "Neutral"}
        score = self.analyzer.polarity_scores(text)["compound"]
        label = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
        return {"compound": score, "label": label}

    def calculate_alignment(self, text1: str, text2: str) -> float:
        """Calculate similarity between manager and employee goal descriptions."""
        if not text1 or not text2:
            return 0.0
        return fuzz.token_set_ratio(str(text1), str(text2))

    def extract_topics(self, text: str) -> list:
        """Find matching topics based on the 2025 lexicon."""
        if not text or not isinstance(text, str):
            return []
        text_lower = text.lower()
        matched = []
        for topic, keywords in self.topic_lexicon.items():
            if any(k in text_lower for k in keywords):
                matched.append(topic)
        return matched

    def detect_resource_requests(self, text: str) -> list:
        """Extract potential resource/tool requests using patterns."""
        if not text or not isinstance(text, str):
            return []
        # Look for "need", "require", "want", "provide" followed by some keywords
        patterns = [
            r"(?:need|require|want|provide)\s(?:a|an|the|more|some)?\s?([^.,?!]+)",
            r"([^.,?!]+)\s(?:training|course|access|license|tool)"
        ]
        requests = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.I)
            for m in matches:
                m_clean = m.strip()
                if len(m_clean) > 3 and len(m_clean) < 100:
                    requests.append(m_clean)
        return list(set(requests))

    def analyze_goal_gaps(self, df_combined: pd.DataFrame) -> pd.DataFrame:
        """
        Compare Manager and Employee goal columns in a combined DataFrame.
        Expects combined columns like 'What goals were completed?_emp' and 'What goals were completed?_mgr'
        """
        # Find goal-related columns
        emp_goal_cols = [c for c in df_combined.columns if "goal" in c.lower() and c.endswith("_emp")]
        mgr_goal_cols = [c for c in df_combined.columns if "goal" in c.lower() and c.endswith("_mgr")]
        
        results = []
        for _, row in df_combined.iterrows():
            name = row.get("Subordinate Name", "Unknown")
            for ec in emp_goal_cols:
                mc = ec.replace("_emp", "_mgr")
                if mc in df_combined.columns:
                    e_text = str(row[ec])
                    m_text = str(row[mc])
                    if e_text != "nan" and m_text != "nan":
                        score = self.calculate_alignment(e_text, m_text)
                        results.append({
                            "Employee": name,
                            "Question": ec.replace("_emp", ""),
                            "Employee Answer": e_text[:100] + ("..." if len(e_text) > 100 else ""),
                            "Manager Answer": m_text[:100] + ("..." if len(m_text) > 100 else ""),
                            "Alignment Score": score
                        })
        
        if not results:
            return pd.DataFrame(columns=["Employee", "Question", "Employee Answer", "Manager Answer", "Alignment Score"])
            
        return pd.DataFrame(results).sort_values("Alignment Score", ascending=True)
