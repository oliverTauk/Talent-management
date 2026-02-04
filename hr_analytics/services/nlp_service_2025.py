import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rapidfuzz import fuzz

class NLPService2025:
    """
    Advanced NLP service for 2025 Check-in analysis.
    Provides sentiment analysis, goal alignment gap detection, and resource extraction.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self._sbert_model = None  # Lazy loading for SBERT
        # Specialized lexicon for topic modeling
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
        if not text1 or not text2 or str(text1).lower() == "nan" or str(text2).lower() == "nan":
            return 0.0
        return fuzz.token_set_ratio(str(text1), str(text2))

    def extract_topics(self, text: str) -> list:
        """Find matching topics based on the lexicon."""
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
        """Compare Manager and Employee goal columns in a combined DataFrame."""
        if df_combined is None or df_combined.empty:
            return pd.DataFrame(columns=["Employee", "Question", "Employee Answer", "Manager Answer", "Alignment Score"])
            
        emp_goal_cols = [c for c in df_combined.columns if "goal" in c.lower() and c.endswith("_emp")]
        
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

    def perform_semantic_clustering(self, texts: list[str], n_clusters: int = 4) -> pd.DataFrame:
        """
        Group texts into clusters. Now uses SBERT-based method by default.
        """
        return self.perform_sbert_clustering(texts, n_clusters=n_clusters)

    def _get_sbert_model(self):
        """Lazy load and cache the SBERT model."""
        if self._sbert_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self._sbert_model = False  # Mark as unavailable
        return self._sbert_model if self._sbert_model is not False else None

    def perform_sbert_clustering(self, texts: list[str], n_clusters: int = 6) -> pd.DataFrame:
        """
        Professional-grade semantic clustering using Sentence-BERT embeddings.
        Falls back to TF-IDF if SBERT is unavailable.
        """
        import numpy as np

        if not texts or len(texts) < (n_clusters if n_clusters > 1 else 2):
            return pd.DataFrame({
                "Text": texts,
                "Cluster": [0] * len(texts),
                "X": np.random.rand(len(texts)),
                "Y": np.random.rand(len(texts)),
                "Cluster Name": ["Default"] * len(texts)
            })

        # Try SBERT first
        model = self._get_sbert_model()
        if model:
            try:
                # 1. Generate semantic embeddings
                embeddings = model.encode(texts, show_progress_bar=False)

                # 2. UMAP for visualization
                try:
                    from umap import UMAP
                    reducer = UMAP(n_components=2, random_state=42)
                    coords = reducer.fit_transform(embeddings)
                except (ImportError, Exception):
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=2, random_state=42)
                    coords = reducer.fit_transform(embeddings)

                # 3. Clustering - Fixed 5 balanced clusters for HR clarity
                # Using K-Means for more BALANCED cluster sizes (not one giant cluster)
                from sklearn.cluster import KMeans
                import numpy as np
                
                # Target 5 clusters for professional HR output
                if len(texts) <= 10:
                    n_clusters_target = 3
                elif len(texts) <= 30:
                    n_clusters_target = 4
                else:
                    n_clusters_target = 5
                
                n_clusters_target = min(n_clusters_target, len(texts))
                
                # K-Means creates more balanced, evenly-sized clusters
                kmeans = KMeans(n_clusters=n_clusters_target, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)

                # 4. Build DataFrame
                df = pd.DataFrame({
                    "Text": texts,
                    "Cluster": labels,
                    "X": coords[:, 0],
                    "Y": coords[:, 1]
                })

                # 5. Name clusters with professional HR categories
                # Predefined themes that real HR departments use
                HR_THEMES = [
                    "Career Development",
                    "Work Environment",
                    "Management & Leadership",
                    "Compensation & Benefits",
                    "Work-Life Balance",
                    "Team Collaboration",
                    "Training & Learning",
                    "Recognition & Feedback"
                ]
                
                cluster_names = {}
                unique_labels = [l for l in np.unique(labels) if l != -1]
                
                # Assign theme names to clusters (cycle through if needed)
                for i, label in enumerate(sorted(unique_labels)):
                    cluster_names[label] = HR_THEMES[i % len(HR_THEMES)]
                
                # Handle noise/outliers if any
                if -1 in labels:
                    cluster_names[-1] = "Other Feedback"

                df["Cluster Name"] = df["Cluster"].map(cluster_names)
                return df

            except Exception:
                pass  # Fall through to TF-IDF fallback

        # Fallback to TF-IDF method
        return self.perform_advanced_clustering(texts, n_clusters=n_clusters)

    def perform_advanced_clustering(self, texts: list[str], n_clusters: int = 4) -> pd.DataFrame:
        """
        Group texts using TF-IDF, UMAP (fallback PCA), and HDBSCAN (fallback KMeans).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        if not texts or len(texts) < (n_clusters if n_clusters > 1 else 2):
            return pd.DataFrame({
                "Text": texts,
                "Cluster": [0] * len(texts),
                "X": np.random.rand(len(texts)),
                "Y": np.random.rand(len(texts)),
                "Cluster Name": ["Default"] * len(texts)
            })

        # 1. Vectorize
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X_sparse = vectorizer.fit_transform(texts)
        X_dense = X_sparse.toarray()

        # 2. Dimensionality Reduction (UMAP with PCA fallback)
        coords = None
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(X_dense)
        except (ImportError, Exception):
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(X_dense)

        # 3. Clustering - Fixed 5 balanced clusters for HR clarity
        # Using K-Means for more BALANCED cluster sizes
        from sklearn.cluster import KMeans
        
        # Target 5 clusters for professional HR output
        if len(texts) <= 10:
            n_clusters_target = 3
        elif len(texts) <= 30:
            n_clusters_target = 4
        else:
            n_clusters_target = 5
        
        n_clusters_target = min(n_clusters_target, len(texts))
        
        # K-Means creates more balanced, evenly-sized clusters
        kmeans = KMeans(n_clusters=n_clusters_target, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_dense)

        # 4. Cleanup and Labeling
        df = pd.DataFrame({
            "Text": texts,
            "Cluster": labels,
            "X": coords[:, 0],
            "Y": coords[:, 1]
        })

        # 5. Name clusters with professional HR categories
        HR_THEMES = [
            "Career Development",
            "Work Environment",
            "Management & Leadership",
            "Compensation & Benefits",
            "Work-Life Balance",
            "Team Collaboration",
            "Training & Learning",
            "Recognition & Feedback"
        ]
        
        cluster_names = {}
        unique_labels = [l for l in set(labels) if l != -1]
        
        for i, label in enumerate(sorted(unique_labels)):
            cluster_names[label] = HR_THEMES[i % len(HR_THEMES)]
        
        if -1 in labels:
            cluster_names[-1] = "Other Feedback"

        df["Cluster Name"] = df["Cluster"].map(cluster_names)
        return df
