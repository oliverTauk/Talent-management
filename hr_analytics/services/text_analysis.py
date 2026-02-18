"""
Text-analysis helpers: sentiment scoring, keyword extraction, topic
bucketing, open-ended column detection, and "Reasons for No" pairing.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from hr_analytics.services.column_detection import _norm
from hr_analytics.services.normalizers import (
    _is_identity_col,
    _looks_like_yesno,
    _looks_like_scale_1_5,
)


# ============================================================
# Stop-words & topic buckets
# ============================================================

STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "of", "in",
    "on", "at", "to", "for", "from", "by", "is", "are", "was", "were", "be",
    "been", "being", "it", "its", "as", "with", "that", "this", "these",
    "those", "we", "us", "our", "you", "your", "they", "them", "their", "i",
    "me", "my", "mine",
}

TOPIC_BUCKETS: dict[str, list[str]] = {
    "Workload & Pressure": [
        "workload", "pressure", "overload", "deadline", "deadlines",
        "overtime", "hours", "capacity", "too much", "overworked",
    ],
    "Tools & Systems": [
        "tool", "tools", "system", "systems", "software", "access",
        "technology", "platform", "slow", "bug",
    ],
    "Process & Approvals": [
        "process", "approval", "bureaucracy", "procedure", "slow process",
        "red tape", "bottleneck",
    ],
    "Communication": [
        "communication", "feedback", "unclear", "informed", "update",
        "listen", "transparency", "transparent",
    ],
    "Management & Leadership": [
        "manager", "management", "leadership", "supervisor", "boss",
        "micromanag", "support", "direction",
    ],
    "Culture & Team": [
        "culture", "team", "collaborat", "conflict", "toxic", "respect",
        "inclusiv", "diversity",
    ],
    "Career & Growth": [
        "career", "growth", "promotion", "develop", "training", "learning",
        "opportunity", "advance",
    ],
    "Compensation & Benefits": [
        "salary", "compensation", "pay", "benefit", "bonus", "raise",
        "incentive",
    ],
    "Wellbeing & Stress": [
        "stress", "burnout", "wellbeing", "well-being", "mental health",
        "health", "balance", "exhaustion",
    ],
}


# ============================================================
# Sentiment helpers (VADER)
# ============================================================

_analyzer = SentimentIntensityAnalyzer()


def _sentiment_score(text: str) -> float:
    return _analyzer.polarity_scores(str(text))["compound"]


def _sentiment_label(score: float) -> str:
    if score <= -0.05:
        return "Negative"
    if score >= 0.05:
        return "Positive"
    return "Neutral"


# ============================================================
# Tokenisation / keyword extraction
# ============================================================

def _tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) >= 3 and t not in STOPWORDS]


def _top_keywords(texts: list[str], top_n: int = 25) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(_tokenize(t))
    most = counter.most_common(top_n)
    return pd.DataFrame(most, columns=["Keyword", "Count"])


# ============================================================
# Text cleaning
# ============================================================

def _clean_text(x: str) -> str:
    x = str(x)
    x = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[email]", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


# ============================================================
# Open-ended column detection
# ============================================================

def _detect_open_ended_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    if df is None or df.empty:
        return cols

    for c in df.columns:
        if _is_identity_col(c):
            continue
        s = df[c]
        if _looks_like_yesno(s) or _looks_like_scale_1_5(s):
            continue
        if s.dtype == object:
            vals = s.dropna().astype(str)
            if vals.empty:
                continue
            avg_len = vals.map(len).mean()
            uniq_ratio = vals.nunique() / max(len(vals), 1)
            if avg_len >= 20 and uniq_ratio >= 0.25:
                cols.append(c)
    return cols


# ============================================================
# Topic assignment (single best match)
# ============================================================

def _best_topic_for_text(text: str) -> tuple[str, int]:
    t = (text or "").lower()
    best_topic = "Uncategorized"
    best_score = 0
    for topic, kws in TOPIC_BUCKETS.items():
        score = sum(1 for k in kws if k in t)
        if score > best_score:
            best_topic = topic
            best_score = score
    return best_topic, best_score


def _topic_summary_with_sentiment(texts: list[str]) -> pd.DataFrame:
    total = len(texts)
    buckets: dict[str, list[str]] = {topic: [] for topic in TOPIC_BUCKETS}
    buckets["Uncategorized"] = []

    for txt in texts:
        topic, _ = _best_topic_for_text(txt)
        buckets[topic].append(txt)

    rows = []
    for topic in list(TOPIC_BUCKETS.keys()) + ["Uncategorized"]:
        matched = buckets.get(topic, [])
        if not matched:
            rows.append([topic, 0, 0.0, 0, 0, 0, "", "", ""])
            continue

        scores = [_sentiment_score(t) for t in matched]
        labels = [_sentiment_label(s) for s in scores]

        neg = sum(1 for l in labels if l == "Negative")
        neu = sum(1 for l in labels if l == "Neutral")
        pos = sum(1 for l in labels if l == "Positive")

        neg_examples = [t[:160] for t, l in zip(matched, labels) if l == "Negative"][:2]
        neu_examples = [t[:160] for t, l in zip(matched, labels) if l == "Neutral"][:2]
        pos_examples = [t[:160] for t, l in zip(matched, labels) if l == "Positive"][:2]

        pct_responses = (len(matched) / total * 100.0) if total else 0.0

        rows.append([
            topic,
            len(matched),
            round(pct_responses, 1),
            neg, neu, pos,
            " | ".join(neg_examples),
            " | ".join(neu_examples),
            " | ".join(pos_examples),
        ])

    df = pd.DataFrame(rows, columns=[
        "Topic", "Mentions", "% of responses",
        "Negative", "Neutral", "Positive",
        "Example negative", "Example neutral", "Example positive",
    ])
    return df.sort_values(["Mentions", "% of responses"], ascending=[False, False])


# ============================================================
# Action-plan generator
# ============================================================

TOPIC_ACTIONS: dict[str, list[str]] = {
    "Workload & Pressure": [
        "Review workload distribution and deadline pressure; identify overload peaks.",
        "Agree on top priorities and remove/shift non-essential tasks.",
    ],
    "Tools & Systems": [
        "Collect tool/system pain points and prioritize fixes with IT (access, slowness, bugs).",
        "Provide quick training / cheat-sheets for the main tools used.",
    ],
    "Process & Approvals": [
        "Map approval bottlenecks and simplify steps or set clear SLAs.",
        "Clarify ownership + escalation path when requests are blocked.",
    ],
    "Communication": [
        "Set a consistent feedback cadence (e.g., monthly 1:1) and align expectations.",
        "Share clearer priorities/goals and confirm understanding.",
    ],
    "Management & Leadership": [
        "Coach managers on support/recognition and reduce micromanagement patterns.",
        "Standardize 1:1 agenda templates (goals, blockers, development).",
    ],
    "Culture & Team": [
        "Improve collaboration norms (handoffs, retros, rules of engagement).",
        "Address repeated conflict patterns through HR/leadership intervention.",
    ],
    "Career & Growth": [
        "Introduce a simple development plan template (skills, goals, next steps).",
        "Clarify career paths and training options; track follow-ups.",
    ],
    "Compensation & Benefits": [
        "Collect themes and benchmark; communicate what's feasible and timeline.",
        "If changes aren't possible, improve transparency and alternatives.",
    ],
    "Wellbeing & Stress": [
        "Promote wellbeing resources and encourage time-off planning.",
        "Reduce stress drivers via workload/process fixes; monitor trends.",
    ],
}


def _reasons_action_plan(texts: list[str]) -> pd.DataFrame:
    total = len(texts)
    rows = []

    for topic, kws in TOPIC_BUCKETS.items():
        matched = [t for t in texts if any(k in t.lower() for k in kws)]
        if not matched:
            continue

        scores = [_sentiment_score(t) for t in matched]
        labels = [_sentiment_label(s) for s in scores]
        neg = sum(1 for l in labels if l == "Negative")
        neu = sum(1 for l in labels if l == "Neutral")
        pos = sum(1 for l in labels if l == "Positive")

        pct = (len(matched) / total * 100.0) if total else 0.0
        avg_sent = float(sum(scores) / len(scores)) if scores else 0.0

        examples = [t[:140] for t in matched][:2]
        actions = TOPIC_ACTIONS.get(topic, ["Review and address this theme."])

        rows.append({
            "Theme": topic,
            "% of No elaborations": round(pct, 1),
            "Mentions": len(matched),
            "Avg sentiment (-1..1)": round(avg_sent, 3),
            "Negative": neg,
            "Neutral": neu,
            "Positive": pos,
            "Recommended actions": " • ".join(actions[:2]),
            "Example snippets": " | ".join(examples),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Theme", "% of No elaborations", "Mentions", "Avg sentiment (-1..1)",
            "Negative", "Neutral", "Positive", "Recommended actions", "Example snippets",
        ])

    return pd.DataFrame(rows).sort_values(
        ["Mentions", "Avg sentiment (-1..1)"], ascending=[False, True],
    )


# ============================================================
# "Reasons for No" pairing helpers
# ============================================================

ELAB_NO_KEYWORDS = [
    "if no", "if answered no", "if you answered no", "if not", "if your answer is no",
]


def _is_elaboration_header(col_name: str) -> bool:
    h = _norm(col_name)
    if "if yes" in h or "if answered yes" in h or "if you answered yes" in h:
        return False
    if "indifferent" in h or "detached" in h:
        return False
    if "if answered 1" in h or "1 or 2" in h:
        return False
    return any(k in h for k in ELAB_NO_KEYWORDS)


def _shorten_header(h: str, max_len: int = 70) -> str:
    h = str(h).replace("\n", " ").strip()
    h = re.sub(r"\s+", " ", h)
    return (h[:max_len] + "\u2026") if len(h) > max_len else h


def _build_reasons_for_no_pairs(df: pd.DataFrame) -> list[dict]:
    cols = list(df.columns)
    pairs: list[dict] = []
    for i, c in enumerate(cols):
        if not _is_elaboration_header(c):
            continue
        parent = None
        for j in range(i - 1, -1, -1):
            cj = cols[j]
            if _is_elaboration_header(cj):
                continue
            if _looks_like_yesno(df[cj]):
                parent = cj
                break
        if parent:
            label = f"Reasons for No — {_shorten_header(parent)}"
            count = sum(1 for p in pairs if p["label"].startswith(label))
            if count > 0:
                label = f"{label} ({count + 1})"
            pairs.append({
                "label": label,
                "parent_col": parent,
                "elab_col": c,
            })
    return pairs
