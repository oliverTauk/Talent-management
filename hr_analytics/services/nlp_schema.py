"""
Pydantic models and question registry for the NLP text-classification pipeline.

Fixed taxonomy, scoring rules, output validation, and the full question
registry (Employee Check-In 2025).  Every LLM response is parsed through
these models so downstream code can rely on a stable schema.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================
# Fixed taxonomy — ONLY these labels are allowed
# ============================================================
THEME_LABELS: list[str] = [
    "Role clarity",
    "Workload",
    "Process/Workflow",
    "Tools/Systems",
    "Training/Enablement",
    "Communication",
    "Manager support",
    "Career growth",
    "Culture/Environment",
    "Work-life balance",
    "Other/Unclear",
]

TAXONOMY_VERSION = "V1"

SENTIMENT_LABELS = ("positive", "neutral", "negative")


# ============================================================
# Sub-models
# ============================================================

class ThemeTag(BaseModel):
    """A single theme tag with confidence score."""
    label: str = Field(..., description="Must be one of the fixed taxonomy labels.")
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("label")
    @classmethod
    def label_must_be_valid(cls, v: str) -> str:
        if v not in THEME_LABELS:
            raise ValueError(
                f"Invalid theme label '{v}'. Must be one of: {THEME_LABELS}"
            )
        return v


class SentimentScore(BaseModel):
    """Sentiment classification with label and continuous score."""
    label: Literal["positive", "neutral", "negative"]
    score: float = Field(..., ge=-1.0, le=1.0)


# ============================================================
# Main output model (one per employee × question pair)
# ============================================================

class NLPRecordOutput(BaseModel):
    """Validated output for one employee × question analysis."""
    employee_id: str
    pair_id: str
    question_id: str = ""
    language: str = Field(..., description="Detected language code: 'en', 'ar', or 'mixed'.")
    is_non_answer: bool = Field(..., description="True if the text is empty, '-', 'n/a', etc.")
    themes: list[ThemeTag] = Field(..., max_length=3, description="Up to 3 theme tags.")
    sentiment: SentimentScore
    severity: int = Field(..., ge=0, le=3, description="0=none … 3=high")
    actionability: int = Field(..., ge=0, le=2, description="0=low … 2=high")
    summary: str = Field(..., max_length=200, description="Max ~20-word summary.")
    recommendation: str = Field(..., max_length=500, description="One actionable recommendation.")


class NLPBatchOutput(BaseModel):
    """Wrapper for a batch of records returned by the LLM."""
    results: list[NLPRecordOutput]


# ============================================================
# Question-pair definition
# ============================================================

class QuestionPair(BaseModel):
    """Describes a closed + open question pair for NLP analysis."""

    pair_id: str
    question_id: str
    q1_label: str = ""               # short human-readable label for Q1
    q1_tokens: list[str] = Field(    # keywords used for column detection of Q1
        default_factory=list,
    )
    q2_label: str                     # short human-readable label for Q2
    q2_tokens: list[str] = Field(    # keywords for column detection of Q2
        default_factory=list,
    )
    q1_trigger: Optional[str] = None  # condition on Q1 that makes Q2 relevant
    source: str = "employee"          # "employee" | "manager"


# ============================================================
# Built-in question registry — Employee Check-In 2025
# ============================================================

BUILTIN_PAIRS: list[QuestionPair] = [
    # 1) Goal Alignment with Manager
    QuestionPair(
        pair_id="goal_alignment_department_goal",
        question_id="GOAL_ALIGN_ELAB",
        q1_label="Aligned with manager on dept goal (Yes/No)",
        q1_tokens=["aligned", "manager", "department", "goal"],
        q2_label="Goal alignment elaboration",
        q2_tokens=["no", "elaborate"],
        q1_trigger="== No",
        source="employee",
    ),
    # 2) Professional Goals Discussion
    QuestionPair(
        pair_id="goal_alignment_professional_goals",
        question_id="PROF_GOALS_DISCUSS_ELAB",
        q1_label="Discussed professional goals (Yes/No)",
        q1_tokens=["discussed", "professional", "goals", "year"],
        q2_label="Professional goals elaboration",
        q2_tokens=["no", "elaborate"],
        q1_trigger="== No",
        source="employee",
    ),
    # 3) Goals Completed
    QuestionPair(
        pair_id="goals_progress",
        question_id="GOALS_COMPLETED",
        q2_label="Goals completed",
        q2_tokens=["goals", "completed"],
        source="employee",
    ),
    # 4) Goals In Progress
    QuestionPair(
        pair_id="goals_progress",
        question_id="GOALS_IN_PROGRESS",
        q2_label="Goals in progress",
        q2_tokens=["goals", "still", "progress"],
        source="employee",
    ),
    # 5) Obstacles to Pending Goals
    QuestionPair(
        pair_id="goals_progress",
        question_id="GOALS_BLOCKERS",
        q2_label="Obstacles to pending goals",
        q2_tokens=["way", "achieving", "pending", "goals"],
        source="employee",
    ),
    # 6) Positive Performance Behaviors
    QuestionPair(
        pair_id="performance_behaviors",
        question_id="BEHAVIORS_POSITIVE",
        q2_label="Positive performance behaviors",
        q2_tokens=["behaviors", "positively", "influence", "performance"],
        source="employee",
    ),
    # 7) Behaviors Developed (Past 6 Months)
    QuestionPair(
        pair_id="performance_behaviors",
        question_id="BEHAVIORS_DEVELOPED_PAST6",
        q2_label="Behaviors developed (past 6 months)",
        q2_tokens=["behaviors", "develop", "past", "six"],
        source="employee",
    ),
    # 8) Behaviors to Develop (Next 6 Months)
    QuestionPair(
        pair_id="performance_behaviors",
        question_id="BEHAVIORS_TO_DEVELOP_NEXT6",
        q2_label="Behaviors to develop (next 6 months)",
        q2_tokens=["behaviors", "develop", "adopt", "coming", "six"],
        source="employee",
    ),
    # 9) Support Needed to Develop Behaviors
    QuestionPair(
        pair_id="performance_behaviors",
        question_id="BEHAVIORS_SUPPORT_NEEDED",
        q2_label="Support needed for behavior development",
        q2_tokens=["resources", "tools", "support", "develop", "behaviors"],
        source="employee",
    ),
    # 10) Adaptation to Job Requirement Changes (conditional)
    QuestionPair(
        pair_id="adaptation_changes",
        question_id="ADAPTATION_STATEMENT",
        q1_label="Adapt to job changes (1-5 Likert)",
        q1_tokens=["adapt", "changes", "job", "requirements"],
        q2_label="Reasons for inability to adapt",
        q2_tokens=["elaborate", "reasons", "adjust", "changing", "job"],
        q1_trigger="in {1,2}",
        source="employee",
    ),
    # 11) Professional Growth Alignment (conditional)
    QuestionPair(
        pair_id="growth_alignment",
        question_id="GROWTH_ALIGNMENT_YN",
        q1_label="Tasks aligned with growth (Yes/No)",
        q1_tokens=["tasks", "responsibilities", "aligned", "professional", "growth"],
        q2_label="Growth alignment elaboration",
        q2_tokens=["no", "elaborate"],
        q1_trigger="== No",
        source="employee",
    ),
    # 12) Employee Input Taken into Consideration (conditional)
    QuestionPair(
        pair_id="leadership_dynamics",
        question_id="INPUT_EXAMPLE",
        q1_label="Input taken into consideration (Yes/No)",
        q1_tokens=["input", "taken", "consideration"],
        q2_label="Example of input taken into consideration",
        q2_tokens=["example", "input", "taken", "consideration"],
        q1_trigger="== Yes",
        source="employee",
    ),
    # 13) Collaborative Culture (conditional)
    QuestionPair(
        pair_id="work_culture",
        question_id="CULTURE_COLLAB_ELAB",
        q1_label="Culture fosters collaboration (Yes/No)",
        q1_tokens=["culture", "collaborative", "supporting", "environment"],
        q2_label="Collaborative culture elaboration",
        q2_tokens=["no", "elaborate"],
        q1_trigger="== No",
        source="employee",
    ),
    # 14) Ideal Work Culture
    QuestionPair(
        pair_id="work_culture",
        question_id="IDEAL_WORK_CULTURE",
        q2_label="Ideal work culture description",
        q2_tokens=["describe", "ideal", "work", "culture"],
        source="employee",
    ),
    # 15) Team Detachment Elaboration (conditional)
    QuestionPair(
        pair_id="team_integration",
        question_id="TEAM_DETACHED_ELAB",
        q1_label="Engaged/Indifferent/Detached",
        q1_tokens=["engaged", "indifferent", "detached"],
        q2_label="Team detachment elaboration",
        q2_tokens=["indifferent", "detached", "elaborate"],
        q1_trigger="in {Indifferent,Detached}",
        source="employee",
    ),
    # 16) Recommendation Elaboration (conditional)
    QuestionPair(
        pair_id="recommendation",
        question_id="RECOMMEND_COMPANY_ELAB",
        q1_label="Recommend working at this company (Yes/No)",
        q1_tokens=["recommend", "working", "company"],
        q2_label="Recommendation elaboration",
        q2_tokens=["no", "elaborate"],
        q1_trigger="== No",
        source="employee",
    ),
]


def get_pairs_by_source(source: str = "employee") -> list[QuestionPair]:
    """Return all built-in pairs matching the given source."""
    return [p for p in BUILTIN_PAIRS if p.source == source]


def get_pair_by_question_id(question_id: str) -> QuestionPair | None:
    """Look up a pair by its question_id."""
    for p in BUILTIN_PAIRS:
        if p.question_id == question_id:
            return p
    return None
