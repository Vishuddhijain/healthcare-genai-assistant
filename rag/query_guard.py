"""
query_guard.py
==============

First-pass safety and scope gate for the Healthcare Awareness GenAI Assistant.

This module runs BEFORE retrieval. Retrieval should happen only when
``check_query_policy`` returns ``blocked=False``.

The assistant is limited to health-awareness and government-health-scheme
questions. It must not answer personal diagnosis, dosage, prescription, drug
recommendation, treatment decision, emergency triage, or clearly unrelated
off-topic questions.
"""

from dataclasses import dataclass
import re


# -- Messages -----------------------------------------------------------------

DOSAGE_MESSAGE = (
    "I cannot provide dosage, prescription, or medication advice. "
    "Please consult a qualified healthcare professional."
)

DIAGNOSIS_MESSAGE = (
    "I cannot diagnose medical conditions. "
    "Please consult a qualified healthcare professional."
)

TREATMENT_MESSAGE = (
    "I cannot recommend treatments or medications. "
    "Please consult a qualified healthcare professional."
)

EMERGENCY_MESSAGE = (
    "This may be a medical emergency. Please contact emergency services "
    "or a qualified healthcare professional immediately."
)

OFF_TOPIC_MESSAGE = (
    "I could not find reliable information on this topic in the current "
    "knowledge base. Please ask a health-awareness or government-health query."
)


# -- Blocked Intent Patterns --------------------------------------------------

# Dosage and prescription requests ask the assistant to provide medication
# instructions, even if the medicine itself is common.
DOSAGE_PATTERNS = [
    r"\b(?:what|which|recommended|right|safe|usual|normal)\s+(?:is\s+the\s+)?dose\b",
    r"\b(?:dosage|dose)\s+(?:of|for)\b",
    r"\bhow\s+(?:many|much)\s+(?:mg|milligrams?|ml|milliliters?|tablets?|capsules?|pills?)\b",
    r"\bhow\s+(?:many|much)\s+.*\b(?:should|can)\s+i\s+take\b",
    r"\b(?:should|can)\s+i\s+take\s+\d+\s*(?:mg|milligrams?|ml|milliliters?)\b",
    r"\b\d+\s*(?:mg|milligrams?|ml|milliliters?)\b.*\b(?:should|can)\s+i\s+take\b",
    r"\b(?:prescribe|prescription)\b",
]

# Diagnosis is blocked only when the user asks for a personal determination.
# Educational questions such as "What is the diagnosis process for TB?" should
# remain allowed if they are framed as awareness.
DIAGNOSIS_PATTERNS = [
    r"\bdo\s+i\s+have\b",
    r"\b(?:am|could|might)\s+i\s+(?:have|be\s+suffering\s+from)\b",
    r"\bwhat\s+(?:disease|condition|illness)\s+do\s+i\s+have\b",
    r"\bcan\s+you\s+diagnos(?:e|is)\s+(?:me|my)\b",
    r"\bdiagnos(?:e|is)\s+(?:me|my|this|these symptoms)\b",
    r"\bi\s+have\s+.*\b(?:what\s+is\s+it|what\s+could\s+it\s+be)\b",
]

# Treatment and medicine recommendations are blocked when the user is asking
# what they should take/do personally. General prevention and awareness
# questions are allowed through to retrieval and verifier checks.
TREATMENT_PATTERNS = [
    r"\b(?:which|what|best)\s+(?:medicine|medicines|drug|drugs|tablet|tablets|capsule|capsules)\s+(?:should|can)\s+i\s+take\b",
    r"\b(?:which|what|best)\s+(?:medicine|medicines|drug|drugs|tablet|tablets|capsule|capsules)\s+(?:for|to\s+treat)\b",
    r"\brecommend\s+(?:a\s+)?(?:medicine|medicines|drug|drugs|tablet|tablets|treatment|therapy)\b",
    r"\bwhat\s+treatment\s+should\s+i\s+(?:take|follow|use|get)\b",
    r"\bhow\s+should\s+i\s+treat\b",
    r"\bshould\s+i\s+(?:take|use|start|stop)\s+(?:a\s+|an\s+|the\s+)?(?:medicine|medication|drug|tablet|capsule|antibiotic)\b",
]

# Emergency triage requests ask the assistant to decide urgency or next steps
# during potentially serious symptoms. General emergency-awareness questions
# such as "What emergency services are covered by this scheme?" should not
# match these personal triage forms.
EMERGENCY_PATTERNS = [
    r"\b(?:is\s+this|am\s+i\s+having)\s+(?:an?\s+)?emergency\b",
    r"\b(?:should|do)\s+i\s+(?:go\s+to\s+the\s+)?(?:er|emergency\s+room|hospital)\b",
    r"\b(?:should|do)\s+i\s+call\s+(?:an\s+)?(?:ambulance|emergency\s+services|911|112|108)\b",
    r"\b(?:chest\s+pain|difficulty\s+breathing|shortness\s+of\s+breath|severe\s+bleeding|unconscious|stroke)\b.*\b(?:what\s+should\s+i\s+do|help|urgent)\b",
    r"\bwhat\s+should\s+i\s+do\b.*\b(?:chest\s+pain|difficulty\s+breathing|shortness\s+of\s+breath|severe\s+bleeding|unconscious|stroke)\b",
]


# -- Off-Topic Patterns -------------------------------------------------------

HEALTH_TOPIC_PATTERNS = [
    r"\b(?:health|healthcare|medical|medicine|medicines|medication|doctor|hospital|clinic|patient|disease|illness|infection|symptom|symptoms|fever|pain)\b",
    r"\b(?:malaria|dengue|tb|tuberculosis|hiv|aids|diabetes|hypertension|cancer|cholera|typhoid|measles|rubella|polio|covid|influenza|pneumonia|diarrhea|anaemia|anemia)\b",
    r"\b(?:pregnancy|pregnant|mother|maternal|newborn|infant|child|children|immunization|immunisation|vaccine|vaccination|nutrition|sanitation|hygiene)\b",
    r"\b(?:prevention|prevent|spread|transmission|risk factors|diagnosis|treatment|emergency)\b",
]

GOVERNMENT_HEALTH_SCHEME_PATTERNS = [
    r"\b(?:jssk|janani\s+shishu\s+suraksha\s+karyakram|jsy|janani\s+suraksha\s+yojana)\b",
    r"\b(?:ayushman\s+bharat|pmjay|pm-jay|abha|nhm|national\s+health\s+mission)\b",
    r"\b(?:government\s+health|health\s+scheme|health\s+yojana|public\s+health\s+scheme)\b",
]

# This is intentionally narrow. The retrieval verifier decides whether
# health-awareness content exists in the knowledge base; this guard blocks only
# topics that are plainly outside health or government-health scope.
CLEAR_OFF_TOPIC_PATTERNS = [
    r"\b(?:crypto|cryptocurrency|bitcoin|ethereum|stock|stocks|share market|mutual fund|investment|invest|trading|loan|tax)\b",
    r"\b(?:cricket|football|soccer|tennis|ipl|nba|fifa|score|match|tournament)\b",
    r"\b(?:movie|movies|music|song|songs|celebrity|actor|actress|netflix|game|gaming)\b",
    r"\b(?:programming|coding|python|javascript|java|sql|algorithm|debug|software|computer|laptop)\b",
    r"\b(?:recipe|cooking|restaurant|travel|hotel|flight|weather)\b",
    r"\b(?:prime minister|president|election|politics|political party)\b",
]


# -- Result Object ------------------------------------------------------------

@dataclass
class GuardResult:
    blocked: bool
    reason: str
    message: str


# -- Helpers ------------------------------------------------------------------

def _normalize_query(query: str) -> str:
    """Lowercase and remove punctuation noise while preserving medical units."""

    q = query.lower().strip()
    q = re.sub(r"[^a-z0-9\s/-]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q


def _matches_any(query: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, query) for pattern in patterns)


def _is_clearly_off_topic(query: str) -> bool:
    """Block only topics that are plainly outside this assistant's scope."""

    has_health_topic = _matches_any(query, HEALTH_TOPIC_PATTERNS)
    has_scheme_topic = _matches_any(query, GOVERNMENT_HEALTH_SCHEME_PATTERNS)

    if has_health_topic or has_scheme_topic:
        return False

    return _matches_any(query, CLEAR_OFF_TOPIC_PATTERNS)


# -- Main Policy Check --------------------------------------------------------

def check_query_policy(query: str) -> GuardResult:
    """
    Checks whether a query should be blocked before retrieval.

    Returns:
        GuardResult(
            blocked=True/False,
            reason="dosage|diagnosis|treatment|emergency|off_topic",
            message="..."
        )
    """

    q = _normalize_query(query)

    if not q:
        return GuardResult(
            blocked=True,
            reason="off_topic",
            message=OFF_TOPIC_MESSAGE,
        )

    if _matches_any(q, EMERGENCY_PATTERNS):
        return GuardResult(
            blocked=True,
            reason="emergency",
            message=EMERGENCY_MESSAGE,
        )

    if _matches_any(q, DOSAGE_PATTERNS):
        return GuardResult(
            blocked=True,
            reason="dosage",
            message=DOSAGE_MESSAGE,
        )

    if _matches_any(q, DIAGNOSIS_PATTERNS):
        return GuardResult(
            blocked=True,
            reason="diagnosis",
            message=DIAGNOSIS_MESSAGE,
        )

    if _matches_any(q, TREATMENT_PATTERNS):
        return GuardResult(
            blocked=True,
            reason="treatment",
            message=TREATMENT_MESSAGE,
        )

    if _is_clearly_off_topic(q):
        return GuardResult(
            blocked=True,
            reason="off_topic",
            message=OFF_TOPIC_MESSAGE,
        )

    return GuardResult(
        blocked=False,
        reason="",
        message="",
    )


# if __name__ == "__main__":
#     examples = [
#         "What is the dosage of paracetamol for a child?",  # blocked: dosage
#         "Do I have tuberculosis?",  # blocked: diagnosis
#         "Which medicine should I take for fever?",  # blocked: treatment
#         "What are the symptoms of malaria?",  # allowed
#         "What are the benefits of JSSK?",  # allowed
#         "What is the best cryptocurrency to invest in?",  # blocked: off_topic
#     ]

#     for example in examples:
#         print(example, "->", check_query_policy(example))
