"""MedHelp core engine.

Request pipeline (``MedHelpEngine.query``)::

    question (+ optional chat history)
      │
      ├─► 0. Small talk? ("hi", "what can you do?", "which guidelines do you have?") ─► friendly canned reply
      │
      ├─► 1. Medical guardrail ── personal diagnosis/treatment? ──► hard-coded refusal (no LLM call)
      │
      ├─► 2. Follow-up rewriting: "what about children?" -> standalone question (only when there is history)
      │
      ├─► 3. Hybrid retrieval: wide semantic search (Qdrant) + keyword search (BM25), reference-list
      │      chunks removed, results fused (RRF) and packed into a fixed character budget
      │
      ├─► 4. Grounded answer (citation + evidence-grade prompt, temperature 0)
      │
      ├─► 5. Post-checks: refusal detection, citation present?, evidence grades verifiable in the context?
      │
      ├─► 6. Citation verification (local, no LLM): does each cited sentence match the passage it cites?
      │
      └─► 7. Jargon simplifier + suggested follow-ups (two lightweight LLM calls, run in parallel)

New in this version: server-side sessions (service.py), compare mode (two or three publishers side by side),
real token streaming (``query_stream``), a semantic answer cache, an optional API reranker, and a lean
retrieval path that needs no LlamaIndex or torch at runtime.

Every response carries a ``kind`` ("answer" | "no_answer" | "guardrail" | "smalltalk") and a ``debug``
block (what was retrieved, how many passages survived each filter) so that an "I do not know" is
always explainable rather than a black box.

Nothing heavy is imported at module level, and the
retriever / LLMs / embedder can be injected through the constructor, which keeps the pure
logic in this file unit-testable without network access.
"""
from __future__ import annotations

import json
import math
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from app import config, textutils
from app.errors import (  # noqa: F401  (re-exported: other modules/tests import them from here)
    BadRequestError,
    EmbeddingError,
    IndexNotReadyError,
    LLMAuthError,
    LLMError,
    MedHelpError,
    ModelUnavailableError,
    RateLimitedError,
    RetrievalError,
)
from app.verify import verification_warning, verify_citations

logger = logging.getLogger("medhelp.engine")

NO_ANSWER_TEXT = "I do not know based on the provided guidelines."

DISCLAIMER = (
    "MedHelp summarises published clinical guidelines for educational purposes only. "
    "It is not medical advice and does not replace a qualified healthcare professional."
)

def classify_llm_error(exc: Exception) -> MedHelpError:
    """Translate provider/SDK exceptions into friendly, HTTP-mappable errors."""
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()

    # Groq answers 413 when a single request alone exceeds the per-minute token budget.
    if re.search(r"\b413\b|request too large|reduce your message size", low):
        return LLMError(
            "The request is larger than the model's per-minute token limit on the free plan. "
            "Lower TOP_K in your settings and try again."
        )
    if "ratelimit" in low or "rate limit" in low or "rate_limit" in low or re.search(r"\b429\b", low):
        retry_after = None
        m = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", low)
        if m:
            h, mi, s = (float(g) if g else 0.0 for g in m.groups())
            retry_after = h * 3600 + mi * 60 + s
        wait = f" Please retry in about {int(retry_after) + 1} seconds." if retry_after else " Please retry shortly."
        return RateLimitedError(
            "The language model's free-tier rate limit was reached." + wait, retry_after=retry_after
        )
    if "decommission" in low or "model_not_found" in low or "does not exist" in low:
        return ModelUnavailableError(
            "The configured Groq model is no longer available. Update GROQ_MODEL / GROQ_FAST_MODEL "
            "(see https://console.groq.com/docs/deprecations)."
        )
    if re.search(r"\b401\b", low) or "invalid_api_key" in low or "invalid api key" in low or "authentication" in low:
        return LLMAuthError("The Groq API key was rejected. Check GROQ_API_KEY.")
    return LLMError(f"The language model request failed: {str(exc)[:200]}")


# --------------------------------------------------------------------------- #
# 1. Medical guardrail
# --------------------------------------------------------------------------- #
REFUSAL_MESSAGE = (
    "I can't give personal medical advice, diagnoses or treatment recommendations. "
    "For a health concern affecting you or someone you care for, please speak to a qualified "
    "healthcare professional - and if this could be an emergency, contact your local emergency "
    "number right away.\n\n"
    "I can tell you what published clinical guidelines say. Try rephrasing your question in "
    "general terms, for example: \"What do the guidelines recommend for adults with high blood pressure?\""
)

SELF_HARM_MESSAGE = (
    "I'm really sorry you're going through this, and I'm not the right kind of help for it. "
    "You don't have to face it alone: if you might be in danger or are thinking about harming yourself, "
    "please contact your local emergency number or a crisis line right now, or reach out to someone you trust. "
    "A directory of free helplines by country is available at https://findahelpline.com. "
    "A doctor or mental-health professional can also help."
)

_RELATIVE = (
    r"(?:mother|mom|mum|father|dad|wife|husband|partner|son|daughter|child|children|kid|kids|baby|"
    r"infant|toddler|friend|grandmother|grandfather|grandma|grandpa|brother|sister|uncle|aunt|"
    r"boyfriend|girlfriend|patient)"
)
_SUBJECT = rf"(?:i|i'm|im|i've|ive|my\s+{_RELATIVE})"
_VERB = (
    r"(?:have|has|had|having|feel|feels|feeling|got|get|getting|suffer(?:s|ing)?\s+from|"
    r"experiencing|experience|noticed|developed|keep\s+getting)"
)
_FILL = (
    r"(?:\s+(?:a|an|the|some|severe|bad|mild|sharp|dull|constant|persistent|terrible|awful|really|very|"
    r"been|also|such|no|strong|sudden|slight|recurring|chronic|lots\s+of|of|my))"
)
_SYMPTOM = (
    r"(?:(?:\w+\s+)?(?:pain|pains|ache|aches)|headache|migraine|fever|cough|rash|nausea|vomiting|diarrh(?:o)?ea|"
    r"dizz(?:y|iness)|bleeding|swelling|swollen|chest\s+tightness|short(?:ness)?\s+of\s+breath|"
    r"palpitations?|fatigue|numbness|tingling|itching|itchy|sore\s+throat|cramps?|lump|blurred\s+vision|"
    r"weakness|insomnia|constipation|burning)"
)

_PERSONAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # "what should I take / do ...", "should I stop ..."
        r"\bwhat\s+(?:should|can|could|must|do|shall)\s+i\s+(?:do|take|use|eat|drink|try|give)\b",
        # only action verbs: blocks "should I stop/take/see a doctor..." but not "what should I know about..."
        r"\bshould\s+i\s+(?:take|stop|start|use|see|go|worry|call|skip|avoid|continue|switch|increase|decrease|reduce|"
        r"double|get|give|be\s+(?:worried|concerned|taking|on)|keep\s+taking)\b",
        r"\b(?:can|could|may|might)\s+i\s+(?:take|use|stop|start|have|combine|mix|skip|give|drink|eat|miss)\b",
        # self-diagnosis
        r"\bdo\s+i\s+(?:have|need|require)\b",
        r"\bam\s+i\s+(?:having|dying|sick|ill|pregnant|infected|at\s+risk|going\s+to)\b",
        r"\bis\s+it\s+(?:safe|ok|okay|fine)\s+(?:for\s+me|if\s+i)\b",
        r"\bwhat(?:'s|\s+is)\s+wrong\s+with\s+(?:me|my)\b",
        r"\b(?:diagnose|treat|cure|prescribe)\s+(?:me|my)\b",
        r"\bwhat\s+(?:medicine|medication|drug|pill|tablet|antibiotic|treatment)s?\s+(?:is|are)\s+"
        r"(?:best|good|right|safe)\s+for\s+(?:me|my)\b",
        r"\b(?:tell|advise|recommend)\s+(?:me\s+)?(?:what|which)\s+(?:medicine|medication|drug|pill|tablet|antibiotic|treatment)\b",
        r"\bhow\s+(?:much|many)\b.{0,40}\b(?:should|can|do)\s+i\s+(?:take|use|give|have)\b",
        r"\bdos(?:e|age)\s+for\s+(?:me|my)\b",
        # self-reported symptoms: "I have a headache", "my son has a fever", "I'm feeling dizzy"
        rf"\b{_SUBJECT}(?:\s+\w+){{0,2}}?\s+{_VERB}{_FILL}{{0,3}}\s+{_SYMPTOM}\b",
        # "my stomach hurts", "my head is pounding"
        r"\bmy\s+\w+(?:\s+\w+)?\s+(?:hurts?|aches?|is\s+(?:swollen|bleeding|burning|numb|sore|itchy)|"
        r"are\s+(?:swollen|sore|numb))\b",
    ]
]

_SELF_HARM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:kill|hurt|harm)\s+myself\b",
        r"\bend\s+my\s+(?:own\s+)?life\b",
        r"\bwant\s+to\s+die\b",
        r"\b(?:thinking|thoughts?)\s+(?:of|about)\s+(?:suicide|killing\s+myself)\b",
        r"\bi(?:'m|\s+am|\s+feel)\s+suicidal\b",
    ]
]


@dataclass(frozen=True)
class GuardrailResult:
    triggered: bool
    category: Optional[str] = None
    message: Optional[str] = None


def check_guardrails(query: str) -> GuardrailResult:
    """Detect requests for personal diagnosis/treatment (or self-harm) and return a refusal.

    Deliberately conservative and rule-based: it is free, instant, deterministic and
    never sends sensitive personal text to an LLM. General questions about what
    guidelines say ("What does WHO recommend for hypertension?") pass through.
    """
    text = query.replace("\u2019", "'").replace("\u2018", "'")
    if any(p.search(text) for p in _SELF_HARM_PATTERNS):
        return GuardrailResult(True, "self_harm", SELF_HARM_MESSAGE)
    if any(p.search(text) for p in _PERSONAL_PATTERNS):
        return GuardrailResult(True, "personal_medical_advice", REFUSAL_MESSAGE)
    return GuardrailResult(False)


# --------------------------------------------------------------------------- #
# Evidence-grade detection (ACC/AHA & ESC, GRADE/WHO, CDC, USPSTF styles)
# --------------------------------------------------------------------------- #
_ROMAN_TO_ARABIC = {"I": "1", "II": "2", "III": "3"}

# (kind, pattern). The first non-empty capture group is the canonical value.
_GRADE_RULES: list[tuple[str, re.Pattern[str]]] = [
    # ACC/AHA & ESC class of recommendation: "Class I", "Class IIa", "COR 1", "COR: 2a", "Class III (Harm)"
    (
        "cor",
        re.compile(
            r"\b(?:Class(?:\s+of\s+Recommendation)?\s*[:\-]?\s*|COR\s*[:\-]?\s*)"
            r"((?-i:III|II|I|[1-3])(?-i:[ab])?)\b(?:\s*\((?:No\s+Benefit|Harm)\))?",
            re.IGNORECASE,
        ),
    ),
    # ACC/AHA level of evidence: "Level of Evidence B-R", "LOE: C-LD"
    (
        "loe",
        re.compile(
            r"\b(?:Level\s+of\s+Evidence|LOE)\s*[:\-]?\s*((?-i:B-NR|B-R|C-LD|C-EO|A|B|C))\b",
            re.IGNORECASE,
        ),
    ),
    # ESC style: "Level A"
    ("loe", re.compile(r"\bLevel\s+((?-i:A|B|C))\b", re.IGNORECASE)),
    # GRADE (WHO etc.): "strong recommendation, moderate-certainty evidence"
    (
        "rec",
        re.compile(
            r"\b(strong|conditional|weak)\s+recommendation\b"
            r"(?:\s*,?\s*(?:based\s+on\s+|with\s+)?(?:very\s+low|low|moderate|high)[-\s]certainty(?:\s+of)?\s+evidence)?",
            re.IGNORECASE,
        ),
    ),
    ("cert", re.compile(r"\b(very\s+low|low|moderate|high)[-\s]certainty(?:\s+of)?\s+evidence\b", re.IGNORECASE)),
    ("rec", re.compile(r"\bStrength\s+of\s+Recommendation\s*[:\-]?\s*(strong|conditional|weak)\b", re.IGNORECASE)),
    # CDC (GRADE-based): "Recommendation category: A", "Evidence type: 3"
    ("cdc_cat", re.compile(r"\bRecommendation\s+category\s*[:\-]?\s*((?-i:A|B))\b", re.IGNORECASE)),
    ("cdc_ev", re.compile(r"\bEvidence\s+type\s*[:\-]?\s*([1-4])\b", re.IGNORECASE)),
    # USPSTF
    (
        "uspstf",
        re.compile(
            r"\bUSPSTF\s+Grade\s+((?-i:[A-D]|I))\b|\bGrade\s+((?-i:[A-D]))\s+recommendation\b",
            re.IGNORECASE,
        ),
    ),
]


def _canonical(kind: str, match: re.Match[str]) -> str:
    value = next((g for g in match.groups() if g), match.group(0))
    value = " ".join(value.split())
    if kind == "cor":
        head = value.rstrip("ab")
        return _ROMAN_TO_ARABIC.get(head, head) + value[len(head):]
    if kind in {"rec", "cert"}:
        return value.lower()
    return value.upper()


def _all_grade_matches(text: str) -> list[tuple[int, int, str, str]]:
    """Every grade mention as (start, end, display, token), including nested ones."""
    found: list[tuple[int, int, str, str]] = []
    for kind, pattern in _GRADE_RULES:
        for m in pattern.finditer(text):
            found.append((m.start(), m.end(), " ".join(m.group(0).split()), f"{kind}:{_canonical(kind, m)}"))
    return sorted(found, key=lambda t: (t[0], -(t[1] - t[0])))


def _drop_nested(matches: list[tuple[int, int, str, str]]) -> list[tuple[int, int, str, str]]:
    """Remove matches fully contained in a longer one (e.g. 'moderate-certainty evidence'
    inside 'strong recommendation, moderate-certainty evidence')."""
    return [
        a
        for a in matches
        if not any(b is not a and b[0] <= a[0] and a[1] <= b[1] and (b[1] - b[0]) > (a[1] - a[0]) for b in matches)
    ]


def extract_grades(text: str, limit: int = 8) -> list[str]:
    """Human-readable evidence-grade phrases found in *text* (deduplicated, in order)."""
    seen: set[str] = set()
    out: list[str] = []
    for _, _, display, _ in _drop_nested(_all_grade_matches(text)):
        key = display.lower()
        if key not in seen:
            seen.add(key)
            out.append(display)
        if len(out) >= limit:
            break
    return out


def find_unverified_grades(answer: str, context: str) -> list[str]:
    """Grades stated in the *answer* that cannot be matched to the retrieved *context*.

    A grade the model "remembers" from training data rather than reading in the
    guideline text is a hallucination risk; surfacing it lets the user double-check.
    Matching is on canonical tokens, so "Class I" == "COR 1" and "Level of Evidence B-R" == "LOE B-R".
    """
    context_tokens = {tok for _, _, _, tok in _all_grade_matches(context)}
    all_matches = _all_grade_matches(answer)
    missing: list[str] = []
    for top in _drop_nested(all_matches):
        inner = [m for m in all_matches if top[0] <= m[0] and m[1] <= top[1]]
        if any(m[3] not in context_tokens for m in inner) and top[2] not in missing:
            missing.append(top[2])
    return missing


# --------------------------------------------------------------------------- #
# 3. Prompts
# --------------------------------------------------------------------------- #
# STRICT: the instruction text from the original project brief, verbatim. All-or-nothing: if the
# retrieved context does not contain the complete answer the model must refuse. Selected with
# ANSWER_STYLE=strict.
STRICT_QA_INSTRUCTIONS = (
    "You are a medical assistant referencing clinical guidelines. Answer the query using ONLY the provided "
    "context. You MUST cite the source ([Document Name, Page X]). Furthermore, if the text mentions a "
    "'Class of Recommendation' or 'Level of Evidence', you must explicitly state it. If the context does not "
    "contain the answer, reply strictly with 'I do not know based on the provided guidelines.' "
    "Do not hallucinate.\n\n"
    "ADDITIONAL RULES\n"
    "- Each context passage starts with a header such as [Document Name, Page 12]. Cite using exactly that "
    "header text, after every factual statement.\n"
    "- Guidelines grade recommendations in different ways, e.g. ACC/AHA 'Class I' / 'COR 1' with 'Level of "
    "Evidence B-R'; GRADE 'strong recommendation, moderate-certainty evidence'; CDC 'Recommendation category A, "
    "evidence type 2'. Copy any such grade exactly as written in the context. Never infer, upgrade or invent a "
    "grade.\n"
    "- Be concise (about 180 words at most before the final line). Do not give personal medical advice.\n\n"
    "OUTPUT FORMAT\n"
    "<your answer, with citations>\n"
    "**Strength of Recommendation:** <each grade exactly as stated in the context, with its citation; or "
    "'Not stated in the retrieved text.'>\n\n"
    "If the context does not contain the answer, output ONLY the sentence "
    "'I do not know based on the provided guidelines.' and nothing else.\n"
)

# HELPFUL (default): same grounding and citation guarantees, but answers with whatever the excerpts DO
# contain instead of refusing whenever the answer is incomplete. Declines only for genuinely unrelated
# questions.
HELPFUL_QA_INSTRUCTIONS = (
    "You are MedHelp, a friendly and careful assistant that explains what published clinical guidelines say. "
    "You are given numbered guideline excerpts in CONTEXT.\n\n"
    "RULES\n"
    "1. Ground every medical statement in the CONTEXT. Put a citation right after each factual statement, "
    "copying the excerpt header exactly, e.g. [Document Name, Page 12]. Do not use outside medical knowledge "
    "for facts, doses, thresholds or recommendations.\n"
    "2. Be as helpful as the excerpts allow. If they answer only part of the question, answer that part and "
    "say briefly what the excerpts do not cover. If they are related but indirect, summarise what they do say "
    "and make clear that it is the closest available guidance. If several documents are relevant, say which "
    "document says what (they can differ in population, year or wording).\n"
    "3. Reply with exactly 'I do not know based on the provided guidelines.' ONLY if none of the excerpts is "
    "relevant to the question. Partial or indirect information is NOT a reason to refuse.\n"
    "4. If the text states a 'Class of Recommendation' / COR, 'Level of Evidence' / LOE, a GRADE strength or "
    "certainty of evidence, or a CDC recommendation category, state it exactly as written. Never infer, "
    "upgrade or invent a grade.\n"
    "5. Write in plain language, about 200 words at most, short paragraphs or bullets. Never give personal "
    "medical advice.\n\n"
    "OUTPUT FORMAT\n"
    "<your answer, with citations>\n"
    "**Strength of Recommendation:** <each grade exactly as stated in the excerpts, with its citation; or "
    "'Not stated in the retrieved text.'>\n"
)

# Back-compat alias (older code/tests import QA_INSTRUCTIONS).
QA_INSTRUCTIONS = STRICT_QA_INSTRUCTIONS

JARGON_INSTRUCTIONS = (
    "Identify any highly technical medical terms in the following text and provide a 1-sentence simple "
    "definition for each. Return as a JSON dictionary.\n\n"
    "RULES\n"
    "- Output ONLY a JSON object mapping each term to its plain-language definition (no markdown, no commentary).\n"
    "- Write for a layperson (about an 8th-grade reading level).\n"
    "- At most 8 terms. Skip everyday words, drug brand names and organisation names.\n"
    "- If there are no technical terms, output {}.\n\n"
    "TEXT:\n"
)

CONDENSE_INSTRUCTIONS = (
    "Rewrite the FOLLOW-UP question as one standalone question that can be understood without the conversation. "
    "Replace pronouns and vague references (it, they, that, the same, ...) with the specific topic from the "
    "conversation. Keep the original meaning and wording as far as possible. Do not answer the question. "
    "If it is already standalone, repeat it unchanged. Output ONLY the question, on one line.\n\n"
)


COMPARE_INSTRUCTIONS = (
    "You are MedHelp, a careful assistant that compares what different publishers' clinical guidelines say. "
    "The CONTEXT contains guideline excerpts grouped by publisher (GROUP 1, GROUP 2, ...).\n\n"
    "RULES\n"
    "1. Use ONLY the CONTEXT. Put a citation right after each factual statement, copying the excerpt header exactly, "
    "e.g. [Document Name, Page 12]. Never use outside medical knowledge for facts, doses or thresholds.\n"
    "2. Name the publisher whenever you attribute a statement, and never attribute one publisher's statement to "
    "another.\n"
    "3. Be explicit about differences in population, year, thresholds, wording and strength of evidence. If a group's "
    "excerpts do not address the question, say so instead of guessing.\n"
    "4. If the text states a 'Class of Recommendation' / COR, 'Level of Evidence' / LOE, a GRADE strength or certainty "
    "of evidence, or a CDC recommendation category, state it exactly as written. Never infer or invent a grade.\n"
    "5. Plain language, about 250 words at most. Never give personal medical advice.\n"
    "6. Reply with exactly 'I do not know based on the provided guidelines.' ONLY if no excerpt is relevant.\n\n"
    "OUTPUT FORMAT\n"
    "**Where they agree:** <points, with citations>\n"
    "**Where they differ:** <points, naming each publisher, with citations>\n"
    "**Only one covers:** <points that only some groups address; omit this line if there are none>\n"
    "**Strength of Recommendation:** <each publisher's grade exactly as stated, with its citation; or "
    "'Not stated in the retrieved text.'>\n"
)

FOLLOWUP_INSTRUCTIONS = (
    "Suggest exactly 3 short follow-up questions a reader of the ANSWER below might ask next about clinical guidelines.\n\n"
    "RULES\n"
    "- Each question must be understandable on its own (no 'it', 'they', 'that'), general (not about one specific "
    "person's health) and answerable from published clinical guidelines.\n"
    "- Prefer natural next steps: other populations (children, pregnancy, older adults), thresholds or targets, "
    "monitoring, contraindications, how two recommendations differ.\n"
    "- Under 15 words each. Do not repeat the original question.\n"
    "- Output ONLY a JSON array of 3 strings, nothing else.\n\n"
)


def build_compare_prompt(context: str, question: str, publishers: list[str]) -> str:
    groups = "; ".join(f"GROUP {i} = {p}" for i, p in enumerate(publishers, start=1))
    return (
        f"{COMPARE_INSTRUCTIONS}\n"
        f"{groups}\n"
        "-----------------\n"
        f"CONTEXT:\n{context}\n"
        "-----------------\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )


def build_followup_prompt(question: str, answer: str) -> str:
    return f"{FOLLOWUP_INSTRUCTIONS}ORIGINAL QUESTION: {question}\n\nANSWER:\n{strip_citations(answer)[:1500]}\n"


def parse_followups(raw: str, exclude: str = "", limit: int = 3) -> list[str]:
    """Parse the fast model's follow-up suggestions (a JSON array, or loosely formatted lines) into clean questions.

    Anything that is not a plausible, general guideline question (too short/long, personal medical content,
    duplicate, same as the original question) is dropped.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    raw = _FENCE_RE.sub("", raw).strip()
    items: list[str] = []
    blob = _slice(raw, "[", "]")
    if blob:
        try:
            data = json.loads(blob)
            if isinstance(data, list):
                items = [str(x) for x in data if isinstance(x, (str, int, float))]
        except ValueError:
            items = []
    if not items:
        try:
            data = json.loads(_slice(raw, "{", "}") or "null")
            if isinstance(data, dict):
                for key in ("questions", "followups", "follow_ups", "suggestions"):
                    if isinstance(data.get(key), list):
                        items = [str(x) for x in data[key]]
                        break
        except ValueError:
            pass
    if not items:  # "1. question?" / "- question?" lines
        items = [re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", ln).strip() for ln in raw.splitlines() if "?" in ln]
    out: list[str] = []
    seen = {" ".join(exclude.lower().split())}
    for item in items:
        q = " ".join(item.split()).strip("\"'\u201c\u201d ")
        key = q.lower()
        if not (10 <= len(q) <= 160) or not q.endswith("?") or key in seen:
            continue
        if check_guardrails(q).triggered or detect_smalltalk(q):
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= limit:
            break
    return out


def qa_instructions(style: Optional[str] = None) -> str:
    return STRICT_QA_INSTRUCTIONS if (style or config.ANSWER_STYLE) == "strict" else HELPFUL_QA_INSTRUCTIONS


def build_qa_prompt(context: str, question: str, style: Optional[str] = None) -> str:
    """The QA template: instructions + retrieved context + the user's question."""
    return (
        f"{qa_instructions(style)}\n"
        "-----------------\n"
        f"CONTEXT:\n{context}\n"
        "-----------------\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )


def build_condense_prompt(history: list[dict[str, str]], question: str) -> str:
    lines = []
    for turn in history:
        who = "User" if turn.get("role") == "user" else "Assistant"
        content = strip_citations(" ".join(str(turn.get("content", "")).split()))[:350]
        if content:
            lines.append(f"{who}: {content}")
    return f"{CONDENSE_INSTRUCTIONS}CONVERSATION:\n" + "\n".join(lines) + f"\n\nFOLLOW-UP: {question}\nSTANDALONE QUESTION:"


_FOLLOWUP_HINT_RE = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|he|she|there|such|same|above|previous|former|latter|"
    r"instead|also|another|else|more)\b|^(and|but|so|then|what about|how about|why)\b",
    re.IGNORECASE,
)


def looks_like_followup(question: str) -> bool:
    """Cheap test: short questions or ones containing pronouns/continuations probably depend on earlier turns."""
    return len(question.split()) <= 9 or bool(_FOLLOWUP_HINT_RE.search(question))


# --------------------------------------------------------------------------- #
# 0. Small talk (no retrieval, no LLM call)
# --------------------------------------------------------------------------- #
_SMALLTALK_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("greeting", re.compile(r"^\s*(?:hi+|hello+|hey+|hiya|howdy|namaste|yo|good\s+(?:morning|afternoon|evening))\b[\s!.,?]*(?:there|bot|medhelp|everyone)?[\s!.?]*$", re.I)),
    ("thanks", re.compile(r"^\s*(?:thanks?|thank\s+you|thx|ty|cheers|great|awesome|perfect|cool|nice|ok(?:ay)?|got\s+it)\b[\s!.,]*(?:a\s+lot|so\s+much|very\s+much|you|medhelp)?[\s!.]*$", re.I)),
    ("bye", re.compile(r"^\s*(?:bye+|goodbye|see\s+you|cya|good\s*night)\b[\s!.]*$", re.I)),
    ("documents", re.compile(r"\b(?:which|what)\s+(?:guidelines?|documents?|sources?|pdfs?|topics?|conditions?|diseases?)\b.{0,40}\b(?:do\s+you|have\s+you|are\s+(?:you|there|available|indexed|covered)|can\s+you|cover|know|include|use|support)\b|\b(?:list|show)\b.{0,25}\b(?:guidelines?|documents?|sources?|topics)\b|\bwhat\s+(?:do\s+you|can\s+i\s+ask)\b.{0,20}\b(?:know|cover|about)\b", re.I)),
    ("capabilities", re.compile(r"\b(?:who|what)\s+are\s+you\b|\bwhat\s+can\s+you\s+do\b|\bhow\s+(?:do|does)\s+(?:you|this|it|medhelp)\s+work\b|\bwhat\s+(?:is|does)\s+medhelp\b|^\s*help\b[\s!?.]*$|\bwhat\s+should\s+i\s+ask\b", re.I)),
]

CAPABILITIES_TEXT = (
    "I'm **MedHelp**, an assistant that answers questions using published, open-access clinical guidelines "
    "(WHO, CDC, ACC/AHA, KDIGO, GINA, GOLD, ADA and more).\n\n"
    "- Ask things like *\"What does WHO recommend for treating severe acute malnutrition?\"* or "
    "*\"What HbA1c target does the ADA recommend for adults with diabetes?\"*\n"
    "- Every answer is **cited** to a document and page, shows the guideline's own **evidence grade** "
    "(Class/Level of Evidence, GRADE, CDC category), and explains tricky medical terms in plain words.\n"
    "- I can follow up on earlier questions, so you can ask \"what about in pregnancy?\".\n\n"
    "I can't give personal medical advice or diagnoses - for that, please talk to a healthcare professional."
)


def detect_smalltalk(question: str) -> Optional[str]:
    """Return the small-talk category (greeting/thanks/bye/documents/capabilities) or None."""
    text = question.strip()
    if len(text) > 160:
        return None
    for kind, pattern in _SMALLTALK_RULES:
        if pattern.search(text):
            return kind
    return None


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
_CITATION_RE = re.compile(r"\[[^\[\]]{2,200}?,\s*(?:Page|p\.?)\s*[^\[\]]{1,20}\]", re.IGNORECASE)
_ANY_BRACKET_RE = re.compile(r"\[[^\[\]]{2,200}\]")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_MAX_TERM_LEN = 80
_MAX_DEF_LEN = 400


def is_refusal(answer: str) -> bool:
    """True if the model answered with the mandated 'I do not know ...' sentence."""
    norm = re.sub(r"[\s'\"*.]+", " ", answer.strip().lower()).strip()
    return norm.startswith("i do not know based on the provided guidelines")


def strip_citations(text: str) -> str:
    return " ".join(_ANY_BRACKET_RE.sub("", text).split())


def parse_jargon_json(raw: str, max_terms: int = 10) -> dict[str, str]:
    """Robustly turn the simplifier model's output into ``{term: definition}``.

    Handles markdown code fences, leading/trailing prose, a wrapper object such as
    ``{"terms": {...}}`` and a list of ``{"term": ..., "definition": ...}`` objects.
    Returns ``{}`` when nothing usable can be extracted.
    """
    if not raw:
        return {}
    text = _FENCE_RE.sub("", raw.strip()).strip()
    data: Any = None
    for candidate in (text, _slice(text, "{", "}"), _slice(text, "[", "]")):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if data is None:  # last resort: pull "term": "definition" pairs with a regex
        data = dict(re.findall(r'"([^"\n]{2,%d})"\s*:\s*"([^"\n]{3,})"' % _MAX_TERM_LEN, text))

    if isinstance(data, dict) and len(data) == 1:
        only = next(iter(data.values()))
        if isinstance(only, (dict, list)):
            data = only
    if isinstance(data, list):
        data = {
            str(item.get("term", "")).strip(): item.get("definition", "")
            for item in data
            if isinstance(item, dict)
        }
    if not isinstance(data, dict):
        return {}

    cleaned: dict[str, str] = {}
    for term, definition in data.items():
        if not isinstance(term, str) or not isinstance(definition, str):
            continue
        term, definition = term.strip(), " ".join(definition.split())
        if term and definition and len(term) <= _MAX_TERM_LEN:
            cleaned[term] = definition[:_MAX_DEF_LEN]
        if len(cleaned) >= max_terms:
            break
    return cleaned


def _slice(text: str, open_ch: str, close_ch: str) -> str:
    start, end = text.find(open_ch), text.rfind(close_ch)
    return text[start : end + 1] if start != -1 and end > start else ""


def _score(node_with_score: Any) -> float:
    score = getattr(node_with_score, "score", None)
    return 1.0 if score is None else float(score)


def _clean_snippet(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _unpack(node_with_score: Any) -> tuple[str, dict[str, Any]]:
    node = getattr(node_with_score, "node", node_with_score)
    return _clean_snippet(node.get_content()), dict(getattr(node, "metadata", {}) or {})


def _complete(llm: Any, prompt: str) -> str:
    try:
        response = llm.complete(prompt)
    except MedHelpError:
        raise
    except Exception as exc:
        raise classify_llm_error(exc) from exc
    text = getattr(response, "text", None)
    return (text if text is not None else str(response)).strip()


def _stream(llm: Any, prompt: str) -> Iterator[str]:
    """Yield the answer piece by piece; clients without ``stream`` fall back to one big piece."""
    if not hasattr(llm, "stream"):
        yield _complete(llm, prompt)
        return
    try:
        for piece in llm.stream(prompt):
            if piece:
                yield piece
    except MedHelpError:
        raise
    except Exception as exc:
        raise classify_llm_error(exc) from exc


# --------------------------------------------------------------------------- #
# Retrieval helpers
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    """One retrieved passage, normalised across the semantic and keyword retrievers."""

    id: str
    text: str
    meta: dict[str, Any]
    score: Optional[float] = None  # cosine similarity from the vector search (None = keyword-only hit)
    keyword: bool = False  # also (or only) found by the keyword search
    fused: float = 0.0  # reciprocal-rank-fusion score

    @property
    def doc(self) -> str:
        return str(self.meta.get("document_name") or self.meta.get("file_name") or "Unknown document")

    @property
    def page(self) -> str:
        return str(self.meta.get("page_label") or self.meta.get("pdf_page") or "?")

    @property
    def match_type(self) -> str:
        if self.score is None:
            return "keyword"
        return "both" if self.keyword else "semantic"

    def signature(self) -> tuple[str, str, str]:
        return (self.doc, self.page, " ".join(self.text.split())[:100])


def _hit_from_node(node_with_score: Any) -> Hit:
    text, meta = _unpack(node_with_score)
    node = getattr(node_with_score, "node", node_with_score)
    node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
    doc = meta.get("document_name") or meta.get("file_name") or "Unknown document"
    page = meta.get("page_label") or meta.get("pdf_page") or "?"
    return Hit(
        id=str(node_id) if node_id else f"{doc}|{page}|{hash(text)}",
        text=text,
        meta=meta,
        score=_score(node_with_score),
    )


def fuse_hits(dense: list[Hit], keyword: list[Hit], rrf_k: int = 60) -> list[Hit]:
    """Reciprocal-rank fusion of the semantic and keyword result lists (de-duplicated)."""
    merged: dict[str, Hit] = {}
    for rank, hit in enumerate(dense):
        entry = merged.setdefault(hit.id, hit)
        entry.fused += 1.0 / (rrf_k + rank + 1)
    for rank, hit in enumerate(keyword):
        if hit.id in merged:
            merged[hit.id].keyword = True
            merged[hit.id].fused += 1.0 / (rrf_k + rank + 1)
        else:
            hit.keyword = True
            hit.fused = 1.0 / (rrf_k + rank + 1)
            merged[hit.id] = hit
    ordered = sorted(merged.values(), key=lambda h: (h.fused, h.score or 0.0), reverse=True)
    seen: set[tuple[str, str, str]] = set()
    unique: list[Hit] = []
    for hit in ordered:  # overlapping chunks / the same chunk under two ids
        sig = hit.signature()
        if sig not in seen:
            seen.add(sig)
            unique.append(hit)
    return unique


def pack_hits(hits: list[Hit], top_k: int, char_budget: int) -> list[Hit]:
    """Take the best hits until *top_k* is reached or the character budget is used up."""
    chosen: list[Hit] = []
    used = 0
    for hit in hits:
        cost = len(hit.text) + 80  # + the [Document, Page X] header
        if chosen and used + cost > char_budget:
            continue  # a smaller, lower-ranked passage may still fit
        chosen.append(hit)
        used += cost
        if len(chosen) >= top_k:
            break
    return chosen


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    """Everything decided before the answer is generated (shared by ``query`` and ``query_stream``)."""

    started: float
    question: str
    search_question: str
    prompt: str = ""
    context: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    compare: Optional[list[str]] = None
    vector: Optional[list[float]] = None
    cache_scope: str = "single"
    cacheable: bool = False
    early: Optional[dict[str, Any]] = None  # a finished result (small talk, guardrail, no answer, cache hit)


class MedHelpEngine:
    """Small talk + guardrails + hybrid retrieval + grounded generation + checks, for one or several guidelines."""

    def __init__(
        self,
        retriever: Any = None,
        llm: Any = None,
        fast_llm: Any = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        lexical: Any = None,
        embedder: Any = None,
        reranker: Any = None,
        cache: Any = None,
        followups: Optional[bool] = None,
        verify: Optional[bool] = None,
    ) -> None:
        self._retriever = retriever
        self._retriever_injected = retriever is not None
        self._llm = llm
        self._fast_llm = fast_llm
        self._lexical = lexical
        self._embedder = embedder
        self._reranker = reranker
        self._reranker_ready = reranker is not None or retriever is not None  # injected setups never auto-build one
        self._cache = cache
        self._followups = config.FOLLOWUPS if followups is None else followups
        self._verify = config.VERIFY_CITATIONS if verify is None else verify
        self.top_k = top_k or config.TOP_K
        self.min_similarity = config.MIN_SIMILARITY if min_similarity is None else min_similarity
        self._ready = retriever is not None
        self._build_lock = threading.Lock()
        self._lexical_started = lexical is not None or retriever is not None
        self.lexical_status = "ready" if lexical is not None else ("disabled" if retriever is not None else "pending")

    # ----- readiness / lazy construction -------------------------------------
    def is_ready(self) -> bool:
        """True once the Qdrant collection contains vectors (cached after the first success)."""
        if self._ready:
            return True
        try:
            from app.qdrant_store import collection_point_count, get_qdrant_client

            self._ready = collection_point_count(get_qdrant_client()) > 0
        except Exception as exc:
            logger.warning("Readiness check failed: %s", exc)
            return False
        return self._ready

    def warm_up(self) -> None:
        """Build clients ahead of the first request and start loading the keyword index in the background."""
        if self.is_ready():
            self._components()
            self.start_lexical_loader()

    def _components(self) -> tuple[Any, Any, Any]:
        with self._build_lock:
            if self._retriever is None:
                from app.embeddings import build_embedder
                from app.qdrant_store import QdrantRetriever, get_qdrant_client

                if self._embedder is None:
                    self._embedder = build_embedder()
                self._retriever = QdrantRetriever(
                    get_qdrant_client(), config.QDRANT_COLLECTION, self._embedder, max(config.RETRIEVE_CANDIDATES, self.top_k)
                )
            if self._llm is None:
                self._llm = _build_groq(config.GROQ_MODEL)
            if self._fast_llm is None:
                self._fast_llm = _build_groq(config.GROQ_FAST_MODEL, config.GROQ_HELPER_MAX_TOKENS)
            if not self._reranker_ready:
                self._reranker_ready = True
                from app.rerank import build_reranker

                self._reranker = build_reranker()
            if self._cache is None and config.CACHE_ENABLED and not self._retriever_injected:
                from app.cache import SemanticCache

                self._cache = SemanticCache(
                    max_entries=config.CACHE_MAX_ENTRIES,
                    ttl_seconds=config.CACHE_TTL_MINUTES * 60,
                    threshold=config.CACHE_SIMILARITY,
                )
        return self._retriever, self._llm, self._fast_llm

    # ----- keyword index (SQLite FTS5), loaded in the background -------------------
    def start_lexical_loader(self) -> None:
        if self._lexical_started or not config.LEXICAL_SEARCH:
            if not config.LEXICAL_SEARCH:
                self.lexical_status = "disabled"
            return
        self._lexical_started = True
        self.lexical_status = "loading"
        threading.Thread(target=self._load_lexical, name="keyword-loader", daemon=True).start()

    def _load_lexical(self) -> None:
        """Open the shipped keyword database; rebuild from Qdrant if it is missing or stale. Never raises."""
        try:
            import tempfile
            from pathlib import Path

            from app.keyword_db import KeywordDB, build_from_qdrant, is_current
            from app.qdrant_store import collection_point_count, get_qdrant_client

            client = get_qdrant_client()
            try:
                points: Optional[int] = collection_point_count(client)
            except Exception:  # noqa: BLE001
                points = None
            db = None
            path = Path(config.KEYWORD_DB_PATH)
            if path.exists():
                try:
                    db = KeywordDB.open(path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Keyword database %s is unreadable (%s); rebuilding.", path, exc)
            if db is not None:
                self._lexical = db  # usable immediately, even if slightly stale
                self.lexical_status = "ready"
                if is_current(db, points):
                    logger.info("Keyword index ready (%d chunks, from %s).", len(db), path)
                    return
                logger.warning("Keyword database is out of date with Qdrant; refreshing in the background.")
                self.lexical_status = "stale-refreshing"
            else:
                self.lexical_status = "building"
            runtime = Path(tempfile.gettempdir()) / "medhelp-keyword-runtime.sqlite"
            build_from_qdrant(client, config.QDRANT_COLLECTION, runtime)
            self._lexical = KeywordDB(runtime)
            self.lexical_status = "ready"
            logger.info("Keyword index rebuilt from Qdrant (%d chunks).", len(self._lexical))
        except Exception as exc:  # noqa: BLE001 - keyword search is a bonus
            logger.warning("Keyword index unavailable, using semantic search only: %s", exc)
            if self._lexical is None:
                self.lexical_status = "unavailable"

    # ----- knowledge-base listing ------------------------------------------------
    def documents(self) -> list[dict[str, Any]]:
        """Indexed guidelines for the UI: from data/manifest.json, else from the keyword index."""
        docs: list[dict[str, Any]] = []
        try:
            from app import fetch_data

            for filename, info in fetch_data.load_manifest().items():
                docs.append(
                    {
                        "title": info.get("title") or filename,
                        "publisher": info.get("publisher", ""),
                        "pages": info.get("pages"),
                        "url": info.get("url") or None,
                        "grading": info.get("grading", ""),
                    }
                )
        except Exception as exc:  # never break the UI over a listing
            logger.warning("Could not read the manifest: %s", exc)
        if not docs and self._lexical is not None:
            docs = [{"title": n, "publisher": "", "pages": None, "url": None, "grading": ""} for n in self._lexical.document_names()]
        return sorted(docs, key=lambda d: (d["publisher"], d["title"]))

    def publishers(self) -> list[dict[str, Any]]:
        """Distinct publishers with their document counts (the choices for compare mode)."""
        counts: dict[str, int] = {}
        for d in self.documents():
            if d.get("publisher"):
                counts[d["publisher"]] = counts.get(d["publisher"], 0) + 1
        return [{"name": n, "documents": c} for n, c in sorted(counts.items())]

    def _clean_compare(self, compare: Optional[list[str]]) -> Optional[list[str]]:
        if not compare:
            return None
        names = list(dict.fromkeys(" ".join(str(c).split()) for c in compare if str(c).strip()))
        if len(names) < 2:
            raise BadRequestError("Pick at least two publishers to compare.")
        if len(names) > 3:
            raise BadRequestError("You can compare at most three publishers at once.")
        known = {p["name"] for p in self.publishers()}
        unknown = [n for n in names if known and n not in known]
        if unknown:
            raise BadRequestError("Unknown publisher: " + ", ".join(unknown))
        return names

    # ----- main entry points ---------------------------------------------------
    def query(
        self, question: str, history: Optional[list[dict[str, str]]] = None, compare: Optional[list[str]] = None
    ) -> dict[str, Any]:
        prep = self._prepare(question, history, compare)
        if prep.early is not None:
            return prep.early
        _, llm, _ = self._components()
        answer = _complete(llm, prep.prompt)
        if not answer:
            raise LLMError("The language model returned an empty response. Please try again.")
        return self._finish(prep, answer)

    def query_stream(
        self, question: str, history: Optional[list[dict[str, str]]] = None, compare: Optional[list[str]] = None
    ) -> Iterator[tuple[str, Any]]:
        """Yield ``("status", {...})``, ``("token", text)`` ... and finally ``("final", result)``.

        Errors are raised (as :class:`MedHelpError`) from the generator; the API layer turns them into an ``error`` event.
        """
        yield "status", {"stage": "searching"}
        prep = self._prepare(question, history, compare)
        if prep.early is not None:
            yield "final", prep.early
            return
        _, llm, _ = self._components()
        yield "status", {"stage": "generating", "passages": len(prep.sources)}
        parts: list[str] = []
        for piece in _stream(llm, prep.prompt):
            parts.append(piece)
            yield "token", piece
        answer = "".join(parts).strip()
        if not answer:
            raise LLMError("The language model returned an empty response. Please try again.")
        yield "status", {"stage": "checking"}
        yield "final", self._finish(prep, answer)

    # ----- stage 1: everything before generation ---------------------------------
    def _prepare(self, question: str, history: Optional[list[dict[str, str]]], compare: Optional[list[str]]) -> Prepared:
        started = time.perf_counter()
        question = " ".join((question or "").split())
        if not question:
            raise ValueError("Question must not be empty.")
        publishers = self._clean_compare(compare)
        mode = "compare" if publishers else "single"

        def early(answer: str, **kw: Any) -> Prepared:
            return Prepared(started, question, question, early=self._result(started, answer=answer, mode=mode, **kw))

        # 0. Small talk ------------------------------------------------------
        talk = detect_smalltalk(question)
        if talk:
            return early(self._smalltalk_reply(talk), kind="smalltalk", debug={"smalltalk": talk})

        # 1. Guardrail -------------------------------------------------------
        guard = check_guardrails(question)
        if guard.triggered:
            logger.info("Guardrail triggered (%s)", guard.category)
            return early(guard.message or REFUSAL_MESSAGE, guardrail_triggered=True)

        if not self.is_ready():
            raise IndexNotReadyError(
                "The guideline index is empty or unreachable. Run `python -m app.ingestion` on your own machine "
                "(see the README) and try again."
            )
        retriever, _, fast_llm = self._components()

        # 2. Understand follow-ups ------------------------------------------
        search_question = self._condense(fast_llm, question, history)

        # 3. Embed once: the vector serves both the cache lookup and retrieval -------
        vector: Optional[list[float]] = None
        if self._embedder is not None:
            try:
                vector = self._embedder.embed_query(search_question)
            except MedHelpError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"Could not embed the question: {str(exc)[:160]}") from exc

        scope = "single" if not publishers else "compare:" + "|".join(sorted(publishers))
        # A follow-up whose rewrite failed is ambiguous ("what about children?"): never cache or serve it from cache.
        cacheable = self._cache is not None and not (history and looks_like_followup(question) and search_question == question)
        if cacheable:
            hit = self._cache.get(scope, search_question, vector)
            if hit is not None:
                similarity = hit.pop("_cache_similarity", None)
                hit["cached"] = True
                hit["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
                hit.setdefault("debug", {})["cache_similarity"] = similarity
                return Prepared(started, question, search_question, early=hit)

        # 4. Retrieve (hybrid) ----------------------------------------------
        if publishers:
            selected, debug, missing = self._retrieve_compare(retriever, search_question, vector, publishers)
            if missing:
                debug["reason"] = "compare_missing_publisher"
                return Prepared(
                    started, question, search_question,
                    early=self._result(
                        started, answer=NO_ANSWER_TEXT, mode=mode, debug=debug,
                        warnings=[f"No relevant passages were found for: {', '.join(missing)}. Try a different topic or publishers."],
                    ),
                )
        else:
            selected, debug = self._retrieve(retriever, search_question, vector)
        if search_question != question:
            debug["standalone_question"] = search_question
        if not selected:
            debug["reason"] = debug.get("reason") or "no_relevant_passages"
            return Prepared(
                started, question, search_question,
                early=self._result(started, answer=NO_ANSWER_TEXT, mode=mode, debug=debug, warnings=self._index_warnings(debug)),
            )

        sources = [self._source_dict(h) for h in selected]
        if publishers:
            context = self._compare_context(selected, publishers)
            prompt = build_compare_prompt(context, search_question, publishers)
        else:
            context = "\n\n".join(f"[{h.doc}, Page {h.page}]\n{h.text}" for h in selected)
            prompt = build_qa_prompt(context, search_question)
        return Prepared(
            started, question, search_question, prompt=prompt, context=context, sources=sources, debug=debug,
            compare=publishers, vector=vector, cache_scope=scope, cacheable=cacheable,
        )

    @staticmethod
    def _compare_context(selected: list[Hit], publishers: list[str]) -> str:
        blocks: list[str] = []
        for i, pub in enumerate(publishers, start=1):
            blocks.append(f"=== GROUP {i}: {pub} ===")
            blocks.extend(f"[{h.doc}, Page {h.page}]\n{h.text}" for h in selected if str(h.meta.get("publisher", "")) == pub)
        return "\n\n".join(blocks)

    # ----- stage 2: checks, extras, caching ------------------------------------------
    def _finish(self, prep: Prepared, answer: str) -> dict[str, Any]:
        mode = "compare" if prep.compare else "single"
        debug = prep.debug
        if is_refusal(answer):
            debug["reason"] = "model_declined"
            # Still show the closest passages so the user can judge for themselves.
            return self._result(
                prep.started, answer=NO_ANSWER_TEXT, mode=mode, sources=prep.sources[:3], debug=debug,
                warnings=self._index_warnings(debug),
            )

        warnings: list[str] = self._index_warnings(debug)
        if not _CITATION_RE.search(answer):
            warnings.append("The answer does not contain a [Document, Page X] citation - verify it against the sources below.")
        unverified = find_unverified_grades(answer, prep.context)
        if unverified:
            warnings.append(
                "Evidence grade(s) mentioned in the answer could not be matched verbatim to the retrieved "
                f"passages ({', '.join(unverified)}). Check the source text before relying on them."
            )
        verification: list[dict[str, Any]] = []
        if self._verify:
            verification = verify_citations(answer, prep.sources)
            note = verification_warning(verification)
            if note:
                warnings.append(note)

        # Jargon definitions and follow-up suggestions are independent: run them side by side.
        _, _, fast_llm = self._components()
        jargon: dict[str, str] = {}
        followups: list[str] = []
        jargon_warning: Optional[str] = None
        with ThreadPoolExecutor(max_workers=2) as pool:
            jargon_future = pool.submit(self._jargon, fast_llm, answer)
            followup_future = pool.submit(self._suggest_followups, fast_llm, prep.search_question, answer) if self._followups else None
            jargon, jargon_warning = jargon_future.result()
            if followup_future is not None:
                followups = followup_future.result()
        if jargon_warning:
            warnings.append(jargon_warning)

        result = self._result(
            prep.started,
            answer=answer,
            mode=mode,
            jargon_dictionary=jargon,
            suggested_followups=followups,
            verification=verification,
            sources=prep.sources,
            answered=True,
            warnings=warnings,
            debug=debug,
        )
        if self._cache is not None and prep.cacheable:
            self._cache.put(prep.cache_scope, prep.search_question, prep.vector, result)
        return result

    def _jargon(self, fast_llm: Any, answer: str) -> tuple[dict[str, str], Optional[str]]:
        try:
            return parse_jargon_json(_complete(fast_llm, JARGON_INSTRUCTIONS + strip_citations(answer))), None
        except MedHelpError as exc:  # never fail the whole request because of the optional step
            logger.warning("Jargon simplifier unavailable: %s", exc)
            return {}, "Plain-language definitions are temporarily unavailable."

    def _suggest_followups(self, fast_llm: Any, question: str, answer: str) -> list[str]:
        try:
            return parse_followups(_complete(fast_llm, build_followup_prompt(question, answer)), exclude=question)
        except MedHelpError as exc:
            logger.warning("Follow-up suggestions unavailable: %s", exc)
            return []

    # ----- pipeline stages -----------------------------------------------------
    def _condense(self, fast_llm: Any, question: str, history: Optional[list[dict[str, str]]]) -> str:
        """Rewrite a follow-up ("what about children?") into a standalone search question."""
        turns = [t for t in (history or []) if str(t.get("content", "")).strip()][-config.HISTORY_MESSAGES :]
        if not turns or not looks_like_followup(question):
            return question
        try:
            rewritten = _complete(fast_llm, build_condense_prompt(turns, question))
        except MedHelpError as exc:
            logger.warning("Follow-up rewriting unavailable, using the raw question: %s", exc)
            return question
        first_line = rewritten.strip().splitlines()[0].strip().strip("\"'\u201c\u201d") if rewritten.strip() else ""
        return first_line if 5 <= len(first_line) <= 400 else question

    def _retrieve(
        self,
        retriever: Any,
        question: str,
        vector: Optional[list[float]] = None,
        publisher: Optional[str] = None,
        top_k: Optional[int] = None,
        budget: Optional[int] = None,
        rerank: bool = True,
    ) -> tuple[list[Hit], dict[str, Any]]:
        kwargs = {"publisher": publisher} if publisher else {}
        try:
            if vector is not None and hasattr(retriever, "retrieve_vector"):
                retrieved = retriever.retrieve_vector(vector, **kwargs)
            else:
                retrieved = retriever.retrieve(question, **kwargs)
        except MedHelpError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Retrieval from the vector database failed: {str(exc)[:200]}") from exc

        candidates = [_hit_from_node(n) for n in retrieved]
        dense: list[Hit] = []
        below_threshold = reference_chunks = 0
        for hit in candidates:
            if (hit.score or 0.0) < self.min_similarity:
                below_threshold += 1
            elif textutils.is_reference_text(hit.text):
                reference_chunks += 1  # bibliography pages rank high for any topical query but never answer it
            else:
                dense.append(hit)

        keyword = self._keyword_search(question, publisher)
        fused = fuse_hits(dense, keyword)
        reranked: Optional[bool] = None
        rerank_error: Optional[str] = None
        if rerank and self._reranker is not None and len(fused) > 1:
            fused, reranked, rerank_error = self._rerank(question, fused)
        selected = pack_hits(fused, top_k or self.top_k, budget or config.CONTEXT_CHAR_BUDGET)

        top_score = max((h.score for h in candidates if h.score is not None), default=None)
        debug: dict[str, Any] = {
            "answer_style": config.ANSWER_STYLE,
            "semantic_candidates": len(candidates),
            "below_similarity_threshold": below_threshold,
            "reference_chunks_dropped": reference_chunks,
            "keyword_hits": len(keyword),
            "passages_used": len(selected),
            "top_similarity": round(top_score, 4) if top_score is not None else None,
            "min_similarity": self.min_similarity,
        }
        if reranked is not None:
            debug["reranked"] = reranked
        if rerank_error:
            debug["rerank_error"] = rerank_error
        stale = self._index_warning(candidates)
        if stale:
            debug["index_warning"] = stale
        if not selected:
            if candidates and reference_chunks and not dense and not keyword:
                debug["reason"] = "only_reference_pages"
            elif candidates and below_threshold == len(candidates):
                debug["reason"] = "below_similarity_threshold"
            else:
                debug["reason"] = "no_relevant_passages"
        return selected, debug

    def _rerank(self, question: str, fused: list[Hit]) -> tuple[list[Hit], bool, Optional[str]]:
        """Re-order the best ``RERANK_CANDIDATES`` hits with the API reranker; any failure keeps the fused order."""
        n = max(2, config.RERANK_CANDIDATES)
        head, tail = fused[:n], fused[n:]
        try:
            order = self._reranker.rerank(question, [h.text for h in head], top_n=len(head))
            indexes = list(dict.fromkeys(i for i, _ in order if 0 <= i < len(head)))
            indexes += [i for i in range(len(head)) if i not in indexes]
            return [head[i] for i in indexes] + tail, True, None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker unavailable, keeping the fused order: %s", exc)
            return fused, False, str(exc)[:120]

    def _retrieve_compare(
        self, retriever: Any, question: str, vector: Optional[list[float]], publishers: list[str]
    ) -> tuple[list[Hit], dict[str, Any], list[str]]:
        """Retrieve separately inside each publisher and split the *existing* TOP_K / context budget between them.

        Returns ``(selected_hits, debug, publishers_without_passages)``.
        """
        n = len(publishers)
        per_k = max(2, math.ceil(self.top_k / n))
        per_budget = config.CONTEXT_CHAR_BUDGET // n
        selected: list[Hit] = []
        per_pub: dict[str, Any] = {}
        missing: list[str] = []
        for pub in publishers:
            hits, dbg = self._retrieve(retriever, question, vector, publisher=pub, top_k=per_k, budget=per_budget, rerank=False)
            for h in hits:
                h.meta.setdefault("publisher", pub)
            selected.extend(hits)
            per_pub[pub] = {"passages_used": len(hits), "reason": dbg.get("reason"), "top_similarity": dbg.get("top_similarity")}
            if not hits:
                missing.append(pub)
        debug = {
            "answer_style": config.ANSWER_STYLE,
            "mode": "compare",
            "per_publisher": per_pub,
            "passages_used": len(selected),
            "min_similarity": self.min_similarity,
        }
        return selected, debug, missing

    @staticmethod
    def _index_warnings(debug: dict[str, Any]) -> list[str]:
        return [debug["index_warning"]] if debug.get("index_warning") else []

    @staticmethod
    def _index_warning(candidates: list[Hit]) -> Optional[str]:
        """Detect an index written by an old/broken ingestion (whole-PDF documents, no page numbers).

        Symptoms: chunks without the ``ingest_version`` stamp, and almost every hit labelled "page 1". Needs at
        least 5 candidates so tiny result sets are never judged.
        """
        if len(candidates) < 5:
            return None
        from app.qdrant_store import INGEST_VERSION

        unstamped = sum(1 for h in candidates if str(h.meta.get("ingest_version", "")) != INGEST_VERSION)
        page_one = sum(1 for h in candidates if h.page == "1")
        if unstamped == len(candidates) and page_one / len(candidates) >= 0.6:
            return (
                f"{page_one} of {len(candidates)} retrieved passages are labelled 'page 1' and none carry the current "
                "ingestion stamp: the index was built by an older ingestion that lost page numbers and kept "
                "bibliography pages. Rebuild it with `python -m app.ingestion --recreate`, then restart the backend."
            )
        if unstamped == len(candidates):
            return (
                "The index was built by an older ingestion version. Rebuild it with "
                "`python -m app.ingestion --recreate` for cleaner retrieval and correct page numbers."
            )
        return None

    def _keyword_search(self, question: str, publisher: Optional[str] = None) -> list[Hit]:
        lexical = self._lexical  # may be swapped by the background loader at any moment
        if lexical is None:
            return []
        try:
            kwargs: dict[str, Any] = {}
            k = config.LEXICAL_TOP_K
            if publisher:
                if getattr(lexical, "supports_filters", False):
                    kwargs["publisher"] = publisher
                else:
                    k *= 4  # filter afterwards, so over-fetch
            found = lexical.search(question, k=k, **kwargs)
            if publisher:
                found = [h for h in found if str(h.meta.get("publisher", "")) == publisher][: config.LEXICAL_TOP_K]
            return [Hit(id=h.id, text=h.text, meta=dict(h.meta), score=None, keyword=True) for h in found]
        except Exception as exc:  # keyword search is a bonus, never a failure point
            logger.warning("Keyword search failed: %s", exc)
            return []

    # ----- helpers -----------------------------------------------------------
    @staticmethod
    def _source_dict(hit: Hit) -> dict[str, Any]:
        pdf_page = hit.meta.get("pdf_page")
        return {
            "source_text": hit.text,
            "document_name": hit.doc,
            "publisher": str(hit.meta.get("publisher") or "") or None,
            "page_number": hit.page,
            "pdf_page": int(pdf_page) if str(pdf_page or "").isdigit() else None,
            "relevance_score": round(hit.score, 4) if hit.score is not None else None,
            "match_type": hit.match_type,
            "source_url": hit.meta.get("source_url") or None,
            "evidence_grades": extract_grades(hit.text),
        }

    def _smalltalk_reply(self, kind: str) -> str:
        if kind == "greeting":
            return (
                "Hi there! \U0001F44B I'm MedHelp. I answer questions using open-access clinical guidelines "
                "(WHO, CDC, ACC/AHA and more), with citations and evidence grades. What would you like to know?"
            )
        if kind == "thanks":
            return "You're welcome! Ask me anything else about the guidelines whenever you like."
        if kind == "bye":
            return "Goodbye, and take care! I'm here whenever you have another guideline question."
        if kind == "documents":
            docs = self.documents()
            if docs:
                lines = [f"- {d['title']}" + (f" _({d['publisher']})_" if d.get("publisher") else "") for d in docs[:60]]
                return f"I can search **{len(docs)} guideline documents**:\n\n" + "\n".join(lines) + (
                    "\n\nAsk about any of these topics - for example treatment recommendations, screening "
                    "thresholds or evidence grades."
                )
        return CAPABILITIES_TEXT

    @staticmethod
    def _result(started: float, answer: str, **overrides: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "answer": answer,
            "kind": None,
            "mode": "single",
            "jargon_dictionary": {},
            "suggested_followups": [],
            "verification": [],
            "sources": [],
            "guardrail_triggered": False,
            "answered": False,
            "cached": False,
            "session_id": None,
            "warnings": [],
            "debug": {},
            "disclaimer": DISCLAIMER,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        result.update(overrides)
        if not result["kind"]:
            result["kind"] = (
                "guardrail" if result["guardrail_triggered"] else "answer" if result["answered"] else "no_answer"
            )
        return result


def _build_groq(model: str, max_tokens: Optional[int] = None) -> Any:
    """Groq client (deterministic; reasoning effort only for gpt-oss models)."""
    from app.llm import build_groq

    return build_groq(model, max_tokens)
