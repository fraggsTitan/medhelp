"""Pure helpers for the MedHelp UI (no Streamlit calls, so they are easy to test)."""
from __future__ import annotations

import re
import secrets
from typing import Any, Optional

HISTORY_SENT = 8  # previous messages sent along; used by the server only to recover a lost session

GRADE_LINE_RE = re.compile(r"(?im)^[\s>*_-]*Strength of Recommendation\b")
CITATION_RE = re.compile(r"\[([^\[\]]{2,200}?),\s*(?:Page|p\.?)\s*([^\[\]]{1,20})\]", re.IGNORECASE)
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")

STATUS_ICONS = {"supported": "\u2705", "partial": "\U0001F7E1", "unsupported": "\u274c", "source_not_found": "\u2753"}
STATUS_LABELS = {
    "supported": "supported by the cited passage",
    "partial": "partly supported",
    "unsupported": "not clearly supported by the cited passage",
    "source_not_found": "cited passage not found among the retrieved sources",
}


def new_session_id() -> str:
    return secrets.token_urlsafe(18)


def valid_session_id(value: Any) -> bool:
    return isinstance(value, str) and bool(SESSION_ID_RE.match(value))


def escape_md(text: str) -> str:
    """Stop Streamlit from interpreting '$' as LaTeX in medical text."""
    return text.replace("$", r"\$")


def split_answer(answer: str) -> tuple[str, Optional[str]]:
    """Separate the body from the trailing 'Strength of Recommendation' block."""
    match = GRADE_LINE_RE.search(answer)
    if not match:
        return answer.strip(), None
    return answer[: match.start()].strip(), answer[match.start():].strip()


def number_citations(text: str, sources: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Replace ``[Document, Page X]`` with ``**[n]**`` and return the numbered reference list.

    Identical (document, page) pairs share a number. Links are filled in from *sources* when the cited
    document name matches one of them.
    """
    refs: list[dict[str, Any]] = []
    index: dict[tuple[str, str], int] = {}

    def url_for(doc: str) -> Optional[str]:
        low = doc.strip().lower()
        for src in sources:
            name = str(src.get("document_name", "")).lower()
            if name and (name == low or name in low or low in name):
                return src.get("source_url")
        return None

    def repl(match: re.Match[str]) -> str:
        doc, page = match.group(1).strip(), match.group(2).strip()
        key = (doc.lower(), page.lower())
        if key not in index:
            index[key] = len(refs) + 1
            refs.append({"n": index[key], "document": doc, "page": page, "url": url_for(doc)})
        return f"**[{index[key]}]**"

    return CITATION_RE.sub(repl, text), refs


def history_for_api(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Previous chat turns in the shape the backend expects (errors, guardrail refusals and empty messages left out)."""
    turns = [
        {"role": m["role"], "content": str(m.get("content", ""))[:6000]}
        for m in messages
        if m.get("content") and not m.get("error") and (m.get("result") or {}).get("kind") not in ("guardrail", "smalltalk")
    ]
    return turns[-HISTORY_SENT:]


def messages_from_turns(turns: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Rebuild chat messages from the server's text-only session history (sources are not restored)."""
    return [{"role": t["role"], "content": t["content"], "restored": True} for t in turns if t.get("role") in ("user", "assistant")]


def friendly_error(status: Optional[int], detail: str) -> str:
    if status == 401:
        return "The backend rejected the API key. Check API_KEY in the frontend settings."
    if status == 429:
        return "I'm being rate-limited (the free tiers allow only a few questions per minute). " + detail
    if status == 503:
        return "The backend isn't ready yet (it may still be waking up or loading its index). " + detail
    return detail


def compare_caption(publishers: list[str]) -> str:
    return "\u2696\ufe0f Comparing: " + "  vs  ".join(publishers)


def source_label(src: dict[str, Any]) -> str:
    page = f"page {src['page_number']}"
    if src.get("pdf_page") and str(src["pdf_page"]) != str(src["page_number"]):
        page += f" (PDF page {src['pdf_page']})"
    return page


def verification_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {k: 0 for k in STATUS_ICONS}
    for r in results:
        counts[r.get("status", "unsupported")] = counts.get(r.get("status", "unsupported"), 0) + 1
    return counts


def verification_summary(results: list[dict[str, Any]]) -> str:
    c = verification_counts(results)
    bad = c["unsupported"] + c["source_not_found"]
    head = f"\U0001F50D Citation check: {c['supported']}/{len(results)} supported"
    return head + (f", {bad} to verify" if bad else "")
