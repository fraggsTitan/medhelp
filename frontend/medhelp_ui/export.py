"""Export a conversation as Markdown or PDF (built in the front-end, so the backend stays small).

Both formats are produced from the same intermediate "blocks", so they always agree. The PDF uses fpdf2's
built-in Helvetica (Latin-1 only): common medical symbols are transliterated (>= for the greater-or-equal sign
and so on) and anything else becomes '?'.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from medhelp_ui.helpers import number_citations, split_answer

DISCLAIMER = (
    "MedHelp summarises published clinical guidelines for educational purposes only. "
    "It is not medical advice and does not replace a qualified healthcare professional."
)


def _plain(text: str) -> str:
    return text.replace("**[", "[").replace("]**", "]").replace("**", "").replace("__", "")


def conversation_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One block per chat message, with citations numbered and the evidence trail attached."""
    blocks: list[dict[str, Any]] = []
    for m in messages:
        if m.get("error") or not str(m.get("content", "")).strip():
            continue
        if m["role"] == "user":
            blocks.append({"role": "user", "body": str(m["content"]).strip()})
            continue
        result = m.get("result") or {}
        block: dict[str, Any] = {"role": "assistant", "kind": result.get("kind"), "grade": None, "refs": [], "sources": [], "notes": []}
        if result.get("kind") in (None, "smalltalk", "guardrail", "no_answer"):
            block["body"] = _plain(str(m["content"]).strip())
        else:
            numbered, refs = number_citations(str(m["content"]), result.get("sources", []))  # one numbering for body + grade
            body, grade = split_answer(numbered)
            block["body"] = _plain(body)
            block["grade"] = _plain(grade) if grade else None
            block["refs"] = [
                f"[{r['n']}] {r['document']}, page {r['page']}" + (f" - {r['url']}" if r.get("url") else "") for r in refs
            ]
        if result.get("mode") == "compare" and result.get("sources"):
            pubs = list(dict.fromkeys(s.get("publisher") for s in result["sources"] if s.get("publisher")))
            if pubs:
                block["notes"].append("Comparison of: " + " vs ".join(pubs))
        verification = result.get("verification") or []
        if verification:
            ok = sum(1 for v in verification if v.get("status") == "supported")
            block["notes"].append(f"Automatic citation check: {ok} of {len(verification)} cited statements matched their passage.")
        block["notes"].extend(result.get("warnings") or [])
        block["sources"] = [
            f"{s['document_name']}" + (f" ({s['publisher']})" if s.get("publisher") else "") + f", page {s['page_number']}"
            for s in result.get("sources", [])
        ]
        blocks.append(block)
    return blocks


def conversation_markdown(messages: list[dict[str, Any]], generated_at: Optional[str] = None) -> str:
    stamp = generated_at or time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    out = ["# MedHelp conversation", "", f"_Exported {stamp}_", ""]
    for b in conversation_blocks(messages):
        if b["role"] == "user":
            out += ["## You", "", b["body"], ""]
            continue
        out += ["## MedHelp", "", b["body"], ""]
        if b.get("grade"):
            out += [f"> {b['grade']}", ""]
        if b["refs"]:
            out += ["**References**", ""] + [f"- {r}" for r in b["refs"]] + [""]
        if b["notes"]:
            out += ["**Notes**", ""] + [f"- {n}" for n in b["notes"]] + [""]
        if b["sources"]:
            out += ["**Passages consulted**", ""] + [f"- {s}" for s in dict.fromkeys(b["sources"])] + [""]
    out += ["---", "", f"_{DISCLAIMER}_", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# PDF (fpdf2)
# --------------------------------------------------------------------------- #
_LATIN1_MAP = {
    "\u2265": ">=", "\u2264": "<=", "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u2026": "...", "\u00a0": " ", "\u2192": "->", "\u2191": "(up)",
    "\u2193": "(down)", "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma", "\u03bc": "u", "\u2248": "~",
    "\u00b1": "+/-", "\u2713": "ok", "\u2705": "", "\ufe0f": "", "\u200b": "",
}


def to_latin1(text: str) -> str:
    for src, dst in _LATIN1_MAP.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def soften_long_tokens(text: str, limit: int = 60) -> str:
    """fpdf2 cannot wrap a single 'word' wider than the page (long URLs): add break opportunities."""
    out = []
    for word in text.split(" "):
        while len(word) > limit:
            out.append(word[:limit])
            word = word[limit:]
        out.append(word)
    return " ".join(out)


def conversation_pdf(messages: list[dict[str, Any]]) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError("PDF export needs fpdf2 (pip install fpdf2).") from exc

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    def write(text: str, style: str = "", size: int = 10.5, gap: float = 1.5) -> None:
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, size * 0.52, soften_long_tokens(to_latin1(text)), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(gap)

    write("MedHelp conversation", "B", 16, 1)
    write("Exported " + time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "I", 9, 4)
    for b in conversation_blocks(messages):
        if b["role"] == "user":
            write("You", "B", 11, 0.5)
            write(b["body"])
            continue
        write("MedHelp", "B", 11, 0.5)
        write(b["body"])
        if b.get("grade"):
            write(b["grade"], "B", 10)
        for r in b["refs"]:
            write(r, "", 9, 0.3)
        for n in b["notes"]:
            write("Note: " + n, "I", 9, 0.3)
        if b["sources"]:
            write("Passages consulted: " + "; ".join(dict.fromkeys(b["sources"])), "", 8.5, 3)
        pdf.ln(2)
    write(DISCLAIMER, "I", 8.5)
    return bytes(pdf.output())
