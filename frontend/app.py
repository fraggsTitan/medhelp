"""MedHelp Streamlit front-end.

Run from the ``frontend/`` folder::

    streamlit run app.py

Settings (environment variables, or Streamlit secrets when deployed on Streamlit Community Cloud):

    API_URL   address of the MedHelp backend, e.g. https://medhelp-api.onrender.com   (default http://localhost:8000)
    API_KEY   the shared secret the backend expects in its X-API-Key header (only if you set API_KEY there)

Features
--------
* Real token streaming of answers, numbered citations, "Strength of Recommendation" callout.
* Per-browser-tab chat sessions kept on the server (the tab's id lives in the URL as ?sid=..., so a refresh resumes).
* Compare mode: ask one question across 2-3 publishers' guidelines, side by side.
* Citation check (does each cited sentence match its passage?), suggested follow-up questions.
* Voice questions (record, review the transcript, send) and Markdown / PDF export of the conversation.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Iterator, Optional

import streamlit as st

from medhelp_ui.client import ApiError, MedHelpClient
from medhelp_ui.export import conversation_markdown, conversation_pdf
from medhelp_ui.helpers import (
    STATUS_ICONS,
    STATUS_LABELS,
    compare_caption,
    escape_md,
    history_for_api,
    messages_from_turns,
    new_session_id,
    number_citations,
    source_label,
    split_answer,
    valid_session_id,
    verification_summary,
)

SUGGESTIONS: list[tuple[str, str]] = [
    ("\u2764\ufe0f Blood pressure", "What does WHO recommend as first-line drug treatment for hypertension in adults?"),
    ("\U0001F4A7 Diabetes", "What HbA1c target does the ADA recommend for most adults with diabetes?"),
    ("\U0001FAC1 Asthma", "How does GINA recommend treating mild asthma in adults?"),
    ("\U0001F9EA Kidney disease", "How does KDIGO classify chronic kidney disease and when should patients be referred?"),
    ("\U0001F48A Opioids", "What does the CDC say about prescribing opioids for acute pain?"),
    ("\U0001F9A0 Tuberculosis", "What treatment regimen does WHO recommend for drug-susceptible tuberculosis?"),
]

NO_ANSWER_REASONS = {
    "no_relevant_passages": "Nothing in my knowledge base matched your question closely enough.",
    "below_similarity_threshold": "The closest passages were not similar enough to your question.",
    "only_reference_pages": "The only matches were bibliography pages, which can't answer a question.",
    "model_declined": "I found some passages, but they don't answer this directly.",
    "compare_missing_publisher": "At least one of the selected publishers has nothing on this topic.",
}

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 52rem;}
div[data-testid="stChatMessage"] {border-radius: 14px;}
.medhelp-hero {text-align: center; padding: 2.2rem 0 0.6rem 0;}
.medhelp-hero h1 {font-size: 2.3rem; margin-bottom: 0.2rem;}
.medhelp-hero p {color: #6b7280; font-size: 1.02rem; margin: 0;}
div[data-testid="stButton"] > button {border-radius: 12px; text-align: left;}
</style>
"""


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if not value:
        try:
            value = st.secrets.get(name)  # type: ignore[assignment]
        except Exception:  # no secrets file
            value = None
    return (value or default).strip()


API_URL = _setting("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = _setting("API_KEY")


@st.cache_resource(show_spinner=False)
def get_client(api_url: str, api_key: str) -> MedHelpClient:
    return MedHelpClient(api_url, api_key)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_health(api_url: str, api_key: str) -> dict[str, Any]:
    return get_client(api_url, api_key).health()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_documents(api_url: str, api_key: str) -> dict[str, Any]:
    return get_client(api_url, api_key).documents()


client = get_client(API_URL, API_KEY)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_sources(sources: list[dict[str, Any]], key: str, label: str) -> None:
    with st.expander(label):
        for i, src in enumerate(sources, start=1):
            match = f" \u00b7 similarity {src['relevance_score']:.2f}" if src.get("relevance_score") is not None else " \u00b7 keyword match"
            who = f" ({escape_md(src['publisher'])})" if src.get("publisher") else ""
            st.markdown(f"**{i}. {escape_md(src['document_name'])}**{who} \u2014 {source_label(src)}{match}")
            if src.get("evidence_grades"):
                st.markdown("Evidence grades in this passage: " + ", ".join(f"`{g}`" for g in src["evidence_grades"]))
            st.text_area("Exact passage", value=src["source_text"], height=150, disabled=True, key=f"{key}_src_{i}", label_visibility="collapsed")
            if src.get("source_url"):
                st.markdown(f"[Open original document]({src['source_url']})")
            if i < len(sources):
                st.divider()


def render_glossary(jargon: dict[str, str]) -> None:
    with st.expander(f"\U0001F4D6 Plain-language glossary ({len(jargon)})"):
        for term, definition in jargon.items():
            st.markdown(f"**{escape_md(term)}** \u2014 {escape_md(definition)}")
        st.caption("AI-generated general explanations, not quoted from the guidelines.")


def render_verification(results: list[dict[str, Any]]) -> None:
    with st.expander(verification_summary(results)):
        st.caption(
            "An automatic, word-level check of each cited sentence against the passage it cites. "
            "It catches wrong citations and numbers that are not in the source; it cannot judge meaning."
        )
        for r in results:
            icon = STATUS_ICONS.get(r["status"], "\u2753")
            st.markdown(f"{icon} **{escape_md(r['document'])}, page {escape_md(r['page'])}** \u2014 {STATUS_LABELS.get(r['status'], r['status'])}")
            st.markdown(f"> {escape_md(r['claim'])}")
            if r.get("missing_numbers"):
                st.markdown("Not found in the passage: " + ", ".join(f"`{n}`" for n in r["missing_numbers"]))
            if r.get("evidence"):
                st.caption("Closest sentence in the passage: " + escape_md(r["evidence"]))


def render_no_answer(result: dict[str, Any], key: str) -> None:
    debug = result.get("debug") or {}
    reason = NO_ANSWER_REASONS.get(debug.get("reason", ""), NO_ANSWER_REASONS["no_relevant_passages"])
    st.markdown(f"**I couldn't find a reliable answer in the guidelines I have.** {reason}")
    st.markdown(
        "Things you can try:\n"
        "- Rephrase with the condition and organisation, e.g. *\"WHO recommendation for ...\"*\n"
        "- Ask *\"Which guidelines do you have?\"* to see what's covered\n"
        "- Ask about what a guideline says in general rather than about one person's situation"
    )
    for warning in result.get("warnings", []):
        st.error(warning, icon="\U0001F6E0\ufe0f")
    if result.get("sources"):
        render_sources(result["sources"], key, f"\U0001F4DA Closest passages I found ({len(result['sources'])})")


def render_assistant(msg: dict[str, Any], key: str) -> None:
    """Render one assistant message from the chat history."""
    result = msg.get("result") or {}
    kind = result.get("kind")

    if msg.get("error"):
        st.error(msg["content"], icon="\u26a0\ufe0f")
        return
    if kind == "guardrail":
        st.warning(escape_md(msg["content"]), icon="\U0001F6D1")
        return
    if kind == "smalltalk" or not result:
        st.markdown(escape_md(msg["content"]))
        if msg.get("restored"):
            st.caption("Restored from your session (sources are not kept after a refresh).")
        return
    if kind == "no_answer":
        render_no_answer(result, key)
    else:
        if result.get("mode") == "compare":
            pubs = list(dict.fromkeys(s["publisher"] for s in result.get("sources", []) if s.get("publisher")))
            if pubs:
                st.caption(compare_caption(pubs))
        numbered, refs = number_citations(msg["content"], result.get("sources", []))  # one numbering for body + grade line
        body, grade = split_answer(numbered)
        st.markdown(escape_md(body))
        if grade:
            not_stated = "not stated" in grade.lower()
            (st.info if not_stated else st.success)(escape_md(grade), icon="\U0001F3F7\ufe0f")
        if refs:
            lines = []
            for r in refs:
                label = f"{escape_md(r['document'])}, page {escape_md(r['page'])}"
                lines.append(f"**[{r['n']}]** " + (f"[{label}]({r['url']})" if r.get("url") else label))
            st.caption("  \n".join(lines))
        for warning in result.get("warnings", []):
            st.warning(warning, icon="\u26a0\ufe0f")
        if result.get("verification"):
            render_verification(result["verification"])
        if result.get("jargon_dictionary"):
            render_glossary(result["jargon_dictionary"])
        if result.get("sources"):
            render_sources(result["sources"], key, f"\U0001F4DA Sources ({len(result['sources'])})")

    if st.session_state.get("show_debug") and result.get("debug"):
        with st.expander("\U0001F50E Retrieval details"):
            st.json(result["debug"])
    notes = []
    if result.get("cached"):
        notes.append("\u26a1 from cache")
    if result.get("elapsed_ms"):
        notes.append(f"{result['elapsed_ms'] / 1000:.1f}s")
    if notes:
        st.caption(" \u00b7 ".join(notes))


def render_followups(result: dict[str, Any], key: str) -> None:
    suggestions = result.get("suggested_followups") or []
    if not suggestions:
        return
    st.caption("Ask next:")
    for i, question in enumerate(suggestions):
        if st.button(f"\u21aa {question}", key=f"{key}_fu_{i}"):
            st.session_state["pending"] = question
            st.rerun()


def render_welcome() -> None:
    st.markdown(
        '<div class="medhelp-hero"><h1>\U0001FA7A MedHelp</h1>'
        "<p>Ask about clinical guidelines from WHO, CDC, ACC/AHA, KDIGO and more.<br>"
        "Every answer is cited and shows the guideline's own evidence grade.</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    cols = st.columns(2)
    for i, (label, question) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(f"{label}\n\n{question}", key=f"suggest_{i}", use_container_width=True):
                st.session_state["pending"] = question
                st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def new_chat() -> None:
    client.delete_session(st.session_state["session_id"])
    st.session_state.update(messages=[], pending=None, session_id=new_session_id(), voice_text="", voice_sig=None, pdf_bytes=None)
    st.query_params["sid"] = st.session_state["session_id"]


def _send_voice() -> None:  # runs as a button callback, i.e. before the next script run
    text = " ".join(str(st.session_state.get("voice_text", "")).split())
    if text:
        st.session_state["pending"] = text
    st.session_state["voice_text"] = ""


def _clear_voice() -> None:
    st.session_state["voice_text"] = ""


def render_voice() -> None:
    st.markdown("**\U0001F399\ufe0f Ask by voice**")
    if not hasattr(st, "audio_input"):
        st.caption("Voice input needs Streamlit 1.40 or newer.")
        return
    audio = st.audio_input("Record your question", key="voice_audio", label_visibility="collapsed")
    if audio is not None:
        data = audio.getvalue()
        signature = hashlib.sha1(data).hexdigest()
        if signature != st.session_state.get("voice_sig"):
            st.session_state["voice_sig"] = signature
            try:
                with st.spinner("Transcribing..."):
                    text = client.transcribe(data, "question.wav", getattr(audio, "type", None) or "audio/wav", st.session_state["session_id"])
                st.session_state["voice_text"] = text
                if not text:
                    st.warning("I couldn't hear anything. Try again closer to the microphone.")
            except ApiError as exc:
                st.error(str(exc), icon="\u26a0\ufe0f")
    if st.session_state.get("voice_text"):
        st.text_area("Transcript - fix any mistakes (drug names!) before sending", key="voice_text", height=90)
        c1, c2 = st.columns(2)
        c1.button("Send", key="voice_send", on_click=_send_voice, use_container_width=True)
        c2.button("Discard", key="voice_discard", on_click=_clear_voice, use_container_width=True)


def render_compare(publishers: list[dict[str, Any]]) -> Optional[list[str]]:
    """Compare-mode controls. Returns the chosen publishers when the mode is active and valid, else None."""
    st.toggle("\u2696\ufe0f Compare guidelines", key="compare_mode", help="Answer one question from 2-3 publishers side by side.")
    if not st.session_state.get("compare_mode"):
        return None
    names = [p["name"] for p in publishers]
    if not names:
        st.caption("Publisher list unavailable.")
        return None
    chosen = st.multiselect("Publishers (pick 2-3)", names, key="compare_pubs", max_selections=3)
    if len(chosen) < 2:
        st.caption("Pick at least two publishers; until then questions are answered normally.")
        return None
    return list(chosen)


def render_export() -> None:
    messages = st.session_state["messages"]
    if not any(m["role"] == "assistant" and not m.get("error") for m in messages):
        st.caption("Export becomes available after your first answer.")
        return
    st.download_button("\u2b07\ufe0f Download as Markdown", data=conversation_markdown(messages), file_name="medhelp-conversation.md", mime="text/markdown", use_container_width=True)
    if st.button("Prepare PDF", use_container_width=True):
        try:
            st.session_state["pdf_bytes"] = conversation_pdf(messages)
        except Exception as exc:  # noqa: BLE001 - missing fpdf2 or an odd character
            st.session_state["pdf_bytes"] = None
            st.error(f"PDF export failed: {exc}")
    if st.session_state.get("pdf_bytes"):
        st.download_button("\u2b07\ufe0f Download PDF", data=st.session_state["pdf_bytes"], file_name="medhelp-conversation.pdf", mime="application/pdf", use_container_width=True)


def render_sidebar() -> Optional[list[str]]:
    with st.sidebar:
        st.markdown("### \U0001FA7A MedHelp")
        if st.button("\u2795 New chat", use_container_width=True):
            new_chat()
            st.rerun()

        health = fetch_health(API_URL, API_KEY)
        if health.get("status") == "ok":
            st.success("Backend ready", icon="\u2705")
        elif health.get("status") == "degraded":
            st.warning("Backend is up but the guideline index is not ready yet.")
        elif health.get("status") == "misconfigured":
            st.error("Backend is missing settings: " + ", ".join(health.get("missing_settings", [])))
        else:
            st.error("Backend unreachable (free hosts need about a minute to wake up).")

        docs = fetch_documents(API_URL, API_KEY)
        doc_list = docs.get("documents", [])
        with st.expander(f"\U0001F4DA Knowledge base ({len(doc_list)})" if doc_list else "\U0001F4DA Knowledge base"):
            if not doc_list:
                st.caption("Document list unavailable.")
            publisher = None
            for d in doc_list:
                if d.get("publisher") != publisher:
                    publisher = d.get("publisher")
                    st.markdown(f"**{publisher or 'Other'}**")
                title = escape_md(d["title"])
                st.markdown(f"- [{title}]({d['url']})" if d.get("url") else f"- {title}")

        st.markdown("---")
        compare = render_compare(docs.get("publishers", []))
        st.markdown("---")
        render_voice()
        st.markdown("---")
        st.toggle("Stream answers", key="stream_answers", help="Show the answer as it is written. Turn off if your network blocks streaming.")
        st.toggle("Show retrieval details", key="show_debug", help="Diagnostics: what was retrieved for each answer.")
        render_export()
        st.caption(f"API: {API_URL}")
        st.caption(
            "Educational use only. MedHelp summarises published guidelines; it is not medical advice. "
            "In an emergency call your local emergency number."
        )
    return compare


# --------------------------------------------------------------------------- #
# Asking a question
# --------------------------------------------------------------------------- #
def token_stream(events: Iterator[tuple[str, Any]], holder: dict[str, Any], on_first_token: Any) -> Iterator[str]:
    """Adapt backend events for ``st.write_stream``: yield answer text, remember the final result."""
    first = True
    for event, data in events:
        if event == "token":
            if first:
                on_first_token()
                first = False
            yield escape_md(str(data))
        elif event == "final":
            holder["result"] = data


def ask(prompt: str, compare: Optional[list[str]]) -> None:
    messages = st.session_state["messages"]
    sid = st.session_state["session_id"]
    history = history_for_api(messages)
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="\U0001F9D1"):
        st.markdown(escape_md(prompt))

    with st.chat_message("assistant", avatar="\U0001FA7A"):
        holder: dict[str, Any] = {}
        status = st.empty()
        try:
            if st.session_state.get("stream_answers", True):
                status.caption("\U0001F50E Searching the guidelines...")
                st.write_stream(token_stream(client.stream_query(prompt, history, sid, compare), holder, status.empty))
                result = holder.get("result")
                if result is None:
                    raise ApiError("The answer stream ended unexpectedly. Please try again.")
            else:
                with st.spinner("Searching the guidelines..."):
                    result = client.query(prompt, history, sid, compare)
            status.empty()
            reply = {"role": "assistant", "content": result["answer"], "result": result}
        except ApiError as exc:
            status.empty()
            reply = {"role": "assistant", "content": str(exc), "error": True}
            st.error(str(exc), icon="\u26a0\ufe0f")
    messages.append(reply)
    st.rerun()  # redraw from history so the layout is identical to a reloaded page


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="MedHelp - Clinical Guideline Assistant", page_icon="\U0001FA7A", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

for _key, _default in (("messages", []), ("pending", None), ("stream_answers", True), ("show_debug", False),
                       ("compare_mode", False), ("voice_text", ""), ("voice_sig", None), ("pdf_bytes", None)):
    st.session_state.setdefault(_key, _default)

if "session_id" not in st.session_state:
    from_url = st.query_params.get("sid")
    if valid_session_id(from_url):
        st.session_state["session_id"] = from_url
        restored = client.get_session(from_url)  # a refresh (or a bookmarked link) resumes the conversation
        if restored:
            st.session_state["messages"] = messages_from_turns(restored)
    else:
        st.session_state["session_id"] = new_session_id()
st.query_params["sid"] = st.session_state["session_id"]

compare_publishers = render_sidebar()

typed = st.chat_input("Ask about a clinical guideline...")
prompt: Optional[str] = typed or st.session_state.pop("pending", None)
if prompt:
    prompt = prompt.strip()

if not st.session_state["messages"] and not prompt:
    render_welcome()

for i, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"], avatar="\U0001F9D1" if message["role"] == "user" else "\U0001FA7A"):
        if message["role"] == "user":
            st.markdown(escape_md(message["content"]))
        else:
            render_assistant(message, key=f"m{i}")

if not prompt and st.session_state["messages"]:
    last = st.session_state["messages"][-1]
    if last["role"] == "assistant" and last.get("result"):
        render_followups(last["result"], key=f"m{len(st.session_state['messages']) - 1}")

if prompt:
    ask(prompt, compare_publishers)
