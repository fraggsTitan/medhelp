"""Executes app.py end to end against a stub Streamlit and a patched API client."""
import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock

from medhelp_ui.client import ApiError, MedHelpClient
from tests.stub_streamlit import Audio, Rerun, Stub, installed

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SID = "S" * 24
RESULT = {
    "answer": "WHO recommends drug treatment at 140 mmHg [WHO HTN, Page 9].\n**Strength of Recommendation:** strong recommendation [WHO HTN, Page 9]",
    "kind": "answer", "mode": "single", "answered": True, "cached": True, "elapsed_ms": 1200, "session_id": SID,
    "sources": [{"document_name": "WHO HTN", "publisher": "WHO", "page_number": "9", "pdf_page": 13, "source_text": "text", "relevance_score": 0.8, "match_type": "semantic", "evidence_grades": ["strong recommendation"]}],
    "verification": [{"status": "supported", "document": "WHO HTN", "page": "9", "claim": "WHO recommends drug treatment", "missing_numbers": [], "evidence": "WHO recommends"},
                     {"status": "unsupported", "document": "WHO HTN", "page": "9", "claim": "x", "missing_numbers": ["160"], "evidence": ""}],
    "suggested_followups": ["What about children?", "Which drugs are first-line?"], "jargon_dictionary": {"systolic": "top number"},
    "warnings": ["Citation check: 1 cited statement(s) could not be matched."], "debug": {"reason": None},
}
HEALTH = {"status": "ok", "index_ready": True}
DOCS = {"documents": [{"title": "WHO HTN", "publisher": "WHO", "url": "http://who"}], "publishers": [{"name": "WHO", "documents": 1}, {"name": "ACC", "documents": 1}]}


def run_app(stub, stream=None, query=None, **patches):
    """Run app.py; returns the exception it ended with (Rerun is the normal end of an interaction)."""
    sys.modules.pop("app", None)
    defaults = dict(
        health=mock.patch.object(MedHelpClient, "health", return_value=HEALTH),
        documents=mock.patch.object(MedHelpClient, "documents", return_value=DOCS),
        delete=mock.patch.object(MedHelpClient, "delete_session", return_value=None),
    )
    if stream is not None:
        defaults["stream"] = mock.patch.object(MedHelpClient, "stream_query", stream)
    if query is not None:
        defaults["query"] = mock.patch.object(MedHelpClient, "query", query)
    for name, p in patches.items():
        defaults[name] = p
    started = [p.start() for p in defaults.values()]
    try:
        runpy.run_path(APP)
    except Rerun:
        return "rerun"
    finally:
        mock.patch.stopall()
    return None


def events_for(result, tokens=("WHO recommends ", "$5 treatment")):
    def gen(self, question, history, session_id=None, compare=None):
        gen.calls.append(dict(question=question, history=history, session_id=session_id, compare=compare))
        yield "status", {"stage": "searching"}
        for t in tokens:
            yield "token", t
        yield "final", result

    gen.calls = []
    return gen


class AppSmokeTests(unittest.TestCase):
    def test_first_visit_shows_welcome_and_creates_a_session_in_the_url(self):
        with installed(Stub()) as st:
            self.assertIsNone(run_app(st))
        self.assertTrue(any('class="medhelp-hero"' in t for t in st.texts("markdown")))
        self.assertEqual(st.query_params["sid"], st.session_state["session_id"])
        self.assertEqual(len(st.session_state["session_id"]), 24)

    def test_refresh_with_sid_in_the_url_restores_the_conversation(self):
        with installed(Stub()) as st:
            st.query_params["sid"] = SID
            restored = mock.patch.object(MedHelpClient, "get_session", return_value=[{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}])
            self.assertIsNone(run_app(st, get_session=restored))
        self.assertEqual(st.session_state["session_id"], SID)
        self.assertEqual([m["content"] for m in st.session_state["messages"]], ["q1", "a1"])
        self.assertFalse(any('class="medhelp-hero"' in t for t in st.texts("markdown")))

    def test_invalid_sid_in_the_url_is_replaced(self):
        with installed(Stub()) as st:
            st.query_params["sid"] = "not valid!"
            run_app(st)
        self.assertNotEqual(st.session_state["session_id"], "not valid!")

    def test_streamed_answer_is_stored_with_session_and_escaped_dollars(self):
        gen = events_for(RESULT)
        with installed(Stub()) as st:
            st.session_state["session_id"] = SID
            st.typed = "What does WHO recommend?"
            self.assertEqual(run_app(st, stream=gen), "rerun")
        self.assertEqual(gen.calls[0]["session_id"], SID)
        self.assertIsNone(gen.calls[0]["compare"])
        self.assertEqual(st.streamed, ["WHO recommends \\$5 treatment"])
        msgs = st.session_state["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["result"]["kind"], "answer")

    def test_non_streaming_mode_uses_plain_query(self):
        with installed(Stub()) as st:
            st.session_state.update(session_id=SID, stream_answers=False)
            st.typed = "a question"
            q = mock.Mock(return_value=RESULT)
            self.assertEqual(run_app(st, query=lambda self, *a, **k: q(*a, **k)), "rerun")
        q.assert_called_once()
        self.assertEqual(st.session_state["messages"][1]["content"], RESULT["answer"])

    def test_api_errors_become_a_readable_chat_message(self):
        def failing(self, *a, **k):
            raise ApiError("I'm being rate-limited. wait", 429)
            yield  # pragma: no cover

        with installed(Stub()) as st:
            st.session_state["session_id"] = SID
            st.typed = "a question"
            self.assertEqual(run_app(st, stream=failing), "rerun")
        reply = st.session_state["messages"][-1]
        self.assertTrue(reply["error"])
        self.assertIn("rate-limited", reply["content"])

    def test_stream_without_final_event_is_reported(self):
        def no_final(self, *a, **k):
            yield "token", "partial"

        with installed(Stub()) as st:
            st.session_state["session_id"] = SID
            st.typed = "a question"
            run_app(st, stream=no_final)
        self.assertTrue(st.session_state["messages"][-1]["error"])

    def test_compare_mode_sends_the_selected_publishers_only_when_two_or_more(self):
        for chosen, expected in ((["WHO", "ACC"], ["WHO", "ACC"]), (["WHO"], None)):
            gen = events_for(RESULT)
            with installed(Stub()) as st:
                st.session_state.update(session_id=SID, compare_mode=True, compare_pubs=chosen)
                st.typed = "blood pressure threshold"
                run_app(st, stream=gen)
            self.assertEqual(gen.calls[0]["compare"], expected)
            self.assertIn(["WHO", "ACC"], [v for n, v in st.calls if n == "multiselect"])

    def test_history_rendering_shows_checks_followups_and_cache_badge(self):
        with installed(Stub()) as st:
            st.session_state.update(session_id=SID, messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": RESULT["answer"], "result": RESULT}])
            run_app(st)
        labels = st.texts("expander")
        self.assertTrue(any("Citation check: 1/2 supported, 1 to verify" in l for l in labels))
        self.assertTrue(any("Sources (1)" in l for l in labels))
        self.assertTrue(any("glossary" in l for l in labels))
        buttons = st.texts("button")
        self.assertIn("\u21aa What about children?", buttons)
        self.assertTrue(any("from cache" in c for c in st.texts("caption")))
        self.assertTrue(any("**[1]**" in t for t in st.texts("markdown")))  # citations are numbered
        self.assertTrue(any(n == "download_button" and "Markdown" in v for n, v in st.calls))

    def test_clicking_a_followup_asks_it(self):
        with installed(Stub()) as st:
            st.session_state.update(session_id=SID, messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": RESULT["answer"], "result": RESULT}])
            st.clicked = {"\u21aa Which drugs are first-line?"}
            self.assertEqual(run_app(st), "rerun")
        self.assertEqual(st.session_state["pending"], "Which drugs are first-line?")

    def test_pending_question_is_asked_on_the_next_run(self):
        gen = events_for(RESULT)
        with installed(Stub()) as st:
            st.session_state.update(session_id=SID, pending="Which drugs are first-line?")
            run_app(st, stream=gen)
        self.assertEqual(gen.calls[0]["question"], "Which drugs are first-line?")

    def test_voice_recording_is_transcribed_into_an_editable_box_not_sent(self):
        with installed(Stub()) as st:
            st.session_state["session_id"] = SID
            st.audio = Audio()
            tr = mock.patch.object(MedHelpClient, "transcribe", return_value="what is the blood pressure target")
            gen = events_for(RESULT)
            self.assertIsNone(run_app(st, stream=gen, transcribe=tr))
        self.assertEqual(st.session_state["voice_text"], "what is the blood pressure target")
        self.assertIn("voice_text", st.texts("text_area"))
        self.assertEqual(gen.calls, [])  # nothing was sent to the backend yet
        self.assertEqual(st.session_state["messages"], [])

    def test_same_recording_is_not_transcribed_twice(self):
        with installed(Stub()) as st:
            st.session_state["session_id"] = SID
            st.audio = Audio()
            tr = mock.patch.object(MedHelpClient, "transcribe", return_value="hello there")
            run_app(st, transcribe=tr)
            with mock.patch.object(MedHelpClient, "transcribe", side_effect=AssertionError("transcribed again")):
                run_app(st)

    def test_sending_the_transcript_asks_it_and_clears_the_box(self):
        gen = events_for(RESULT)
        with installed(Stub()) as st:
            st.session_state.update(session_id=SID, voice_text="  what is   the target ")
            st.clicked = {"voice_send"}
            self.assertEqual(run_app(st, stream=gen), "rerun")
        self.assertEqual(gen.calls[0]["question"], "what is the target")
        self.assertEqual(st.session_state["voice_text"], "")

    def test_discarding_the_transcript_sends_nothing(self):
        gen = events_for(RESULT)
        with installed(Stub()) as st:
            st.session_state.update(session_id=SID, voice_text="oops wrong")
            st.clicked = {"voice_discard"}
            self.assertIsNone(run_app(st, stream=gen))
        self.assertEqual((gen.calls, st.session_state["voice_text"]), ([], ""))

    def test_voice_errors_are_shown(self):
        with installed(Stub()) as st:
            st.session_state["session_id"] = SID
            st.audio = Audio()
            tr = mock.patch.object(MedHelpClient, "transcribe", side_effect=ApiError("too large", 422))
            run_app(st, transcribe=tr)
        self.assertIn("too large", st.texts("error"))

    def test_new_chat_deletes_the_server_session_and_starts_fresh(self):
        with installed(Stub()) as st:
            st.session_state.update(session_id=SID, messages=[{"role": "user", "content": "q"}])
            st.clicked = {"\u2795 New chat"}
            deleter = mock.patch.object(MedHelpClient, "delete_session", return_value=None)
            self.assertEqual(run_app(st, delete=deleter), "rerun")
        self.assertNotEqual(st.session_state["session_id"], SID)
        self.assertEqual(st.session_state["messages"], [])
        self.assertEqual(st.query_params["sid"], st.session_state["session_id"])

    def test_backend_states_in_the_sidebar(self):
        cases = [({"status": "degraded"}, "warning", "index is not ready"), ({"status": "misconfigured", "missing_settings": ["GROQ_API_KEY"]}, "error", "GROQ_API_KEY"), ({"status": "unreachable"}, "error", "unreachable")]
        for health, kind, text in cases:
            with installed(Stub()) as st:
                run_app(st, health=mock.patch.object(MedHelpClient, "health", return_value=health))
            self.assertTrue(any(text in t for t in st.texts(kind)), (health, st.calls[:6]))


if __name__ == "__main__":
    unittest.main()
