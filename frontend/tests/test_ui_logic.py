"""Helpers, API client (against a local fake server) and export; no Streamlit needed."""
import json
import unittest

from medhelp_ui.client import ApiError, MedHelpClient, parse_sse
from medhelp_ui.export import conversation_blocks, conversation_markdown, soften_long_tokens, to_latin1
from medhelp_ui.helpers import (
    escape_md, friendly_error, history_for_api, messages_from_turns, new_session_id, number_citations,
    split_answer, valid_session_id, verification_summary,
)
from tests.fake_server import ok, serve

SOURCES = [{"document_name": "WHO HTN 2021", "source_url": "http://who/htn"}]
ANSWER = "Start drug treatment at 140 mmHg [WHO HTN 2021, Page 9] and monitor [WHO HTN 2021, Page 9] [Other, Page 2].\n**Strength of Recommendation:** strong [WHO HTN 2021, Page 9]"


class HelperTests(unittest.TestCase):
    def test_numbering_is_shared_between_body_and_grade_line(self):
        numbered, refs = number_citations(ANSWER, SOURCES)
        body, grade = split_answer(numbered)
        self.assertIn("**[1]**", body)
        self.assertIn("**[1]**", grade)  # same passage => same number
        self.assertEqual([(r["n"], r["document"], r["page"]) for r in refs], [(1, "WHO HTN 2021", "9"), (2, "Other", "2")])
        self.assertEqual(refs[0]["url"], "http://who/htn")

    def test_misc_helpers(self):
        self.assertEqual(escape_md("$5 and $6"), r"\$5 and \$6")
        self.assertEqual(split_answer("no grade here"), ("no grade here", None))
        sid = new_session_id()
        self.assertTrue(valid_session_id(sid))
        self.assertFalse(valid_session_id("short"))
        self.assertIn("rate-limited", friendly_error(429, "wait"))
        self.assertIn("API key", friendly_error(401, "x"))

    def test_history_skips_errors_and_private_turns(self):
        msgs = [
            {"role": "user", "content": "I have chest pain"},
            {"role": "assistant", "content": "refusal", "result": {"kind": "guardrail"}},
            {"role": "user", "content": "hypertension targets?"},
            {"role": "assistant", "content": "boom", "error": True},
            {"role": "assistant", "content": "answer", "result": {"kind": "answer"}},
        ]
        self.assertEqual([t["content"] for t in history_for_api(msgs)], ["I have chest pain", "hypertension targets?", "answer"])

    def test_restored_messages_and_summary(self):
        out = messages_from_turns([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, {"role": "system", "content": "x"}])
        self.assertEqual([m["role"] for m in out], ["user", "assistant"])
        self.assertTrue(all(m["restored"] for m in out))
        self.assertIn("1/2 supported, 1 to verify", verification_summary([{"status": "supported"}, {"status": "unsupported"}]))


class SseParserTests(unittest.TestCase):
    def test_parses_events_comments_and_multiline_data(self):
        lines = [": ping", "event: token", 'data: "Hel"', "", "event: token", 'data: "lo"', "", "event: final", 'data: {"a": 1}', "", ""]
        self.assertEqual(list(parse_sse(lines)), [("token", "Hel"), ("token", "lo"), ("final", {"a": 1})])

    def test_non_json_data_and_missing_trailing_blank_line(self):
        self.assertEqual(list(parse_sse(["data: plain text"])), [("message", "plain text")])


def sse_body(events):
    return "".join(f"event: {e}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n" for e, d in events).encode("utf-8")


class ClientTests(unittest.TestCase):
    def test_query_sends_session_compare_and_api_key(self):
        with serve({("POST", "/query"): lambda r: ok({"answer": "hi"})}) as (base, rec):
            out = MedHelpClient(base, "SECRET").query("q?", [{"role": "user", "content": "x"}], "S" * 20, ["WHO", "ACC"])
        self.assertEqual(out["answer"], "hi")
        req = rec.requests[0]
        self.assertEqual(req["headers"]["X-API-Key"], "SECRET")
        self.assertEqual(req["json"], {"question": "q?", "history": [{"role": "user", "content": "x"}], "session_id": "S" * 20, "compare": ["WHO", "ACC"]})

    def test_errors_are_translated(self):
        cases = [(429, {"Retry-After": "8"}, "rate-limited"), (503, {}, "isn't ready"), (401, {}, "API key"), (422, {}, "bad input")]
        for status, headers, expect in cases:
            with serve({("POST", "/query"): lambda r, s=status, h=headers: (s, h, json.dumps({"detail": "bad input"}).encode())}) as (base, _):
                with self.assertRaises(ApiError) as ctx:
                    MedHelpClient(base).query("q?", [])
            self.assertEqual(ctx.exception.status, status)
            self.assertIn(expect, str(ctx.exception))
        self.assertEqual(ctx.exception.retry_after, None)

    def test_validation_detail_lists_are_flattened(self):
        body = json.dumps({"detail": [{"msg": "too short"}, {"msg": "bad id"}]}).encode()
        with serve({("POST", "/query"): lambda r: (422, {}, body)}) as (base, _):
            with self.assertRaisesRegex(ApiError, "too short; bad id"):
                MedHelpClient(base).query("q?", [])

    def test_unreachable_backend(self):
        with self.assertRaisesRegex(ApiError, "Cannot reach"):
            MedHelpClient("http://127.0.0.1:9", timeout=1).query("q?", [])
        self.assertEqual(MedHelpClient("http://127.0.0.1:9", timeout=1).health()["status"], "unreachable")

    def test_stream_yields_events_and_keeps_utf8(self):
        events = [("status", {"stage": "searching"}), ("token", "\u2265140 mmHg "), ("token", "caf\u00e9"), ("final", {"answer": "\u2265140 mmHg caf\u00e9"})]
        route = lambda r: (200, {"Content-Type": "text/event-stream"}, sse_body(events))
        with serve({("POST", "/query/stream"): route}) as (base, _):
            got = list(MedHelpClient(base).stream_query("q?", []))
        self.assertEqual([e for e, _ in got], ["status", "token", "token", "final"])
        self.assertEqual("".join(d for e, d in got if e == "token"), "\u2265140 mmHg caf\u00e9")

    def test_stream_error_event_becomes_api_error(self):
        events = [("status", {"stage": "searching"}), ("error", {"status": 429, "detail": "busy", "retry_after": 4})]
        with serve({("POST", "/query/stream"): lambda r: (200, {"Content-Type": "text/event-stream"}, sse_body(events))}) as (base, _):
            gen = MedHelpClient(base).stream_query("q?", [])
            self.assertEqual(next(gen)[0], "status")
            with self.assertRaises(ApiError) as ctx:
                next(gen)
        self.assertEqual(ctx.exception.status, 429)
        self.assertIn("retry in about 5s", str(ctx.exception))

    def test_stream_http_error_before_streaming(self):
        with serve({("POST", "/query/stream"): lambda r: (401, {}, b'{"detail": "no"}')}) as (base, _):
            with self.assertRaises(ApiError) as ctx:
                list(MedHelpClient(base).stream_query("q?", []))
        self.assertEqual(ctx.exception.status, 401)

    def test_sessions_documents_and_health(self):
        routes = {
            ("GET", "/sessions/" + "A" * 20): lambda r: ok({"turns": [{"role": "user", "content": "q"}]}),
            ("GET", "/sessions/" + "B" * 20): lambda r: (404, {}, b'{"detail": "x"}'),
            ("DELETE", "/sessions/" + "A" * 20): lambda r: ok({"deleted": True}),
            ("GET", "/documents"): lambda r: ok({"documents": [{"title": "T"}], "publishers": [{"name": "P", "documents": 1}]}),
            ("GET", "/health"): lambda r: ok({"status": "ok"}),
        }
        with serve(routes) as (base, rec):
            c = MedHelpClient(base)
            self.assertEqual(c.get_session("A" * 20), [{"role": "user", "content": "q"}])
            self.assertIsNone(c.get_session("B" * 20))
            c.delete_session("A" * 20)
            self.assertEqual(c.documents()["publishers"][0]["name"], "P")
            self.assertEqual(c.health()["status"], "ok")

    def test_transcribe_uploads_audio(self):
        with serve({("POST", "/transcribe"): lambda r: ok({"text": "blood pressure target"})}) as (base, rec):
            text = MedHelpClient(base, "K").transcribe(b"RIFFdata", "q.wav", "audio/wav", "S" * 20)
        self.assertEqual(text, "blood pressure target")
        self.assertIn(b"RIFFdata", rec.requests[0]["body"])
        self.assertIn(b'name="session_id"', rec.requests[0]["body"])
        with serve({("POST", "/transcribe"): lambda r: (429, {}, b'{"detail": "slow down"}')}) as (base, _):
            with self.assertRaises(ApiError):
                MedHelpClient(base).transcribe(b"x")


class ExportTests(unittest.TestCase):
    RESULT = {"kind": "answer", "mode": "compare", "sources": [{"document_name": "WHO HTN 2021", "publisher": "WHO", "page_number": "9", "source_url": "http://who/htn"}],
              "warnings": ["Check the numbers."], "verification": [{"status": "supported"}, {"status": "unsupported"}]}
    MSGS = [
        {"role": "user", "content": "What does WHO recommend?"},
        {"role": "assistant", "content": ANSWER, "result": RESULT},
        {"role": "assistant", "content": "network error", "error": True},
    ]

    def test_markdown_has_numbered_citations_references_and_notes(self):
        md = conversation_markdown(self.MSGS, "2026-10-26")
        self.assertIn("## You\n\nWhat does WHO recommend?", md)
        self.assertIn("monitor [1] [2]", md)
        self.assertIn("> Strength of Recommendation: strong [1]", md)
        self.assertIn("- [1] WHO HTN 2021, page 9 - http://who/htn", md)
        self.assertIn("Comparison of: WHO", md)
        self.assertIn("1 of 2 cited statements matched", md)
        self.assertIn("Check the numbers.", md)
        self.assertNotIn("network error", md)
        self.assertIn("not medical advice", md)

    def test_blocks_skip_errors_and_keep_plain_text_for_refusals(self):
        blocks = conversation_blocks(self.MSGS + [{"role": "assistant", "content": "I do not know based on the provided guidelines.", "result": {"kind": "no_answer", "sources": []}}])
        self.assertEqual([b["role"] for b in blocks], ["user", "assistant", "assistant"])
        self.assertEqual(blocks[-1]["refs"], [])

    def test_pdf_text_helpers(self):
        self.assertEqual(to_latin1("BP \u2265 140 \u2013 150 mmHg \u2705 \u4e2d"), "BP >= 140 - 150 mmHg  ?")
        self.assertTrue(all(len(w) <= 10 for w in soften_long_tokens("x" * 35 + " ok", 10).split(" ")))

    def test_pdf_export_if_fpdf2_is_installed(self):
        try:
            import fpdf  # noqa: F401
        except ImportError:
            self.skipTest("fpdf2 not installed")
        from medhelp_ui.export import conversation_pdf

        pdf = conversation_pdf(self.MSGS)
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
