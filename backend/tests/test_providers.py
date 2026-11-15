"""Cloudflare embeddings, Groq chat/stream/STT and the Voyage reranker, against a local fake server."""
import json
import unittest

import requests

from app.embeddings import BGE_QUERY_INSTRUCTION, CachedEmbedder, CloudflareEmbedder, cosine, normalize, parity_report
from app.engine import classify_llm_error
from app.errors import EmbeddingError, RateLimitedError
from app.llm import GroqLLM
from app.rerank import RerankError, VoyageReranker
from app.stt import Transcriber
from tests.helpers import ok, serve

CF_PATH = "/accounts/ACC/ai/run/@cf/baai/bge-large-en-v1.5"


def cf(base, **kw):
    return CloudflareEmbedder("ACC", "TOKEN", base_url=base, retries=kw.pop("retries", 1), **kw)


class CloudflareTests(unittest.TestCase):
    def test_query_gets_the_bge_instruction_and_vector_is_normalised(self):
        with serve({("POST", CF_PATH): lambda r: ok({"result": {"shape": [1, 3], "data": [[3.0, 4.0, 0.0]]}, "success": True})}) as (base, rec):
            vec = cf(base).embed_query("hypertension targets")
        self.assertEqual([round(x, 3) for x in vec], [0.6, 0.8, 0.0])
        req = rec.requests[0]
        self.assertEqual(req["headers"]["Authorization"], "Bearer TOKEN")
        self.assertEqual(req["json"]["text"], [BGE_QUERY_INSTRUCTION + "hypertension targets"])

    def test_rate_limit_auth_and_bad_payload_are_mapped(self):
        with serve({("POST", CF_PATH): lambda r: (429, {"Retry-After": "7"}, b"{}")}) as (base, _):
            with self.assertRaises(RateLimitedError) as ctx:
                cf(base).embed_query("q")
            self.assertEqual(ctx.exception.retry_after, 7.0)
        with serve({("POST", CF_PATH): lambda r: (403, {}, b"{}")}) as (base, _):
            with self.assertRaisesRegex(EmbeddingError, "credentials"):
                cf(base).embed_query("q")
        with serve({("POST", CF_PATH): lambda r: ok({"result": {"data": []}})}) as (base, _):
            with self.assertRaises(EmbeddingError):
                cf(base).embed_query("q")

    def test_retries_a_transient_server_error_once(self):
        good = lambda r: ok({"result": {"data": [[1.0, 0.0]]}})
        with serve({("POST", CF_PATH): [lambda r: (503, {}, b"{}"), good]}) as (base, rec):
            self.assertEqual(cf(base).embed_query("q"), [1.0, 0.0])
        self.assertEqual(len(rec.requests), 2)

    def test_unreachable_service(self):
        with self.assertRaisesRegex(EmbeddingError, "unreachable"):
            CloudflareEmbedder("A", "T", base_url="http://127.0.0.1:9", retries=0, timeout=1).embed_query("q")

    def test_cached_embedder_and_math_helpers(self):
        calls = []

        class Inner:
            name = "x"

            def embed_query(self, t):
                calls.append(t)
                return [1.0, 2.0]

        e = CachedEmbedder(Inner(), size=2)
        e.embed_query("a"), e.embed_query("a"), e.embed_query("b"), e.embed_query("c"), e.embed_query("a")
        self.assertEqual(calls, ["a", "b", "c", "a"])  # 'a' was evicted by 'c'
        self.assertAlmostEqual(cosine([1, 0], [1, 1]), 0.7071, places=3)
        self.assertEqual(normalize([0, 0]), [0, 0])

    def test_parity_report(self):
        class Same:
            def embed_query(self, t):
                return [1.0, 0.0]

        class Other:
            def embed_query(self, t):
                return [0.0, 1.0]

        self.assertTrue(parity_report(Same(), Same(), ["a", "b"])["verdict"].startswith("OK"))
        self.assertTrue(parity_report(Same(), Other(), ["a"])["verdict"].startswith("MISMATCH"))


def chat_reply(text):
    return ok({"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})


class GroqTests(unittest.TestCase):
    def test_complete_sends_deterministic_request(self):
        with serve({("POST", "/chat/completions"): lambda r: chat_reply("  hello  ")}) as (base, rec):
            llm = GroqLLM("openai/gpt-oss-120b", "KEY", max_tokens=99, reasoning_effort="low", base_url=base)
            self.assertEqual(llm.complete("prompt").text, "hello")
        body = rec.requests[0]["json"]
        self.assertEqual((body["model"], body["temperature"], body["max_tokens"], body["stream"]), ("openai/gpt-oss-120b", 0.0, 99, False))
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(rec.requests[0]["headers"]["Authorization"], "Bearer KEY")

    def test_reasoning_effort_only_for_gpt_oss(self):
        self.assertEqual(GroqLLM("llama-3.3-70b", "k", reasoning_effort="low").reasoning_effort, "")

    def test_errors_are_classified_by_the_engine(self):
        body = json.dumps({"error": {"message": "Rate limit reached ... Please try again in 7.5s", "code": "rate_limit_exceeded"}}).encode()
        with serve({("POST", "/chat/completions"): lambda r: (429, {}, body)}) as (base, _):
            with self.assertRaises(RuntimeError) as ctx:
                GroqLLM("m", "k", base_url=base).complete("p")
        err = classify_llm_error(ctx.exception)
        self.assertIsInstance(err, RateLimitedError)
        self.assertAlmostEqual(err.retry_after, 7.5)
        with serve({("POST", "/chat/completions"): lambda r: (413, {}, b'{"error": {"message": "Request too large"}}')}) as (base, _):
            with self.assertRaises(RuntimeError) as ctx:
                GroqLLM("m", "k", base_url=base).complete("p")
        self.assertIn("token limit", str(classify_llm_error(ctx.exception)))

    def test_streaming_yields_text_pieces_only(self):
        def sse(r):
            events = [
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"reasoning": "thinking..."}}]},  # reasoning is never shown
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            payload = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            return 200, {"Content-Type": "text/event-stream"}, payload.encode()

        with serve({("POST", "/chat/completions"): sse}) as (base, rec):
            pieces = list(GroqLLM("m", "k", base_url=base).stream("p"))
        self.assertEqual(pieces, ["Hel", "lo"])
        self.assertTrue(rec.requests[0]["json"]["stream"])

    def test_streaming_keeps_non_ascii_characters_intact(self):
        """Regression: a charset-less text/event-stream was decoded as Latin-1 (level\u20112 became 'levelâ\x80\x912')."""
        text = "level\u20112 hypoglycemia \u2013 patient\u2019s heart\u2011failure \u2014 \u2265 70 mg/dL \u00b5g caf\u00e9"

        def sse(r):
            events = [{"choices": [{"delta": {"content": text[:20]}}]}, {"choices": [{"delta": {"content": text[20:]}}]}]
            body = "".join("data: " + json.dumps(e, ensure_ascii=False) + "\n\n" for e in events) + "data: [DONE]\n\n"
            return 200, {"Content-Type": "text/event-stream"}, body.encode("utf-8")  # note: no charset

        with serve({("POST", "/chat/completions"): sse}) as (base, _):
            self.assertEqual("".join(GroqLLM("m", "k", base_url=base).stream("p")), text)

    def test_non_streaming_completion_keeps_non_ascii_characters_intact(self):
        text = "heart\u2011failure \u2013 patient\u2019s \u2265 70"
        with serve({("POST", "/chat/completions"): lambda r: chat_reply(text)}) as (base, _):
            self.assertEqual(GroqLLM("m", "k", base_url=base).complete("p").text, text)

    def test_stream_error_before_first_token(self):
        with serve({("POST", "/chat/completions"): lambda r: (429, {}, b"rate_limit_exceeded")}) as (base, _):
            with self.assertRaises(RuntimeError):
                list(GroqLLM("m", "k", base_url=base).stream("p"))

    def test_unreachable_groq_is_a_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "Could not reach Groq"):
            GroqLLM("m", "k", base_url="http://127.0.0.1:9", timeout=1).complete("p")


class SttTests(unittest.TestCase):
    def test_transcribe_posts_multipart_and_cleans_text(self):
        with serve({("POST", "/audio/transcriptions"): lambda r: ok({"text": "  what is the   blood pressure target "})}) as (base, rec):
            text = Transcriber("KEY", base_url=base, language="en").transcribe(b"RIFFfakewav", "q.wav", "audio/wav")
        self.assertEqual(text, "what is the blood pressure target")
        body = rec.requests[0]["body"]
        for needle in (b"whisper-large-v3-turbo", b'filename="q.wav"', b"RIFFfakewav", b'name="language"'):
            self.assertIn(needle, body)
        self.assertEqual(rec.requests[0]["headers"]["Authorization"], "Bearer KEY")

    def test_error_text_is_classifiable(self):
        with serve({("POST", "/audio/transcriptions"): lambda r: (429, {}, b"rate limit reached, try again in 3s")}) as (base, _):
            with self.assertRaises(RuntimeError) as ctx:
                Transcriber("k", base_url=base).transcribe(b"x")
        self.assertIsInstance(classify_llm_error(ctx.exception), RateLimitedError)


class RerankTests(unittest.TestCase):
    def test_returns_indexes_best_first(self):
        reply = ok({"data": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}]})
        with serve({("POST", "/rerank"): lambda r: reply}) as (base, rec):
            out = VoyageReranker("KEY", base_url=base).rerank("q", ["a", "b", "c"], top_n=2)
        self.assertEqual(out, [(2, 0.9), (0, 0.4)])
        self.assertEqual(rec.requests[0]["json"]["model"], "rerank-2.5-lite")
        self.assertEqual(rec.requests[0]["json"]["documents"], ["a", "b", "c"])

    def test_failures_raise_rerank_error(self):
        with serve({("POST", "/rerank"): lambda r: (429, {}, b"{}")}) as (base, _):
            with self.assertRaises(RerankError):
                VoyageReranker("k", base_url=base).rerank("q", ["a"])
        with self.assertRaises(RerankError):
            VoyageReranker("k", base_url="http://127.0.0.1:9", timeout=1).rerank("q", ["a"])


if __name__ == "__main__":
    unittest.main()
