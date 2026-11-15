"""Service layer: sessions across concurrent users, privacy of stored turns, limits, voice, health."""
import threading
import time
import unittest
from unittest import mock

from app import config
from app.errors import BadRequestError, IndexNotReadyError, RateLimitedError
from app.limits import RateLimiter, RequestGate
from app.service import MedHelpService, read_rss_mb
from app.sessions import SessionStore


class FakeEngine:
    lexical_status = "ready"
    _cache = None

    def __init__(self, kind="answer", delay=0.0):
        self.kind, self.delay, self.calls, self._lock = kind, delay, [], threading.Lock()

    def query(self, question, history=None, compare=None):
        time.sleep(self.delay)
        with self._lock:
            self.calls.append({"question": question, "history": [dict(t) for t in (history or [])], "compare": compare})
        return {"answer": f"answer to {question}", "kind": self.kind, "sources": [], "debug": {}}

    def query_stream(self, question, history=None, compare=None):
        yield "status", {"stage": "searching"}
        yield "token", "ans"
        yield "final", self.query(question, history, compare)

    def is_ready(self):
        return True

    def documents(self):
        return [{"title": "T", "publisher": "P"}]

    def publishers(self):
        return [{"name": "P", "documents": 1}]


def make(engine=None, **kw):
    kw.setdefault("limiter", RateLimiter(per_minute=0))
    return MedHelpService(engine=engine or FakeEngine(), sessions=kw.pop("sessions", SessionStore()), check_config=False, **kw)


class SessionFlowTests(unittest.TestCase):
    def test_five_simultaneous_users_never_see_each_others_context(self):
        eng = FakeEngine(delay=0.01)
        svc = make(eng)
        sids = [svc.new_session() for _ in range(5)]
        topics = ["hypertension", "asthma", "diabetes", "malaria", "epilepsy"]

        def user(i):
            for turn in range(3):
                svc.answer(f"{topics[i]} question {turn}", session_id=sids[i], client_key=f"ip{i}")

        threads = [threading.Thread(target=user, args=(i,)) for i in range(5)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(eng.calls), 15)
        for call in eng.calls:
            topic = call["question"].split()[0]
            for turn in call["history"]:
                self.assertIn(topic, turn["content"], "history from another session leaked into this one")
        for i, sid in enumerate(sids):
            turns = svc.get_session(sid)
            self.assertEqual(len(turns), 6)
            self.assertTrue(all(topics[i] in t["content"] for t in turns))

    def test_each_turn_sees_the_previous_ones(self):
        eng = FakeEngine()
        svc = make(eng)
        sid = svc.new_session()
        svc.answer("first question", session_id=sid)
        svc.answer("second question", session_id=sid)
        self.assertEqual(eng.calls[1]["history"], [{"role": "user", "content": "first question"}, {"role": "assistant", "content": "answer to first question"}])

    def test_unknown_session_is_recovered_from_the_clients_copy(self):
        eng = FakeEngine()
        svc = make(eng)
        sid = "R" * 24
        client_copy = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
        out = svc.answer("and for children?", client_copy, session_id=sid)
        self.assertTrue(out["session_recovered"])
        self.assertEqual(eng.calls[0]["history"], client_copy)
        self.assertEqual(len(svc.get_session(sid)), 4)
        out = svc.answer("another", [{"role": "assistant", "content": "FORGED"}], session_id=sid)
        self.assertFalse(out["session_recovered"])
        self.assertNotIn("FORGED", str(eng.calls[1]["history"]))

    def test_stateless_calls_use_the_clients_history(self):
        eng = FakeEngine()
        out = make(eng).answer("a question", [{"role": "user", "content": "earlier"}])
        self.assertIsNone(out["session_id"])
        self.assertEqual(eng.calls[0]["history"][0]["content"], "earlier")

    def test_guardrail_and_smalltalk_turns_are_not_retained(self):
        for kind in ("guardrail", "smalltalk"):
            svc = make(FakeEngine(kind=kind))
            sid = svc.new_session()
            svc.answer("I have chest pain, what should I take?", session_id=sid)
            self.assertEqual(svc.get_session(sid), [], kind)

    def test_no_answer_turns_are_retained(self):
        svc = make(FakeEngine(kind="no_answer"))
        sid = svc.new_session()
        svc.answer("something obscure", session_id=sid)
        self.assertEqual(len(svc.get_session(sid)), 2)

    def test_delete_and_validation(self):
        svc = make()
        sid = svc.new_session()
        self.assertTrue(svc.delete_session(sid))
        self.assertIsNone(svc.get_session(sid))
        for bad in ("short", "bad id with spaces!!"):
            with self.assertRaises(BadRequestError):
                svc.get_session(bad)
            with self.assertRaises(BadRequestError):
                svc.answer("question here", session_id=bad)

    def test_question_length_is_enforced(self):
        svc = make()
        with self.assertRaises(BadRequestError):
            svc.answer("x")
        with self.assertRaises(BadRequestError):
            svc.answer("y" * 1001)


class LimitTests(unittest.TestCase):
    def test_per_session_rate_limit(self):
        svc = make(limiter=RateLimiter(per_minute=2))
        sid = svc.new_session()
        svc.answer("question one", session_id=sid)
        svc.answer("question two", session_id=sid)
        with self.assertRaises(RateLimitedError):
            svc.answer("question three", session_id=sid)
        svc.answer("question one", session_id=svc.new_session())  # another user is unaffected

    def test_queue_gate_turns_overload_into_a_polite_429(self):
        svc = make(FakeEngine(delay=0.4), gate=RequestGate(max_concurrent=1, wait_seconds=0.05))
        errors = []
        t = threading.Thread(target=lambda: svc.answer("slow question", client_key="a"))
        t.start()
        time.sleep(0.1)
        try:
            svc.answer("second question", client_key="b")
        except RateLimitedError as exc:
            errors.append(exc)
        t.join()
        self.assertEqual(len(errors), 1)
        self.assertGreater(errors[0].retry_after, 0)

    def test_missing_configuration_is_reported_not_crashed(self):
        svc = MedHelpService(engine=FakeEngine(), check_config=True)
        with mock.patch.object(config, "GROQ_API_KEY", ""):
            with self.assertRaisesRegex(IndexNotReadyError, "GROQ_API_KEY"):
                svc.answer("a question")


class StreamTests(unittest.TestCase):
    def test_stream_passes_events_and_stores_the_turn_on_final(self):
        svc = make()
        sid = svc.new_session()
        events = list(svc.answer_stream("streamed question", session_id=sid))
        self.assertEqual([e for e, _ in events], ["status", "token", "final"])
        self.assertEqual(events[-1][1]["session_id"], sid)
        self.assertEqual(len(svc.get_session(sid)), 2)

    def test_stream_validates_before_returning_a_generator(self):
        svc = make(limiter=RateLimiter(per_minute=1))
        svc.answer_stream("first question here", client_key="x")
        with self.assertRaises(RateLimitedError):  # raised eagerly, so the HTTP layer can send a real 429
            svc.answer_stream("second question here", client_key="x")


class VoiceTests(unittest.TestCase):
    class T:
        def __init__(self, exc=None):
            self.exc, self.calls = exc, []

        def transcribe(self, audio, filename, content_type):
            self.calls.append((len(audio), filename, content_type))
            if self.exc:
                raise self.exc
            return "what is the blood pressure target"

    def test_transcribe_ok(self):
        t = self.T()
        out = make(transcriber=t).transcribe(b"x" * 100, "q.wav", "audio/wav")
        self.assertEqual(out["text"], "what is the blood pressure target")
        self.assertEqual(t.calls, [(100, "q.wav", "audio/wav")])

    def test_validation(self):
        svc = make(transcriber=self.T())
        with self.assertRaises(BadRequestError):
            svc.transcribe(b"")
        with mock.patch.object(config, "MAX_AUDIO_BYTES", 10):
            with self.assertRaisesRegex(BadRequestError, "too large"):
                svc.transcribe(b"x" * 11)
        with self.assertRaisesRegex(BadRequestError, "Unsupported"):
            svc.transcribe(b"x", "a.exe", "application/x-msdownload")
        svc.transcribe(b"x", "a.webm", "audio/webm;codecs=opus")  # parameters are ignored

    def test_provider_errors_are_translated(self):
        svc = make(transcriber=self.T(RuntimeError("Error code: 429 - rate limit reached, try again in 3s")))
        with self.assertRaises(RateLimitedError):
            svc.transcribe(b"x")

    def test_voice_has_its_own_rate_limit(self):
        svc = make(transcriber=self.T(), limiter=RateLimiter(per_minute=1))
        svc.transcribe(b"x", client_key="u")
        with self.assertRaises(RateLimitedError):
            svc.transcribe(b"x", client_key="u")
        svc.answer("a typed question", client_key="u")  # typing is a separate budget


class InfoTests(unittest.TestCase):
    def test_health_reports_what_matters_for_a_small_host(self):
        h = make().health()
        self.assertEqual(h["status"], "ok")
        for key in ("keyword_index", "embed_provider", "sessions", "queue", "uptime_s", "model", "memory_mb"):
            self.assertIn(key, h)
        self.assertEqual(h["keyword_index"], "ready")
        rss = read_rss_mb()
        self.assertTrue(rss is None or rss > 1)

    def test_health_when_misconfigured(self):
        svc = MedHelpService(engine=FakeEngine(), check_config=True)
        with mock.patch.object(config, "GROQ_API_KEY", ""):
            h = svc.health()
        self.assertEqual(h["status"], "misconfigured")
        self.assertIn("GROQ_API_KEY", h["missing_settings"])

    def test_documents(self):
        d = make().documents()
        self.assertEqual(d["publishers"], [{"name": "P", "documents": 1}])

    def test_keepalive_touches_qdrant_and_is_rate_limited(self):
        svc = make()
        with mock.patch("app.qdrant_store.get_qdrant_client", lambda: object()), mock.patch("app.qdrant_store.collection_point_count", lambda c: 123):
            self.assertEqual(svc.keepalive(), {"ok": True, "qdrant_points": 123})
            for _ in range(5):
                svc.keepalive()
            with self.assertRaises(RateLimitedError):
                svc.keepalive()

    def test_keepalive_reports_unreachable_qdrant(self):
        def boom():
            raise RuntimeError("down")

        with mock.patch("app.qdrant_store.get_qdrant_client", boom):
            with self.assertRaises(IndexNotReadyError):
                make().keepalive()


if __name__ == "__main__":
    unittest.main()
