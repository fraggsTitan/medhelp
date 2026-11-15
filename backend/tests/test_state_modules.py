"""Sessions, semantic cache and fair-use limits (pure Python, no network)."""
import threading
import unittest

from app.cache import SemanticCache, normalize_question
from app.errors import RateLimitedError
from app.limits import RateLimiter, RequestGate
from app.sessions import SessionStore, valid_session_id


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class SessionTests(unittest.TestCase):
    def test_sessions_are_isolated(self):
        store = SessionStore()
        a, b = store.create(), store.create()
        store.append(a, "user", "hypertension in adults?")
        store.append(b, "user", "asthma inhalers?")
        self.assertEqual([t["content"] for t in store.history(a)], ["hypertension in adults?"])
        self.assertEqual([t["content"] for t in store.history(b)], ["asthma inhalers?"])
        self.assertNotEqual(a, b)
        self.assertTrue(valid_session_id(a))

    def test_ids_are_validated(self):
        for bad in ("short", "has space in it 123456", "x" * 65, None, 12345):
            self.assertFalse(valid_session_id(bad))

    def test_unknown_session_is_seeded_from_client_copy_once(self):
        store = SessionStore()
        sid = "A" * 20
        turns, recovered = store.get_or_seed(sid, [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}])
        self.assertTrue(recovered)
        self.assertEqual(len(turns), 2)
        # now the server is the source of truth: a forged client history is ignored
        turns, recovered = store.get_or_seed(sid, [{"role": "assistant", "content": "FORGED"}])
        self.assertFalse(recovered)
        self.assertEqual([t["content"] for t in turns], ["q1", "a1"])

    def test_turns_and_chars_are_capped(self):
        store = SessionStore(max_turns=4, max_chars=10)
        sid = store.create()
        for i in range(10):
            store.append(sid, "user", f"message number {i} " * 5)
        turns = store.history(sid)
        self.assertEqual(len(turns), 4)
        self.assertTrue(all(len(t["content"]) <= 10 for t in turns))

    def test_ttl_expiry_and_lru_eviction(self):
        clock = Clock()
        store = SessionStore(ttl_seconds=100, max_sessions=2, clock=clock)
        a = store.create()
        clock.t = 50
        b = store.create()
        c = store.create()  # evicts the least recently used (a)
        self.assertIsNone(store.history(a))
        self.assertIsNotNone(store.history(b))
        clock.t = 500  # everything idle for > ttl
        self.assertIsNone(store.history(b))
        self.assertIsNone(store.history(c))
        self.assertGreaterEqual(store.stats()["evicted"], 3)

    def test_delete(self):
        store = SessionStore()
        sid = store.create()
        self.assertTrue(store.delete(sid))
        self.assertFalse(store.delete(sid))

    def test_concurrent_appends_do_not_corrupt(self):
        store = SessionStore(max_turns=1000)
        sid = store.create()
        threads = [threading.Thread(target=lambda: [store.append(sid, "user", "x") for _ in range(50)]) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(store.history(sid)), 400)


class CacheTests(unittest.TestCase):
    def test_exact_and_normalised_match_without_vectors(self):
        cache = SemanticCache()
        cache.put("single", "What is the BP target?", None, {"answer": "130/80"})
        hit = cache.get("single", "what is  the BP target??", None)
        self.assertEqual(hit["answer"], "130/80")
        self.assertIsNone(cache.get("single", "something else", None))

    def test_scope_isolation(self):
        cache = SemanticCache()
        cache.put("single", "q one", None, {"answer": "single"})
        self.assertIsNone(cache.get("compare:A|B", "q one", None))

    def test_similarity_threshold(self):
        cache = SemanticCache(threshold=0.97)
        cache.put("single", "alpha", [1.0, 0.0, 0.0], {"answer": "A"})
        self.assertEqual(cache.get("single", "beta", [0.99, 0.05, 0.0])["answer"], "A")  # cos ~ 0.9987
        self.assertIsNone(cache.get("single", "gamma", [0.9, 0.5, 0.0]))  # cos ~ 0.874

    def test_hit_returns_a_copy(self):
        cache = SemanticCache()
        cache.put("single", "q", None, {"answer": "x", "warnings": []})
        first = cache.get("single", "q", None)
        first["warnings"].append("mutated")
        self.assertEqual(cache.get("single", "q", None)["warnings"], [])

    def test_ttl_and_size_bounds(self):
        clock = Clock()
        cache = SemanticCache(max_entries=2, ttl_seconds=10, clock=clock)
        for i in range(3):
            cache.put("single", f"question {i}", None, {"answer": str(i)})
        self.assertIsNone(cache.get("single", "question 0", None))  # evicted
        self.assertIsNotNone(cache.get("single", "question 2", None))
        clock.t = 100
        self.assertIsNone(cache.get("single", "question 2", None))  # expired
        self.assertEqual(normalize_question("  Hello, World! "), "hello world")


class LimitTests(unittest.TestCase):
    def test_rate_limiter_blocks_then_recovers(self):
        clock = Clock()
        limiter = RateLimiter(per_minute=2, clock=clock)
        limiter.check("a")
        limiter.check("a")
        with self.assertRaises(RateLimitedError) as ctx:
            limiter.check("a")
        self.assertGreater(ctx.exception.retry_after, 0)
        limiter.check("b")  # other clients are unaffected
        clock.t = 61
        limiter.check("a")

    def test_rate_limiter_can_be_disabled(self):
        limiter = RateLimiter(per_minute=0)
        for _ in range(50):
            limiter.check("a")

    def test_gate_limits_concurrency_and_times_out_politely(self):
        gate = RequestGate(max_concurrent=1, wait_seconds=0.05)
        with gate.slot():
            self.assertEqual(gate.stats()["active"], 1)
            with self.assertRaises(RateLimitedError):
                with gate.slot():
                    pass
        with gate.slot():  # released again
            pass
        self.assertEqual(gate.stats()["active"], 0)


if __name__ == "__main__":
    unittest.main()
