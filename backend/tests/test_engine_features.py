"""New engine features: compare mode, follow-ups, cache, streaming, reranker, keyword filters, background loader."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app import config
from app.cache import SemanticCache
from app.engine import MedHelpEngine, NO_ANSWER_TEXT, parse_followups
from app.errors import BadRequestError, EmbeddingError, RateLimitedError
from app.keyword_db import KeywordDB, build_keyword_db
from tests.test_engine import FakeNode, nws

WHO_TXT = "WHO recommends initiating drug treatment in adults at a systolic blood pressure of 140 mmHg or higher (strong recommendation, moderate-certainty evidence)."
ACC_TXT = "ACC/AHA recommends drug treatment for adults at a blood pressure of 130/80 mmHg or higher (COR 1, LOE A)."

WHO_NODES = [nws(WHO_TXT + f" Detail {i}.", 0.8 - i * 0.01, document_name="WHO HTN 2021", publisher="WHO", page_label=str(9 + i), ingest_version="2") for i in range(4)]
ACC_NODES = [nws(ACC_TXT + f" Detail {i}.", 0.78 - i * 0.01, document_name="ACC HTN 2017", publisher="ACC", page_label=str(4 + i), ingest_version="2") for i in range(4)]

COMPARE_ANSWER = (
    "**Where they agree:** both recommend drug treatment in adults with raised blood pressure [WHO HTN 2021, Page 9] [ACC HTN 2017, Page 4].\n"
    "**Where they differ:** WHO uses a systolic threshold of 140 mmHg [WHO HTN 2021, Page 9], ACC/AHA uses 130/80 mmHg [ACC HTN 2017, Page 4].\n"
    "**Strength of Recommendation:** WHO strong recommendation, moderate-certainty evidence [WHO HTN 2021, Page 9]; ACC/AHA COR 1, LOE A [ACC HTN 2017, Page 4]"
)
SINGLE_ANSWER = (
    "WHO recommends starting drug treatment when systolic pressure is 140 mmHg or higher [WHO HTN 2021, Page 9].\n"
    "**Strength of Recommendation:** strong recommendation, moderate-certainty evidence [WHO HTN 2021, Page 9]"
)
FOLLOWUPS = json.dumps([
    "How does the target differ for older adults?",
    "What does the guideline say about children?",
    "What is hypertension and why does it matter for the heart?",  # fine
    "I have chest pain, what should I take?",
])


class FakeEmbedder:
    def __init__(self, vectors=None, exc=None):
        self.vectors, self.exc, self.calls = vectors or {}, exc, []

    def embed_query(self, text):
        self.calls.append(text)
        if self.exc:
            raise self.exc
        return self.vectors.get(text, [1.0, 0.0, 0.0])


class VecRetriever:
    """Retriever with a vector entry point and optional per-publisher results."""

    def __init__(self, nodes=None, by_publisher=None):
        self.nodes, self.by_publisher = nodes or [], by_publisher
        self.vectors, self.publishers, self.questions = [], [], []

    def retrieve_vector(self, vector, publisher=None):
        self.vectors.append(vector)
        self.publishers.append(publisher)
        return list(self.by_publisher.get(publisher, [])) if self.by_publisher is not None and publisher else list(self.nodes)

    def retrieve(self, question, publisher=None):
        self.questions.append(question)
        return self.retrieve_vector([0.0], publisher)


class StreamLLM:
    def __init__(self, reply="", pieces=None, exc=None, stream_exc=None):
        self.reply, self.pieces, self.exc, self.stream_exc = reply, pieces, exc, stream_exc
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if self.exc:
            raise self.exc
        return SimpleNamespace(text=self.reply)

    def stream(self, prompt):
        self.prompts.append(prompt)
        if self.stream_exc:
            raise self.stream_exc
        yield from (self.pieces if self.pieces is not None else [self.reply])


class FastLLM:
    """Routes by prompt: condense / jargon / follow-ups."""

    def __init__(self, followups=FOLLOWUPS, jargon='{"systolic": "The top number."}', condense="What are the blood pressure thresholds for children with hypertension?", condense_exc=None):
        self.followups, self.jargon, self.condense, self.condense_exc = followups, jargon, condense, condense_exc
        self.kinds = []

    def complete(self, prompt):
        if "Suggest exactly 3" in prompt:
            self.kinds.append("followups")
            return SimpleNamespace(text=self.followups)
        if "Identify any highly technical" in prompt:
            self.kinds.append("jargon")
            return SimpleNamespace(text=self.jargon)
        self.kinds.append("condense")
        if self.condense_exc:
            raise self.condense_exc
        return SimpleNamespace(text=self.condense)


def engine_for(retriever, llm, fast=None, **kw):
    kw.setdefault("min_similarity", 0.25)
    eng = MedHelpEngine(retriever=retriever, llm=llm, fast_llm=fast or FastLLM(), **kw)
    eng.publishers = lambda: [{"name": "WHO", "documents": 2}, {"name": "ACC", "documents": 1}, {"name": "GINA", "documents": 1}]
    return eng


class FollowupTests(unittest.TestCase):
    def test_parse_followups_filters_bad_suggestions(self):
        out = parse_followups(FOLLOWUPS, exclude="What does the guideline say about children?")
        self.assertEqual(out, ["How does the target differ for older adults?", "What is hypertension and why does it matter for the heart?"])

    def test_parse_followups_accepts_fences_dicts_and_lines(self):
        self.assertEqual(parse_followups('```json\n["What is the target for older adults?"]\n```'), ["What is the target for older adults?"])
        self.assertEqual(parse_followups('{"questions": ["Which drugs are first-line agents?"]}'), ["Which drugs are first-line agents?"])
        self.assertEqual(parse_followups("1. When should treatment start?\n2. Is salt reduction advised?"), ["When should treatment start?", "Is salt reduction advised?"])
        self.assertEqual(parse_followups("not json at all"), [])
        self.assertEqual(parse_followups('{"systolic": "The top number."}'), [])  # a jargon-shaped reply never crashes

    def test_followups_are_returned_with_the_answer(self):
        fast = FastLLM()
        out = engine_for(VecRetriever(WHO_NODES), StreamLLM(SINGLE_ANSWER), fast).query("What does WHO recommend for hypertension?")
        self.assertEqual(out["kind"], "answer")
        self.assertEqual(len(out["suggested_followups"]), 3)
        self.assertIn("jargon", fast.kinds)
        self.assertTrue(all("chest pain" not in q for q in out["suggested_followups"]))
        self.assertEqual(out["jargon_dictionary"], {"systolic": "The top number."})

    def test_followups_can_be_switched_off(self):
        fast = FastLLM()
        out = engine_for(VecRetriever(WHO_NODES), StreamLLM(SINGLE_ANSWER), fast, followups=False).query("What does WHO recommend for hypertension?")
        self.assertEqual(out["suggested_followups"], [])
        self.assertNotIn("followups", fast.kinds)

    def test_followup_failure_never_breaks_the_answer(self):
        class Boom(FastLLM):
            def complete(self, prompt):
                if "Suggest exactly 3" in prompt:
                    raise RuntimeError("Error code: 429 - rate limit, try again in 2s")
                return super().complete(prompt)

        out = engine_for(VecRetriever(WHO_NODES), StreamLLM(SINGLE_ANSWER), Boom()).query("What does WHO recommend for hypertension?")
        self.assertEqual((out["kind"], out["suggested_followups"]), ("answer", []))


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.retriever = VecRetriever(by_publisher={"WHO": WHO_NODES, "ACC": ACC_NODES})
        self.llm = StreamLLM(COMPARE_ANSWER)
        self.eng = engine_for(self.retriever, self.llm, embedder=FakeEmbedder())

    def test_compare_retrieves_per_publisher_and_splits_the_existing_top_k(self):
        out = self.eng.query("blood pressure threshold for starting drugs", compare=["WHO", "ACC"])
        self.assertEqual(self.retriever.publishers, ["WHO", "ACC"])
        self.assertEqual(out["mode"], "compare")
        self.assertEqual(out["kind"], "answer")
        pubs = [s["publisher"] for s in out["sources"]]
        self.assertEqual((pubs.count("WHO"), pubs.count("ACC")), (3, 3))  # TOP_K=6 split between two publishers
        prompt = self.llm.prompts[0]
        self.assertIn("=== GROUP 1: WHO ===", prompt)
        self.assertIn("=== GROUP 2: ACC ===", prompt)
        self.assertLess(prompt.index("GROUP 1: WHO"), prompt.index("[WHO HTN 2021, Page 9]"))
        self.assertEqual(out["debug"]["per_publisher"]["WHO"]["passages_used"], 3)
        self.assertEqual({v["status"] for v in out["verification"]}, {"supported"})

    def test_three_publishers_share_the_budget(self):
        gina = [nws("GINA says inhaled corticosteroids are first-line controllers for asthma. " + str(i), 0.7, document_name="GINA 2024", publisher="GINA", page_label=str(i + 1), ingest_version="2") for i in range(4)]
        self.retriever.by_publisher["GINA"] = gina
        out = self.eng.query("first-line treatment", compare=["WHO", "ACC", "GINA"])
        self.assertEqual(len(out["sources"]), 6)  # ceil(6/3)=2 each
        self.assertEqual(out["debug"]["passages_used"], 6)

    def test_bad_compare_requests(self):
        for bad in (["WHO"], ["WHO", "ACC", "GINA", "CDC"], ["WHO", "Nobody"]):
            with self.assertRaises(BadRequestError):
                self.eng.query("blood pressure", compare=bad)
        # duplicates collapse, then fail the "at least two" rule
        with self.assertRaises(BadRequestError):
            self.eng.query("blood pressure", compare=["WHO", "WHO"])

    def test_publisher_without_passages_gives_a_clear_no_answer(self):
        self.retriever.by_publisher["ACC"] = []
        out = self.eng.query("blood pressure threshold", compare=["WHO", "ACC"])
        self.assertEqual(out["kind"], "no_answer")
        self.assertEqual(out["answer"], NO_ANSWER_TEXT)
        self.assertTrue(any("ACC" in w for w in out["warnings"]))
        self.assertEqual(self.llm.prompts, [])  # no LLM call was wasted

    def test_compare_answers_never_mix_with_single_answers_in_the_cache(self):
        cache = SemanticCache()
        eng = engine_for(self.retriever, self.llm, embedder=FakeEmbedder(), cache=cache)
        eng.query("blood pressure threshold", compare=["WHO", "ACC"])
        self.llm.reply = SINGLE_ANSWER
        out = eng.query("blood pressure threshold")  # same words, different scope
        self.assertFalse(out["cached"])
        self.assertEqual(out["mode"], "single")


class CacheTests(unittest.TestCase):
    def make(self, **kw):
        llm = StreamLLM(SINGLE_ANSWER)
        eng = engine_for(VecRetriever(WHO_NODES), llm, embedder=kw.pop("embedder", FakeEmbedder()), cache=SemanticCache(), **kw)
        return eng, llm

    def test_repeat_question_is_served_from_cache_without_an_llm_call(self):
        eng, llm = self.make()
        first = eng.query("What does WHO recommend for hypertension?")
        second = eng.query("what does who recommend for hypertension")
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(len(llm.prompts), 1)
        self.assertEqual(second["answer"], first["answer"])
        self.assertEqual(second["sources"], first["sources"])

    def test_near_duplicate_via_embedding_similarity(self):
        emb = FakeEmbedder({"q-a": [1.0, 0.0, 0.0], "q-b": [0.999, 0.02, 0.0], "q-c": [0.0, 1.0, 0.0]})
        eng, llm = self.make(embedder=emb)
        eng.query("q-a")
        self.assertTrue(eng.query("q-b")["cached"])
        self.assertFalse(eng.query("q-c")["cached"])

    def test_refusals_and_guardrails_are_not_cached(self):
        llm = StreamLLM("I do not know based on the provided guidelines.")
        cache = SemanticCache()
        eng = engine_for(VecRetriever(WHO_NODES), llm, embedder=FakeEmbedder(), cache=cache)
        eng.query("What does WHO say about moon landings?")
        eng.query("What does WHO say about moon landings?")
        self.assertEqual(len(llm.prompts), 2)
        self.assertEqual(cache.stats()["entries"], 0)

    def test_ambiguous_followup_whose_rewrite_failed_is_never_cached_or_served(self):
        fast = FastLLM(condense_exc=RuntimeError("Error code: 429 - rate limit"))
        eng, llm = self.make()
        eng._fast_llm = fast
        history = [{"role": "user", "content": "hypertension in adults?"}, {"role": "assistant", "content": "WHO says ..."}]
        eng.query("what about children?", history)
        out = eng.query("what about children?", [{"role": "user", "content": "asthma inhalers?"}, {"role": "assistant", "content": "GINA says ..."}])
        self.assertFalse(out["cached"])
        self.assertEqual(len(llm.prompts), 2)

    def test_rewritten_followup_is_cached_under_the_standalone_question(self):
        eng, llm = self.make()
        history = [{"role": "user", "content": "hypertension in adults?"}, {"role": "assistant", "content": "WHO says ..."}]
        eng.query("what about children?", history)
        out = eng.query("What are the blood pressure thresholds for children with hypertension?")
        self.assertTrue(out["cached"])

    def test_embedder_is_used_once_and_vector_reaches_the_retriever(self):
        emb = FakeEmbedder()
        retriever = VecRetriever(WHO_NODES)
        engine_for(retriever, StreamLLM(SINGLE_ANSWER), embedder=emb).query("What does WHO recommend for hypertension?")
        self.assertEqual(len(emb.calls), 1)
        self.assertEqual(retriever.vectors, [[1.0, 0.0, 0.0]])
        self.assertEqual(retriever.questions, [])  # the plain-text entry point was not used

    def test_embedding_failure_surfaces_as_an_error(self):
        eng = engine_for(VecRetriever(WHO_NODES), StreamLLM(SINGLE_ANSWER), embedder=FakeEmbedder(exc=EmbeddingError("down")))
        with self.assertRaises(EmbeddingError):
            eng.query("What does WHO recommend for hypertension?")
        eng = engine_for(VecRetriever(WHO_NODES), StreamLLM(SINGLE_ANSWER), embedder=FakeEmbedder(exc=ValueError("weird")))
        with self.assertRaises(EmbeddingError):
            eng.query("What does WHO recommend for hypertension?")


class StreamingTests(unittest.TestCase):
    def test_events_in_order_and_final_matches_the_streamed_text(self):
        pieces = ["WHO recommends ", "starting drug treatment when systolic pressure is 140 mmHg or higher [WHO HTN 2021, Page 9].\n", "**Strength of Recommendation:** strong recommendation, moderate-certainty evidence [WHO HTN 2021, Page 9]"]
        eng = engine_for(VecRetriever(WHO_NODES), StreamLLM(pieces=pieces))
        events = list(eng.query_stream("What does WHO recommend for hypertension?"))
        names = [e for e, _ in events]
        self.assertEqual(names[0], "status")
        self.assertEqual(names.count("token"), 3)
        self.assertEqual(names[-1], "final")
        self.assertEqual("".join(d for e, d in events if e == "token"), events[-1][1]["answer"])
        self.assertEqual(events[-1][1]["kind"], "answer")
        self.assertEqual([d["stage"] for e, d in events if e == "status"], ["searching", "generating", "checking"])

    def test_early_results_skip_token_events(self):
        eng = engine_for(VecRetriever(WHO_NODES), StreamLLM(SINGLE_ANSWER))
        events = list(eng.query_stream("hello"))
        self.assertEqual([e for e, _ in events], ["status", "final"])
        self.assertEqual(events[-1][1]["kind"], "smalltalk")

    def test_llm_without_stream_falls_back_to_one_piece(self):
        class NoStream:
            def complete(self, prompt):
                return SimpleNamespace(text=SINGLE_ANSWER)

        events = list(engine_for(VecRetriever(WHO_NODES), NoStream()).query_stream("What does WHO recommend for hypertension?"))
        self.assertEqual([e for e, _ in events].count("token"), 1)

    def test_errors_surface_from_the_generator(self):
        llm = StreamLLM(stream_exc=RuntimeError("Error code: 429 - rate_limit_exceeded, try again in 4s"))
        eng = engine_for(VecRetriever(WHO_NODES), llm)
        with self.assertRaises(RateLimitedError):
            list(eng.query_stream("What does WHO recommend for hypertension?"))

    def test_streamed_refusal_becomes_no_answer(self):
        eng = engine_for(VecRetriever(WHO_NODES), StreamLLM(pieces=["I do not know ", "based on the provided guidelines."]))
        final = list(eng.query_stream("What does WHO say about moon landings?"))[-1][1]
        self.assertEqual(final["kind"], "no_answer")


class RerankTests(unittest.TestCase):
    class Reverse:
        def __init__(self, exc=None):
            self.calls, self.exc = [], exc

        def rerank(self, query, texts, top_n=None):
            self.calls.append(texts)
            if self.exc:
                raise self.exc
            return [(i, 1.0 - i / 10) for i in reversed(range(len(texts)))]

    def nodes(self):
        return [nws(f"hypertension passage number {i} about treatment thresholds", 0.8 - i * 0.05, document_name="D", publisher="WHO", page_label=str(i + 1), ingest_version="2") for i in range(5)]

    def test_reranker_reorders_but_never_changes_how_many_passages_are_used(self):
        rr = self.Reverse()
        eng = engine_for(VecRetriever(self.nodes()), StreamLLM(SINGLE_ANSWER), reranker=rr, top_k=3)
        out = eng.query("hypertension treatment thresholds")
        self.assertEqual(len(out["sources"]), 3)
        self.assertEqual(out["debug"]["reranked"], True)
        self.assertEqual([s["page_number"] for s in out["sources"]], ["5", "4", "3"])

    def test_reranker_failure_keeps_the_fused_order(self):
        eng = engine_for(VecRetriever(self.nodes()), StreamLLM(SINGLE_ANSWER), reranker=self.Reverse(RuntimeError("quota")), top_k=3)
        out = eng.query("hypertension treatment thresholds")
        self.assertEqual([s["page_number"] for s in out["sources"]], ["1", "2", "3"])
        self.assertEqual(out["debug"]["reranked"], False)
        self.assertIn("quota", out["debug"]["rerank_error"])

    def test_reranker_is_not_called_in_compare_mode(self):
        rr = self.Reverse()
        eng = engine_for(VecRetriever(by_publisher={"WHO": WHO_NODES, "ACC": ACC_NODES}), StreamLLM(COMPARE_ANSWER), reranker=rr)
        eng.query("blood pressure threshold", compare=["WHO", "ACC"])
        self.assertEqual(rr.calls, [])


class KeywordFilterTests(unittest.TestCase):
    def test_publisher_filter_is_applied_to_plain_lexical_indexes_too(self):
        from app.lexical import LexicalIndex

        idx = LexicalIndex.from_records([
            ("a", "hypertension drug treatment threshold WHO text", {"document_name": "W", "publisher": "WHO"}),
            ("b", "hypertension drug treatment threshold ACC text", {"document_name": "A", "publisher": "ACC"}),
        ])
        eng = engine_for(VecRetriever(WHO_NODES), StreamLLM(SINGLE_ANSWER), lexical=idx)
        hits = eng._keyword_search("hypertension drug treatment threshold", "ACC")
        self.assertEqual([h.id for h in hits], ["b"])
        self.assertEqual({h.id for h in eng._keyword_search("hypertension drug treatment threshold")}, {"a", "b"})

    def test_sqlite_keyword_db_filters_in_sql(self):
        tmp = Path(tempfile.mkdtemp()) / "k.sqlite"
        build_keyword_db([("a", "hypertension drug treatment threshold WHO text", {"publisher": "WHO", "document_name": "W"}),
                          ("b", "hypertension drug treatment threshold ACC text", {"publisher": "ACC", "document_name": "A"})], tmp, 2)
        eng = engine_for(VecRetriever(WHO_NODES), StreamLLM(SINGLE_ANSWER), lexical=KeywordDB(tmp))
        self.assertEqual([h.id for h in eng._keyword_search("hypertension drug treatment threshold", "WHO")], ["a"])


class LexicalLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.records = [("a", "hypertension drug treatment threshold text", {"publisher": "WHO", "document_name": "W"})]
        self.eng = MedHelpEngine()  # nothing injected: the loader path is what we test

    def run_loader(self, db_path, points, build=None):
        patches = [
            mock.patch.object(config, "KEYWORD_DB_PATH", db_path),
            mock.patch("app.qdrant_store.get_qdrant_client", lambda: object()),
            mock.patch("app.qdrant_store.collection_point_count", lambda c, name=None: points),
        ]
        if build:
            patches.append(mock.patch("app.keyword_db.build_from_qdrant", build))
        for p in patches:
            p.start()
        try:
            self.eng._load_lexical()
        finally:
            mock.patch.stopall()

    def test_current_database_is_used_as_is(self):
        path = self.tmp / "k.sqlite.gz"
        build_keyword_db(self.records, path, qdrant_points=5)
        self.run_loader(path, 5)
        self.assertEqual(self.eng.lexical_status, "ready")
        self.assertEqual(len(self.eng._lexical), 1)

    def test_stale_database_is_used_then_replaced(self):
        path = self.tmp / "k.sqlite.gz"
        build_keyword_db(self.records, path, qdrant_points=5)

        def rebuild(client, collection, out):
            build_keyword_db(self.records + [("b", "asthma controller inhaled corticosteroid text", {"publisher": "GINA", "document_name": "G"})], out, 6)
            return {}

        self.run_loader(path, 6, build=rebuild)
        self.assertEqual(self.eng.lexical_status, "ready")
        self.assertEqual(len(self.eng._lexical), 2)

    def test_missing_database_is_built_from_qdrant(self):
        def build(client, collection, out):
            build_keyword_db(self.records, out, 1)
            return {}

        self.run_loader(self.tmp / "nope.sqlite.gz", 1, build=build)
        self.assertEqual(self.eng.lexical_status, "ready")

    def test_failure_degrades_to_semantic_only(self):
        def boom(client, collection, out):
            raise RuntimeError("qdrant down")

        self.run_loader(self.tmp / "nope.sqlite.gz", 1, build=boom)
        self.assertEqual(self.eng.lexical_status, "unavailable")
        self.assertIsNone(self.eng._lexical)


if __name__ == "__main__":
    unittest.main()
