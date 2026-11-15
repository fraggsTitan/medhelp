"""Tests for the pure logic and the full query pipeline of MedHelpEngine, using fakes
for the retriever and LLMs (no network, no llama-index needed)."""
import unittest
from types import SimpleNamespace

from app import engine as eng
from app.engine import (
    NO_ANSWER_TEXT,
    MedHelpEngine,
    classify_llm_error,
    extract_grades,
    find_unverified_grades,
    is_refusal,
    parse_jargon_json,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeNode:
    def __init__(self, text, **metadata):
        self._text, self.metadata = text, metadata

    def get_content(self, *args, **kwargs):
        return self._text


def nws(text, score=0.8, **metadata):
    """Stand-in for llama_index NodeWithScore."""
    return SimpleNamespace(node=FakeNode(text, **metadata), score=score)


class FakeRetriever:
    def __init__(self, nodes):
        self.nodes = nodes
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return self.nodes


class FakeLLM:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.prompts = reply, exc, []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if self.exc:
            raise self.exc
        return SimpleNamespace(text=self.reply)


CTX_ACC = (
    "Recommendation 1. In adults with ASCVD risk >=10%, statin therapy is recommended. "
    "COR 1, LOE A. Statins reduce LDL-C."
)
CTX_WHO = (
    "WHO recommends initiating pharmacological treatment at SBP >=140 mmHg "
    "(strong recommendation, moderate-certainty evidence)."
)

GOOD_ANSWER = (
    "WHO recommends starting drug treatment when systolic pressure is 140 mmHg or higher "
    "[WHO Hypertension 2021, Page 9].\n"
    "**Strength of Recommendation:** Strong recommendation, moderate-certainty evidence "
    "[WHO Hypertension 2021, Page 9]"
)


def make_engine(nodes, answer, jargon='{"systolic": "The top number in a blood pressure reading."}', **kw):
    retriever = FakeRetriever(nodes)
    llm, fast = FakeLLM(answer), FakeLLM(jargon)
    return MedHelpEngine(retriever=retriever, llm=llm, fast_llm=fast, min_similarity=0.25, **kw), retriever, llm, fast


# --------------------------------------------------------------------------- #
# Evidence grades
# --------------------------------------------------------------------------- #
class GradeTests(unittest.TestCase):
    def test_acc_aha_styles(self):
        grades = extract_grades("COR 1, LOE B-R. Also Class IIa, Level of Evidence C-LD and Class III (Harm).")
        joined = " | ".join(grades)
        for expected in ("COR 1", "LOE B-R", "Class IIa", "Level of Evidence C-LD", "Class III (Harm)"):
            self.assertIn(expected, joined)

    def test_grade_who_style_nested_phrase_kept_whole(self):
        grades = extract_grades(CTX_WHO)
        self.assertEqual(grades, ["strong recommendation, moderate-certainty evidence"])

    def test_cdc_style(self):
        grades = extract_grades("Recommendation category: A; Evidence type: 3")
        self.assertEqual(grades, ["Recommendation category: A", "Evidence type: 3"])

    def test_no_false_positives(self):
        text = "Class Iodine is not a thing; at a level a nurse can reach; this is a recommendation."
        self.assertEqual(extract_grades(text), [])

    def test_lowercase_article_is_not_a_level(self):
        self.assertEqual(extract_grades("the level of evidence a trial provides"), [])

    def test_equivalence_between_roman_and_arabic(self):
        self.assertEqual(find_unverified_grades("This is a Class I recommendation (Level of Evidence B-R).",
                                                "COR 1 ... LOE B-R"), [])

    def test_unverified_grade_is_flagged(self):
        missing = find_unverified_grades("Class IIa, Level of Evidence B-R", "COR 1, LOE A")
        self.assertTrue(any("IIa" in m for m in missing))
        self.assertTrue(any("B-R" in m for m in missing))

    def test_nested_certainty_is_verified(self):
        # context only says "low-certainty"; the answer claims "moderate-certainty" inside a longer phrase
        missing = find_unverified_grades("strong recommendation, moderate-certainty evidence",
                                         "strong recommendation, low-certainty evidence")
        self.assertEqual(len(missing), 1)


# --------------------------------------------------------------------------- #
# Parsing / classification helpers
# --------------------------------------------------------------------------- #
class ParsingTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_jargon_json('{"a": "b c d"}'), {"a": "b c d"})

    def test_code_fence_and_prose(self):
        raw = 'Sure! Here you go:\n```json\n{"hypertension": "High blood pressure."}\n```\nHope that helps.'
        self.assertEqual(parse_jargon_json(raw), {"hypertension": "High blood pressure."})

    def test_wrapper_object_and_list_forms(self):
        self.assertEqual(parse_jargon_json('{"terms": {"x": "def x"}}'), {"x": "def x"})
        self.assertEqual(parse_jargon_json('[{"term": "x", "definition": "def x"}]'), {"x": "def x"})

    def test_garbage_and_empty(self):
        self.assertEqual(parse_jargon_json("no json here"), {})
        self.assertEqual(parse_jargon_json(""), {})
        self.assertEqual(parse_jargon_json("{}"), {})

    def test_regex_fallback_for_truncated_json(self):
        raw = '{"systolic": "The top number.", "diastolic": "The bottom number.", "broken'
        self.assertEqual(parse_jargon_json(raw), {"systolic": "The top number.", "diastolic": "The bottom number."})

    def test_non_string_values_dropped_and_cap(self):
        self.assertEqual(parse_jargon_json('{"a": 1, "b": "ok def"}'), {"b": "ok def"})
        many = "{" + ",".join(f'"t{i}": "definition {i}"' for i in range(20)) + "}"
        self.assertEqual(len(parse_jargon_json(many, max_terms=5)), 5)

    def test_is_refusal(self):
        self.assertTrue(is_refusal(NO_ANSWER_TEXT))
        self.assertTrue(is_refusal("'I do not know based on the provided guidelines.'"))
        self.assertFalse(is_refusal("WHO says I do not know..."))

    def test_error_classification(self):
        rl = classify_llm_error(Exception("Error code: 429 - rate_limit_exceeded. Please try again in 7.5s."))
        self.assertIsInstance(rl, eng.RateLimitedError)
        self.assertAlmostEqual(rl.retry_after, 7.5)
        rl2 = classify_llm_error(Exception("429 ... Please try again in 1m26.4s"))
        self.assertAlmostEqual(rl2.retry_after, 86.4)
        self.assertIsInstance(classify_llm_error(Exception("model_decommissioned: The model x has been decommissioned")),
                              eng.ModelUnavailableError)
        self.assertIsInstance(classify_llm_error(Exception("Error code: 401 invalid_api_key")), eng.LLMAuthError)
        self.assertIsInstance(classify_llm_error(Exception("Error code: 413 Request too large")), eng.LLMError)
        self.assertEqual(classify_llm_error(Exception("boom")).status_code, 502)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
class PipelineTests(unittest.TestCase):
    def test_guardrail_short_circuits_everything(self):
        e, retriever, llm, fast = make_engine([nws(CTX_WHO)], GOOD_ANSWER)
        out = e.query("I have a headache, what do I take?")
        self.assertTrue(out["guardrail_triggered"])
        self.assertFalse(out["answered"])
        self.assertEqual(out["sources"], [])
        self.assertEqual((retriever.queries, llm.prompts, fast.prompts), ([], [], []))

    def test_no_relevant_chunks_means_no_llm_call(self):
        e, _, llm, fast = make_engine([nws(CTX_WHO, score=0.05, document_name="WHO", page_label="9")], GOOD_ANSWER)
        out = e.query("What is the capital of France?")
        self.assertEqual(out["answer"], NO_ANSWER_TEXT)
        self.assertFalse(out["answered"])
        self.assertEqual((llm.prompts, fast.prompts), ([], []))

    def test_happy_path(self):
        nodes = [
            nws(CTX_WHO, 0.82, document_name="WHO Hypertension 2021", page_label="9", pdf_page=13,
                source_url="https://example.org/who.pdf"),
            nws(CTX_ACC, 0.40, document_name="ACC/AHA 2019", page_label="e180", pdf_page=4),
        ]
        e, _, llm, fast = make_engine(nodes, GOOD_ANSWER)
        out = e.query("  What does WHO   recommend for starting treatment? ")

        self.assertTrue(out["answered"])
        self.assertFalse(out["guardrail_triggered"])
        self.assertEqual(out["answer"], GOOD_ANSWER)
        self.assertEqual(out["warnings"], [])
        self.assertEqual(out["jargon_dictionary"], {"systolic": "The top number in a blood pressure reading."})

        self.assertEqual(len(out["sources"]), 2)
        first = out["sources"][0]
        self.assertEqual(first["document_name"], "WHO Hypertension 2021")
        self.assertEqual(first["page_number"], "9")
        self.assertEqual(first["pdf_page"], 13)
        self.assertEqual(first["relevance_score"], 0.82)
        self.assertEqual(first["evidence_grades"], ["strong recommendation, moderate-certainty evidence"])
        self.assertIn("COR 1", out["sources"][1]["evidence_grades"])

        # the QA prompt carries the helpful-mode instructions, labelled context and the (normalised) question
        prompt = llm.prompts[0]
        self.assertIn("You are MedHelp", prompt)
        self.assertIn("Partial or indirect information is NOT a reason to refuse", prompt)
        self.assertIn("[WHO Hypertension 2021, Page 9]\nWHO recommends initiating", prompt)
        self.assertIn("QUESTION: What does WHO recommend for starting treatment?", prompt)
        # jargon call receives the answer without citations
        self.assertIn("Identify any highly technical medical terms", fast.prompts[0])
        self.assertNotIn("[WHO Hypertension 2021, Page 9]", fast.prompts[0])

    def test_model_refusal_is_normalised_and_shows_closest_passages(self):
        e, _, _, fast = make_engine([nws(CTX_WHO, document_name="WHO", page_label="9")],
                                    "I do not know based on the provided guidelines. Sorry!")
        out = e.query("Does the guideline discuss quantum computing?")
        self.assertEqual(out["answer"], NO_ANSWER_TEXT)
        self.assertEqual(out["kind"], "no_answer")
        self.assertEqual(out["debug"]["reason"], "model_declined")
        self.assertEqual(len(out["sources"]), 1)  # the closest passages are still offered to the user
        self.assertFalse(out["answered"])
        self.assertEqual(fast.prompts, [])

    def test_strict_style_uses_the_original_brief_prompt(self):
        from app import config
        old = config.ANSWER_STYLE
        config.ANSWER_STYLE = "strict"
        try:
            e, _, llm, _ = make_engine([nws(CTX_WHO, document_name="WHO", page_label="9")], GOOD_ANSWER)
            e.query("What does WHO recommend for starting treatment?")
        finally:
            config.ANSWER_STYLE = old
        self.assertIn("You are a medical assistant referencing clinical guidelines.", llm.prompts[0])

    def test_hallucinated_grade_and_missing_citation_produce_warnings(self):
        bad = "Statins are recommended. This is a Class IIa, Level of Evidence B-R recommendation."
        e, *_ = make_engine([nws(CTX_ACC, document_name="ACC/AHA 2019", page_label="e180")], bad)
        out = e.query("What does the guideline say about statins?")
        self.assertTrue(out["answered"])
        self.assertEqual(len(out["warnings"]), 2)
        self.assertTrue(any("citation" in w for w in out["warnings"]))
        self.assertTrue(any("IIa" in w for w in out["warnings"]))

    def test_jargon_failure_degrades_gracefully(self):
        e, *_ = make_engine([nws(CTX_WHO, document_name="WHO", page_label="9")], GOOD_ANSWER)
        e._fast_llm = FakeLLM(exc=Exception("429 rate limit. Please try again in 3s"))
        out = e.query("What does WHO recommend for starting treatment?")
        self.assertTrue(out["answered"])
        self.assertEqual(out["jargon_dictionary"], {})
        self.assertTrue(any("temporarily unavailable" in w for w in out["warnings"]))

    def test_main_llm_rate_limit_propagates_with_retry_after(self):
        e, *_ = make_engine([nws(CTX_WHO, document_name="WHO", page_label="9")], GOOD_ANSWER)
        e._llm = FakeLLM(exc=Exception("Error code: 429 - Please try again in 7.5s"))
        with self.assertRaises(eng.RateLimitedError) as ctx:
            e.query("What does WHO recommend for starting treatment?")
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertAlmostEqual(ctx.exception.retry_after, 7.5)

    def test_empty_llm_answer_is_an_error(self):
        e, *_ = make_engine([nws(CTX_WHO, document_name="WHO", page_label="9")], "")
        with self.assertRaises(eng.LLMError):
            e.query("What does WHO recommend for starting treatment?")

    def test_empty_question_rejected(self):
        e, *_ = make_engine([], "x")
        with self.assertRaises(ValueError):
            e.query("   ")

    def test_retrieval_failure_is_mapped(self):
        e, *_ = make_engine([], "x")
        e._retriever = SimpleNamespace(retrieve=lambda q: (_ for _ in ()).throw(RuntimeError("qdrant down")))
        with self.assertRaises(eng.RetrievalError):
            e.query("What does WHO recommend?")


if __name__ == "__main__":
    unittest.main()
