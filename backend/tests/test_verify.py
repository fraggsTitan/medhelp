import unittest

from app.verify import split_sentences, summarize, verification_warning, verify_citations

PASSAGE = (
    "WHO recommends initiating pharmacological treatment in adults with a confirmed systolic blood pressure of "
    "140 mmHg or higher. For people with cardiovascular disease, treatment is recommended at 130 mmHg. "
    "Statins reduce LDL cholesterol."
)
SOURCES = [{"document_name": "WHO Hypertension 2021", "page_number": "9", "pdf_page": 13, "source_text": PASSAGE}]


class VerifyTests(unittest.TestCase):
    def test_supported_statement_has_evidence(self):
        ans = "WHO recommends starting drug treatment when systolic pressure is 140 mmHg or higher [WHO Hypertension 2021, Page 9]."
        res = verify_citations(ans, SOURCES)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["status"], "supported")
        self.assertIn("140 mmHg", res[0]["evidence"])
        self.assertEqual(res[0]["missing_numbers"], [])

    def test_number_not_in_passage_is_flagged(self):
        ans = "Treatment is recommended when systolic pressure reaches 160 mmHg [WHO Hypertension 2021, Page 9]."
        res = verify_citations(ans, SOURCES)
        self.assertIn(res[0]["status"], ("partial", "unsupported"))
        self.assertEqual(res[0]["missing_numbers"], ["160"])
        self.assertIn("do not appear", verification_warning(res) or "")

    def test_unrelated_claim_is_unsupported(self):
        ans = "Insulin pumps improve glycaemic control in children [WHO Hypertension 2021, Page 9]."
        self.assertEqual(verify_citations(ans, SOURCES)[0]["status"], "unsupported")

    def test_citation_to_unknown_page_or_document(self):
        for cite in ("[WHO Hypertension 2021, Page 77]", "[Some Other Guide, Page 9]"):
            res = verify_citations(f"WHO recommends drug treatment for high blood pressure {cite}.", SOURCES)
            self.assertEqual(res[0]["status"], "source_not_found")
        self.assertIn("could not be matched", verification_warning(res))

    def test_pdf_page_number_also_matches(self):
        res = verify_citations("WHO recommends drug treatment at 140 mmHg or higher [WHO Hypertension 2021, Page 13].", SOURCES)
        self.assertEqual(res[0]["status"], "supported")

    def test_no_citations_or_trivial_claims_give_nothing(self):
        self.assertEqual(verify_citations("Plain text without any citation.", SOURCES), [])
        self.assertEqual(verify_citations("Yes [WHO Hypertension 2021, Page 9].", SOURCES), [])

    def test_bullets_and_grade_line(self):
        ans = (
            "- WHO recommends drug treatment at 140 mmHg or higher [WHO Hypertension 2021, Page 9]\n"
            "- Statins reduce LDL cholesterol [WHO Hypertension 2021, Page 9]\n"
            "**Strength of Recommendation:** Not stated in the retrieved text."
        )
        res = verify_citations(ans, SOURCES)
        self.assertEqual(len(res), 2)
        self.assertEqual(summarize(res)["supported"], 2)
        self.assertIsNone(verification_warning(res))

    def test_split_sentences(self):
        self.assertEqual(split_sentences("One thing. Another thing [A, Page 1].\n- bullet item"),
                         ["One thing.", "Another thing [A, Page 1].", "bullet item"])


if __name__ == "__main__":
    unittest.main()


class MultiCitationTests(unittest.TestCase):
    ACC = {"document_name": "ACC HTN 2017", "page_number": "4", "source_text": "ACC/AHA recommends drug treatment for adults at a blood pressure of 130/80 mmHg or higher."}

    def test_sentence_citing_two_publishers_is_judged_on_their_combined_text(self):
        ans = "WHO uses a systolic threshold of 140 mmHg [WHO Hypertension 2021, Page 9], ACC/AHA uses 130/80 mmHg [ACC HTN 2017, Page 4]."
        res = verify_citations(ans, SOURCES + [self.ACC])
        self.assertEqual([r["status"] for r in res], ["supported", "supported"])
        self.assertEqual(res[0]["missing_numbers"], [])

    def test_a_citation_that_adds_nothing_is_flagged(self):
        irrelevant = {"document_name": "GINA", "page_number": "2", "source_text": "Inhaled corticosteroids are the cornerstone of asthma therapy."}
        ans = "WHO recommends drug treatment at 140 mmHg or higher [WHO Hypertension 2021, Page 9] [GINA, Page 2]."
        res = verify_citations(ans, SOURCES + [irrelevant])
        self.assertEqual({r["document"]: r["status"] for r in res}, {"WHO Hypertension 2021": "supported", "GINA": "unsupported"})

    def test_missing_number_is_still_caught_across_several_citations(self):
        ans = "WHO uses 150 mmHg [WHO Hypertension 2021, Page 9] and ACC/AHA uses 130/80 mmHg [ACC HTN 2017, Page 4]."
        res = verify_citations(ans, SOURCES + [self.ACC])
        self.assertTrue(all(r["missing_numbers"] == ["150"] for r in res))
