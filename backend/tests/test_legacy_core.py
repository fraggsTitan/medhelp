"""Tests for hybrid retrieval, reference filtering, small talk, follow-ups, link resolvers, the source
list and the chat front-end (run against a stub Streamlit, so no Streamlit install is needed)."""
import http.server
import io
import json
import runpy
import sys
import tempfile
import threading
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pypdf import PdfWriter

from app import engine as eng
from app import fetch_data, sources, textutils
from app.engine import MedHelpEngine, check_guardrails, detect_smalltalk, fuse_hits, pack_hits, Hit, looks_like_followup
from app.lexical import LexicalIndex, payload_to_record, load_index_from_qdrant, tokenize
from tests.test_engine import FakeLLM, FakeRetriever, GOOD_ANSWER, nws, CTX_WHO

REF_PAGE = (
    "1. Smith J, Doe A, et al. Aspirin in primary prevention. J Am Coll Cardiol. 2014;64:959-68. [PubMed: 25190228] "
    "2. Lee K, Park S, et al. Statins and risk. Circulation. 2016;134:e123-45. doi:10.1161/CIR.000 "
    "3. Brown T, et al. Blood pressure targets. N Engl J Med. 2015;373:2103-16. PMID 26551272 "
    "4. WHO. Guideline. Geneva: World Health Organization; 2021. Available at https://iris.who.int accessed 2022."
)


class TextUtilTests(unittest.TestCase):
    def test_reference_detection(self):
        self.assertTrue(textutils.is_reference_text(REF_PAGE))
        self.assertFalse(textutils.is_reference_text(CTX_WHO * 3))
        self.assertFalse(textutils.is_reference_text("et al. 2014;64:959"))  # too short to judge

    def test_repeated_headers_removed_but_content_kept(self):
        drugs = ["aspirin", "warfarin", "heparin", "statin", "insulin", "digoxin", "ceftriaxone", "azithromycin"]
        pages = [f"MMWR / July 23, 2021 / Vol. 70 / No. 4 {n}\nUnique content about {d} dosing.\n{n}" for n, d in enumerate(drugs)]
        out = textutils.strip_repeated_lines(pages)
        self.assertTrue(all("MMWR" not in p for p in out))
        self.assertIn("Unique content about statin dosing.", out[3])


class LexicalTests(unittest.TestCase):
    def setUp(self):
        recs = [
            ("a", "Ceftriaxone 500 mg intramuscularly is recommended for uncomplicated gonorrhea.", {"document_name": "CDC STI", "page_label": "5"}),
            ("b", "Thiazide diuretics, ACE inhibitors and calcium channel blockers are first-line for hypertension.", {"document_name": "WHO HTN", "page_label": "9"}),
            ("c", "Physical activity for children and adolescents.", {"document_name": "WHO PA", "page_label": "2"}),
        ]
        self.index = LexicalIndex.from_records(recs)

    def test_finds_exact_terms_and_stems(self):
        hits = self.index.search("treatment of gonorrhea with ceftriaxone")
        self.assertEqual(hits[0].id, "a")
        self.assertEqual(self.index.search("first-line drugs for hypertensive adults")[0].id, "b")

    def test_unrelated_query_returns_nothing(self):
        self.assertEqual(self.index.search("quantum chromodynamics lattice"), [])

    def test_stopwords_and_stemming(self):
        self.assertEqual(tokenize("What does the guideline recommend for treated patients?"), ["treat", "patient"])

    def test_qdrant_payload_parsing_and_loading(self):
        node = json.dumps({"text": "Statins reduce LDL-C in adults.", "metadata": {"document_name": "ACC", "page_label": "3"}})
        rec = payload_to_record("id1", {"_node_content": node, "file_name": "x.pdf"})
        self.assertEqual(rec[1], "Statins reduce LDL-C in adults.")
        self.assertEqual(rec[2]["document_name"], "ACC")
        self.assertIsNone(payload_to_record("id2", {"_node_content": "{not json"}))

        pts = [SimpleNamespace(id="1", payload={"text": "Good clinical content about statins and LDL cholesterol." * 5}),
               SimpleNamespace(id="2", payload={"text": REF_PAGE})]

        class FakeClient:
            def scroll(self, **kw):
                return (pts, None) if kw["offset"] is None else ([], None)

        index = load_index_from_qdrant(FakeClient(), "c")
        self.assertEqual(len(index), 1)  # the bibliography chunk was skipped


class RetrievalLogicTests(unittest.TestCase):
    def test_fusion_dedupes_and_boosts_double_hits(self):
        d = [Hit("1", "alpha text", {"document_name": "A", "page_label": "1"}, 0.8),
             Hit("2", "beta text", {"document_name": "A", "page_label": "2"}, 0.7)]
        k = [Hit("3", "gamma text", {"document_name": "B", "page_label": "1"}, None),
             Hit("2", "beta text", {"document_name": "A", "page_label": "2"}, None)]
        fused = fuse_hits(d, k)
        self.assertEqual([h.id for h in fused][0], "2")  # found by both retrievers -> first
        self.assertEqual({h.id for h in fused}, {"1", "2", "3"})
        self.assertEqual(fused[0].match_type, "both")
        self.assertEqual([h for h in fused if h.id == "3"][0].match_type, "keyword")

    def test_pack_respects_budget_and_top_k(self):
        hits = [Hit(str(i), "x" * 1000, {}, 0.5) for i in range(10)]
        self.assertEqual(len(pack_hits(hits, top_k=6, char_budget=100000)), 6)
        self.assertEqual(len(pack_hits(hits, top_k=6, char_budget=2500)), 2)
        self.assertEqual(len(pack_hits(hits, top_k=6, char_budget=10)), 1)  # always at least one

    def test_reference_chunks_are_dropped_before_the_llm(self):
        nodes = [nws(REF_PAGE, 0.9, document_name="ACC", page_label="65"),
                 nws(CTX_WHO, 0.6, document_name="WHO", page_label="9")]
        e = MedHelpEngine(retriever=FakeRetriever(nodes), llm=FakeLLM(GOOD_ANSWER), fast_llm=FakeLLM("{}"), min_similarity=0.25)
        out = e.query("What does WHO recommend for starting treatment?")
        self.assertEqual([s["document_name"] for s in out["sources"]], ["WHO"])
        self.assertEqual(out["debug"]["reference_chunks_dropped"], 1)

    def test_only_reference_pages_gives_explainable_no_answer(self):
        e = MedHelpEngine(retriever=FakeRetriever([nws(REF_PAGE, 0.9, document_name="ACC", page_label="65")]),
                         llm=FakeLLM(GOOD_ANSWER), fast_llm=FakeLLM("{}"))
        out = e.query("What does the guideline say about aspirin?")
        self.assertEqual(out["kind"], "no_answer")
        self.assertEqual(out["debug"]["reason"], "only_reference_pages")

    def test_keyword_hits_rescue_a_weak_semantic_result(self):
        lex = LexicalIndex.from_records([("k1", "Ceftriaxone 500 mg IM single dose treats uncomplicated gonorrhea.",
                                          {"document_name": "CDC STI", "page_label": "76"})])
        e = MedHelpEngine(retriever=FakeRetriever([]), llm=FakeLLM(GOOD_ANSWER), fast_llm=FakeLLM("{}"), lexical=lex)
        out = e.query("What is the treatment for uncomplicated gonorrhea?")
        self.assertTrue(out["answered"])
        self.assertEqual(out["sources"][0]["match_type"], "keyword")
        self.assertIsNone(out["sources"][0]["relevance_score"])


class ChatBehaviourTests(unittest.TestCase):
    def test_smalltalk_needs_no_index_or_llm(self):
        e = MedHelpEngine(retriever=FakeRetriever([]), llm=FakeLLM("x"), fast_llm=FakeLLM("x"))
        for text, cat in [("hi", "greeting"), ("Hello!", "greeting"), ("thanks a lot", "thanks"),
                          ("what can you do?", "capabilities"), ("Which guidelines do you have?", "documents")]:
            with self.subTest(text=text):
                self.assertEqual(detect_smalltalk(text), cat)
                out = e.query(text)
                self.assertEqual(out["kind"], "smalltalk")
                self.assertEqual(e._llm.prompts, [])
        self.assertIsNone(detect_smalltalk("What does WHO recommend for hypertension?"))
        self.assertIsNone(detect_smalltalk("hi, what is the first-line treatment for gonorrhea?"))

    def test_followup_is_rewritten_using_history(self):
        fast = FakeLLM("What does WHO recommend for hypertension in pregnancy?")
        retriever = FakeRetriever([nws(CTX_WHO, 0.8, document_name="WHO", page_label="9")])
        e = MedHelpEngine(retriever=retriever, llm=FakeLLM(GOOD_ANSWER), fast_llm=fast)
        history = [{"role": "user", "content": "What does WHO recommend for hypertension?"},
                   {"role": "assistant", "content": "WHO recommends thiazides [WHO, Page 9]."}]
        out = e.query("What about in pregnancy?", history=history)
        self.assertEqual(retriever.queries, ["What does WHO recommend for hypertension in pregnancy?"])
        self.assertEqual(out["debug"]["standalone_question"], "What does WHO recommend for hypertension in pregnancy?")
        self.assertIn("FOLLOW-UP: What about in pregnancy?", fast.prompts[0])

    def test_standalone_question_skips_rewriting(self):
        fast = FakeLLM('{"a": "b c d"}')
        e = MedHelpEngine(retriever=FakeRetriever([nws(CTX_WHO, 0.8, document_name="WHO", page_label="9")]),
                         llm=FakeLLM(GOOD_ANSWER), fast_llm=fast)
        e.query("What does the WHO guideline recommend as first-line drug treatment for hypertension in adults?",
                history=[{"role": "user", "content": "hi"}])
        self.assertTrue(all("FOLLOW-UP" not in p for p in fast.prompts))
        self.assertTrue(looks_like_followup("and for children?"))

    def test_rewrite_failure_falls_back_to_raw_question(self):
        retriever = FakeRetriever([nws(CTX_WHO, 0.8, document_name="WHO", page_label="9")])
        e = MedHelpEngine(retriever=retriever, llm=FakeLLM(GOOD_ANSWER), fast_llm=FakeLLM(exc=Exception("429 rate limit")))
        e.query("what about children?", history=[{"role": "user", "content": "hypertension?"}])
        self.assertEqual(retriever.queries, ["what about children?"])

    def test_guardrail_no_longer_blocks_general_should_i_questions(self):
        self.assertFalse(check_guardrails("What should I know about statin side effects according to the guideline?").triggered)
        self.assertTrue(check_guardrails("Should I stop taking my statin?").triggered)


# --------------------------------------------------------------------------- #
# Downloader: link discovery
# --------------------------------------------------------------------------- #
def _pdf() -> bytes:
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.add_blank_page(width=200, height=200)
    b = io.BytesIO()
    w.write(b)
    return b.getvalue()


PDF = _pdf()
WHO_LIKE_PAGE = (
    '<html><body><img src="/cover.png"><a href="/server/api/core/bitstreams/img-id/content"><img src="x.png"></a>'
    '<a class="btn" href="/server/api/core/bitstreams/doc-id/content">Download (4.9 MB)</a>'
    '<a href="/terms">Terms</a></body></html>'
)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        routes = {
            "/page": (WHO_LIKE_PAGE.encode(), "text/html"),
            "/server/api/core/bitstreams/doc-id/content": (PDF, "application/pdf"),
            "/server/api/core/bitstreams/img-id/content": (b"\x89PNG" + b"0" * 300, "image/png"),
            "/empty": (b"<html><a href='/terms'>x</a></html>", "text/html"),
        }
        body, ctype = routes.get(self.path, (b"nope", "text/plain"))
        self.send_response(200 if self.path in routes else 404)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class ResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls._min = fetch_data.MIN_PDF_BYTES
        fetch_data.MIN_PDF_BYTES = 100

    @classmethod
    def tearDownClass(cls):
        fetch_data.MIN_PDF_BYTES = cls._min
        cls.server.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.session = fetch_data.build_session()
        self.addCleanup(self.session.close)

    def test_extract_pdf_links_prefers_download_anchor(self):
        links = fetch_data.extract_pdf_links(WHO_LIKE_PAGE, "https://www.who.int/publications/i/item/1")
        self.assertEqual(links[0], "https://www.who.int/server/api/core/bitstreams/doc-id/content")
        self.assertNotIn("https://www.who.int/cover.png", links)

    def test_landing_page_is_scraped_and_images_rejected(self):
        src = {"title": "T", "filename": "t.pdf", "urls": [f"page:{self.base}/page"]}
        r = fetch_data.download_source(self.session, src, self.dir)
        self.assertEqual(r["status"], "downloaded")
        self.assertTrue(r["url"].endswith("doc-id/content"))

    def test_page_without_pdf_link_fails_cleanly(self):
        src = {"title": "T", "filename": "t.pdf", "urls": [f"page:{self.base}/empty"]}
        r = fetch_data.download_source(self.session, src, self.dir)
        self.assertEqual(r["status"], "failed")
        self.assertTrue(any("no PDF link" in e for e in r["errors"]))

    def test_pmc_and_doi_expansion(self):
        self.assertIn("accid=PMC123&blobtype=pdf", fetch_data._europepmc_urls("pmc123")[0])
        fake = SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"resultList": {"result": [{"pmcid": "PMC999"}]}})
        with mock.patch.object(self.session, "get", return_value=fake):
            self.assertEqual(fetch_data._doi_to_pmcid(self.session, "10.1/x"), "PMC999")

    def test_alias_file_is_recognised_not_redownloaded_and_manifest_synced(self):
        (self.dir / "RR7004A1-H.PDF").write_bytes(PDF)  # saved by hand, odd capitalisation
        src = {"title": "STI Guidelines", "filename": "cdc_sti.pdf", "urls": [f"{self.base}/nope.pdf"],
               "aliases": ["rr7004a1-H.pdf"], "publisher": "CDC"}
        r = fetch_data.download_source(self.session, src, self.dir)
        self.assertEqual((r["status"], r["filename"]), ("cached", "RR7004A1-H.PDF"))
        with mock.patch.object(fetch_data, "SOURCES", [src]):
            manifest = fetch_data.sync_manifest(self.dir, include_custom=False)
        self.assertEqual(manifest["RR7004A1-H.PDF"]["title"], "STI Guidelines")

    def test_only_filter(self):
        picked = fetch_data.select_sources(sources.SOURCES, "kdigo")
        self.assertEqual(len(picked), 2)


class SourceListTests(unittest.TestCase):
    def test_enough_unique_well_formed_documents(self):
        s = sources.SOURCES
        self.assertGreaterEqual(len(s), 40)
        self.assertEqual(len({x["filename"] for x in s}), len(s))
        self.assertEqual(len({x["title"] for x in s}), len(s))
        names = [n.lower() for x in s for n in [x["filename"], *x.get("aliases", [])]]
        self.assertEqual(len(names), len(set(names)), "file names/aliases must be unique across sources")
        for x in s:
            self.assertTrue(x["urls"] and x["title"] and x["publisher"])
            self.assertTrue(x["filename"].endswith(".pdf"))
        # the five PDFs the project already shipped keep working under their original names
        for legacy in ("9789240033986-eng.pdf", "rr7004a1-H.pdf", "rr7103a1-H.pdf", "nihms-1643235.pdf", "Bookshelf_NBK409108.pdf"):
            self.assertIn(legacy.lower(), names)

    def test_markdown_listing(self):
        md = fetch_data.sources_markdown()
        self.assertIn("who_tb_treatment_care_2025.pdf", md)
        self.assertEqual(md.count("\n"), len(sources.SOURCES) + 1)


# --------------------------------------------------------------------------- #
# Ingestion: page-level reading (no llama-index / network needed)
# --------------------------------------------------------------------------- #
def make_text_pdf(page_texts, labels=None) -> bytes:
    """A minimal valid multi-page text PDF (Helvetica), optionally with /PageLabels."""
    objs = []
    n = len(page_texts)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    page_labels = ""
    if labels:
        nums = " ".join(f"{i} << /P ({lab}) >>" for i, lab in enumerate(labels))
        page_labels = f" /PageLabels << /Nums [{nums}] >>"
    objs.append(f"<< /Type /Catalog /Pages 2 0 R{page_labels} >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(page_texts):
        stream = "BT /F1 12 Tf 50 700 Td (" + text.replace("(", "[").replace(")", "]") + ") Tj ET"
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


class FakeDocument:
    def __init__(self, text="", metadata=None, excluded_embed_metadata_keys=None, excluded_llm_metadata_keys=None):
        self.text, self.metadata = text, metadata or {}
        self.excluded_embed_metadata_keys = excluded_embed_metadata_keys
        self.excluded_llm_metadata_keys = excluded_llm_metadata_keys


class IngestionReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        long = "Thiazide diuretics are recommended as first line therapy for adults with hypertension. " * 3
        pages = [long + f"Section {i}." for i in range(5)] + [REF_PAGE.replace("(", "[").replace(")", "]")]
        (self.dir / "guide.pdf").write_bytes(make_text_pdf(pages, labels=["e10", "e11", "e12", "e13", "e14", "e15"]))
        self.mods = {"llama_index": types.ModuleType("llama_index"), "llama_index.core": types.ModuleType("llama_index.core")}
        self.mods["llama_index.core"].Document = FakeDocument

    def test_read_pdf_pages_gives_one_entry_per_page_with_real_labels(self):
        from app import ingestion
        pages = ingestion.read_pdf_pages(self.dir / "guide.pdf")
        self.assertEqual([label for label, _ in pages], ["e10", "e11", "e12", "e13", "e14", "e15"])
        self.assertIn("Thiazide", pages[0][1])

    def test_load_documents_is_per_page_skips_bibliography_and_stamps_version(self):
        from app import ingestion
        (self.dir / "manifest.json").write_text(json.dumps({"guide.pdf": {"title": "My Guideline", "publisher": "WHO", "url": "https://u"}}))
        with mock.patch.dict(sys.modules, self.mods):
            docs = ingestion.load_documents(self.dir, only=["guide.pdf"])
        self.assertEqual(len(docs), 5)  # the bibliography page was dropped
        self.assertEqual([d.metadata["pdf_page"] for d in docs], [1, 2, 3, 4, 5])
        self.assertEqual(docs[0].metadata["page_label"], "e10")  # real printed label, not "1"
        self.assertEqual(docs[0].metadata["document_name"], "My Guideline")
        self.assertEqual(docs[0].metadata["ingest_version"], ingestion.INGEST_VERSION)
        self.assertIn("ingest_version", docs[0].excluded_embed_metadata_keys)

    def test_unknown_pdf_gets_a_title_from_its_content(self):
        pdf = make_text_pdf(["Guidelines for the management of chronic hepatitis B infection in adults and children"])
        (self.dir / "B0xx-eng.pdf").write_bytes(pdf)
        self.assertIn("hepatitis B", fetch_data.guess_title(self.dir / "B0xx-eng.pdf"))
        with mock.patch.object(fetch_data, "MIN_PDF_BYTES", 100):
            manifest = fetch_data.sync_manifest(self.dir, include_custom=False)
        self.assertIn("B0xx-eng.pdf", manifest)
        self.assertTrue(manifest["B0xx-eng.pdf"]["auto_title"])

    def test_wildcard_alias(self):
        (self.dir / "heidenreich-et-al-2022-very-long-name.pdf").write_bytes(PDF)
        src = {"title": "HF", "filename": "hf.pdf", "urls": ["https://x"], "aliases": ["heidenreich-et-al-2022*"]}
        self.assertEqual(fetch_data.find_existing_file(self.dir, src).name, "heidenreich-et-al-2022-very-long-name.pdf")

    def test_audit_reports_bloat_and_stale_versions(self):
        from app import ingestion
        pts = [SimpleNamespace(payload={"file_name": "a.pdf", "document_name": "A", "pdf_page": 1, "ingest_version": "2"}),
               SimpleNamespace(payload={"file_name": "a.pdf", "document_name": "A old", "pdf_page": 1}),
               SimpleNamespace(payload={"file_name": "gone.pdf", "document_name": "G", "pdf_page": 7, "ingest_version": "2"})]

        class C:
            def scroll(self, **kw):
                return (pts, None) if kw["offset"] is None else ([], None)

        rep = ingestion.audit_collection(C(), ["a.pdf", "new.pdf"])
        self.assertEqual(rep["total_points"], 3)
        self.assertEqual(rep["ingest_versions"], {"2": 2, "none": 1})
        self.assertEqual(rep["files_with_several_titles"], ["a.pdf"])
        self.assertEqual(rep["files_not_in_data_dir"], ["gone.pdf"])
        self.assertEqual(rep["files_in_data_dir_but_not_indexed"], ["new.pdf"])


class StaleIndexWarningTests(unittest.TestCase):
    def _engine(self, nodes, answer="I do not know based on the provided guidelines."):
        return MedHelpEngine(retriever=FakeRetriever(nodes), llm=FakeLLM(answer), fast_llm=FakeLLM("{}"))

    def test_old_index_with_page_one_everywhere_is_flagged(self):
        nodes = [nws(CTX_WHO + str(i), 0.6, document_name=f"Doc {i}", page_label="1") for i in range(6)]
        out = self._engine(nodes).query("What does WHO recommend for hypertension?")
        self.assertEqual(out["kind"], "no_answer")
        self.assertTrue(any("--recreate" in w and "page 1" in w for w in out["warnings"]))

    def test_healthy_results_are_not_flagged(self):
        nodes = [nws(CTX_WHO, 0.6, document_name="WHO Hypertension 2021", page_label="9", ingest_version="2")]
        nodes += [nws(CTX_WHO + str(i), 0.6, document_name=f"Doc {i}", page_label=str(10 + i), ingest_version="2") for i in range(5)]
        out = self._engine(nodes, GOOD_ANSWER).query("What does WHO recommend for hypertension?")
        self.assertEqual(out["warnings"], [])
        self.assertNotIn("index_warning", out["debug"])

    def test_citation_that_matches_no_retrieved_passage_is_flagged(self):
        nodes = [nws(CTX_WHO + str(i), 0.6, document_name=f"Doc {i}", page_label=str(10 + i), ingest_version="2") for i in range(6)]
        out = self._engine(nodes, GOOD_ANSWER).query("What does WHO recommend for hypertension?")
        self.assertTrue(any("Citation check" in w for w in out["warnings"]))
        self.assertEqual({v["status"] for v in out["verification"]}, {"source_not_found"})
