"""Automated download of open-access clinical guidelines.

Run it directly::

    python -m app.fetch_data              # download anything missing
    python -m app.fetch_data --force      # re-download everything
    python -m app.fetch_data --check      # only test that the URLs respond (no download)
    python -m app.fetch_data --list       # markdown table of all sources (for manual downloads)
    python -m app.fetch_data --only who   # only sources whose title/file name contains "who"
    python -m app.fetch_data --sync-manifest   # offline: give PDFs already in data/ their proper titles

What it does
------------
* Downloads a predefined list of open-access guideline PDFs (WHO, CDC, and an
  open PubMed Central article via Europe PMC) into ``data/``.
* Tries each source's URLs in order (primary, then mirrors) and keeps the first
  one that yields a *valid* PDF (``%PDF-`` header, minimum size, readable by pypdf).
  A server that answers 200 with an HTML "access denied" page is therefore
  treated as a failure rather than saved as a broken PDF.
* Retries transient errors (429/5xx) with exponential back-off.
* Never aborts the whole run because of one broken link; it reports a summary and
  exits non-zero only if *no* usable PDF is available at all.
* Writes ``data/manifest.json`` (title, publisher, URL, page count) which the
  ingestion step uses so that citations show readable document names.

Adding your own guidelines
--------------------------
Either drop PDFs into ``data/`` manually, or create ``data/custom_sources.json``::

    [{"title": "My Guideline 2024", "publisher": "Society X",
      "url": "https://example.org/guideline.pdf"}]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app import config
from app.sources import SOURCES, GuidelineSource  # noqa: F401  (SOURCES re-exported for tests/tools)

logger = logging.getLogger("medhelp.fetch")

MIN_PDF_BYTES = 20_000  # anything smaller is almost certainly an error page
CONNECT_TIMEOUT = 20
READ_TIMEOUT = 180
MANIFEST_NAME = "manifest.json"
CUSTOM_SOURCES_NAME = "custom_sources.json"

# Government/WHO CDNs frequently block the default python-requests user agent.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 MedHelp/1.0 (educational research project)"
)

# --------------------------------------------------------------------------- #
# Manifest + custom sources
# --------------------------------------------------------------------------- #
def load_manifest(data_dir: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    """Return ``{filename: metadata}`` from data/manifest.json (empty if absent)."""
    path = Path(data_dir or config.DATA_DIR) / MANIFEST_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_manifest(manifest: dict[str, dict[str, Any]], data_dir: Optional[Path] = None) -> None:
    path = Path(data_dir or config.DATA_DIR) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "guideline"


def load_sources(data_dir: Optional[Path] = None, include_custom: bool = True) -> list[GuidelineSource]:
    """Predefined sources plus any entries from data/custom_sources.json."""
    sources = list(SOURCES)
    if not include_custom:
        return sources
    path = Path(data_dir or config.DATA_DIR) / CUSTOM_SOURCES_NAME
    if not path.exists():
        return sources
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring %s (unreadable: %s)", path.name, exc)
        return sources
    for entry in entries if isinstance(entries, list) else []:
        urls = entry.get("urls") or ([entry["url"]] if entry.get("url") else [])
        if not entry.get("title") or not urls:
            logger.warning("Skipping malformed custom source: %r", entry)
            continue
        sources.append(
            {
                "title": entry["title"],
                "publisher": entry.get("publisher", "Custom"),
                "grading": entry.get("grading", "Unknown"),
                "filename": entry.get("filename") or f"{_slugify(entry['title'])}.pdf",
                "urls": urls,
            }
        )
    return sources


# --------------------------------------------------------------------------- #
# Download logic
# --------------------------------------------------------------------------- #
def build_session() -> requests.Session:
    """HTTP session with browser-like headers and retry/back-off on transient errors."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.8"})
    return session


def inspect_pdf(path: Path) -> int:
    """Validate a downloaded file and return its page count.

    Raises ValueError if the file is not a usable PDF.
    """
    size = path.stat().st_size
    if size < MIN_PDF_BYTES:
        raise ValueError(f"file too small ({size} bytes)")
    with path.open("rb") as fh:
        if fh.read(5) != b"%PDF-":
            raise ValueError("missing %PDF header (server probably returned an HTML page)")
    from pypdf import PdfReader  # imported lazily: only needed for validation

    try:
        pages = len(PdfReader(str(path), strict=False).pages)
    except Exception as exc:  # pypdf raises a variety of exception types
        raise ValueError(f"unreadable PDF: {exc}") from exc
    if pages == 0:
        raise ValueError("PDF contains no pages")
    return pages


def _download(session: requests.Session, url: str, dest: Path) -> int:
    """Stream *url* to *dest* atomically. Returns the PDF page count."""
    tmp = dest.with_name(dest.name + ".part")
    try:
        with session.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), allow_redirects=True) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
        pages = inspect_pdf(tmp)
        tmp.replace(dest)
        return pages
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Finding the PDF behind a landing page / PMC id / DOI
# --------------------------------------------------------------------------- #
_ANCHOR_RE = re.compile(r"<a\b[^>]*?\bhref\s*=\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_NOT_PDF_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js", ".ico", ".webp")
MAX_PAGE_BYTES = 3_000_000


def extract_pdf_links(html: str, base_url: str, limit: int = 6) -> list[str]:
    """Candidate PDF URLs found in a landing page, best first.

    Priority: anchors whose text says Download/PDF/Full text and that look like files; then any link
    containing ``.pdf``; then DSpace-style ``/bitstreams/<id>/content`` links (WHO IRIS). Images are skipped
    and every candidate is still validated after download, so a wrong guess costs one request, nothing more.
    """
    scored: list[tuple[int, int, str]] = []
    for order, (href, inner) in enumerate(_ANCHOR_RE.findall(html)):
        href = href.strip().replace("&amp;", "&")
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        url = urljoin(base_url, href)
        low = url.lower().split("?")[0]
        if low.endswith(_NOT_PDF_EXT):
            continue
        text = _TAG_RE.sub(" ", inner).lower()
        filelike = ".pdf" in low or "/bitstreams/" in low or "/retrieve" in low or "download" in url.lower()
        if filelike and re.search(r"download|\bpdf\b|full[\s-]*text", text):
            rank = 0
        elif ".pdf" in low:
            rank = 1
        elif "/bitstreams/" in low and low.endswith("/content"):
            rank = 2
        else:
            continue
        scored.append((rank, order, url))
    out: list[str] = []
    for _, _, url in sorted(scored):
        if url not in out:
            out.append(url)
    return out[:limit]


def _europepmc_urls(pmcid: str) -> list[str]:
    pmcid = pmcid.strip().upper()
    return [
        f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf",
        f"https://europepmc.org/articles/{pmcid}?pdf=render",
    ]


def _doi_to_pmcid(session: requests.Session, doi: str) -> Optional[str]:
    """Ask Europe PMC whether an open-access PMC copy of *doi* exists."""
    resp = session.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": f"DOI:{doi}", "format": "json", "resultType": "lite"},
        timeout=(CONNECT_TIMEOUT, 30),
    )
    resp.raise_for_status()
    for item in resp.json().get("resultList", {}).get("result", []):
        if item.get("pmcid"):
            return str(item["pmcid"])
    return None


def iter_candidate_urls(session: requests.Session, source: GuidelineSource, errors: list[str]) -> Iterator[str]:
    """Yield direct PDF URLs for *source* in order, expanding ``page:`` / ``pmc:`` / ``doi:`` entries lazily."""
    for entry in source["urls"]:
        try:
            if entry.startswith("page:"):
                page = entry[5:]
                resp = session.get(page, timeout=(CONNECT_TIMEOUT, 60))
                resp.raise_for_status()
                links = extract_pdf_links(resp.text[:MAX_PAGE_BYTES], resp.url)
                if not links:
                    errors.append(f"{page} -> no PDF link found on the page")
                yield from links
            elif entry.startswith("pmc:"):
                yield from _europepmc_urls(entry[4:])
            elif entry.startswith("doi:"):
                pmcid = _doi_to_pmcid(session, entry[4:])
                if pmcid:
                    yield from _europepmc_urls(pmcid)
                else:
                    errors.append(f"{entry} -> no open-access PubMed Central copy found")
            else:
                yield entry
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{entry} -> {type(exc).__name__}: {exc}")


def find_existing_file(data_dir: Path, source: GuidelineSource) -> Optional[Path]:
    """A PDF for *source* already in data/ under its canonical name or any alias (case-insensitive).

    An alias ending in ``*`` is a prefix match, e.g. ``heidenreich-et-al-2022*`` matches the long file name a
    journal website generates.
    """
    present = {p.name.lower(): p for p in data_dir.iterdir() if p.suffix.lower() == ".pdf"}
    for name in [source["filename"], *source.get("aliases", [])]:
        low = name.lower()
        if low.endswith("*"):
            for fname, path in present.items():
                if fname.startswith(low[:-1]):
                    return path
        elif low in present:
            return present[low]
    return None


_GENERIC_TITLE_RE = re.compile(r"untitled|microsoft|\.p65|\.indd|\.pdf$|^\W*$|supplementary material", re.I)


def guess_title(path: Path) -> str:
    """Readable title for a PDF we have no source entry for: embedded PDF title, else the first-page heading."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        meta = (reader.metadata.title or "").strip() if reader.metadata else ""
        if len(meta) >= 15 and not _GENERIC_TITLE_RE.search(meta):
            return " ".join(meta.split())[:160]
        lines = [ln.strip() for ln in (reader.pages[0].extract_text() or "").splitlines() if ln.strip()]
        heading = " ".join(ln for ln in lines[:5] if len(ln) > 3)
        if len(heading) >= 15:
            return heading[:160]
    except Exception:  # noqa: BLE001 - unreadable PDF: fall through to the file name
        pass
    return re.sub(r"[_\-]+", " ", path.stem).strip().title()


def download_source(
    session: requests.Session, source: GuidelineSource, data_dir: Path, force: bool = False
) -> dict[str, Any]:
    """Download one guideline, trying each candidate in turn. Never raises for network errors."""
    existing = find_existing_file(data_dir, source)
    dest = existing or (data_dir / source["filename"])
    result: dict[str, Any] = {"title": source["title"], "filename": dest.name, "errors": []}

    if existing and not force:
        try:
            result.update(status="cached", pages=inspect_pdf(existing), url=None)
            return result
        except ValueError as exc:
            logger.warning("Cached %s is invalid (%s); re-downloading", existing.name, exc)
            existing.unlink(missing_ok=True)

    for url in iter_candidate_urls(session, source, result["errors"]):
        logger.info("Downloading %s\n    from %s", dest.name, url)
        try:
            pages = _download(session, url, dest)
        except (requests.RequestException, ValueError, OSError) as exc:
            msg = f"{url} -> {type(exc).__name__}: {exc}"
            logger.warning("    failed: %s", msg)
            result["errors"].append(msg)
            continue
        result.update(status="downloaded", url=url, pages=pages)
        logger.info("    OK (%d pages)", pages)
        return result

    result["status"] = "failed"
    return result


def manifest_entry(source: GuidelineSource, pages: Optional[int], url: Optional[str], previous: Optional[dict] = None) -> dict[str, Any]:
    previous = previous or {}
    return {
        "title": source["title"],
        "publisher": source.get("publisher", ""),
        "grading": source.get("grading", ""),
        "topic": source.get("topic", ""),
        "year": source.get("year"),
        "url": source.get("landing") or url or previous.get("url") or source["urls"][0],
        "pages": pages if pages is not None else previous.get("pages"),
        "retrieved_at": previous.get("retrieved_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def sync_manifest(data_dir: Optional[Path] = None, include_custom: bool = True) -> dict[str, dict[str, Any]]:
    """Offline: make data/manifest.json describe every known PDF already sitting in data/.

    PDFs saved by hand (e.g. ``rr7004a1-H.pdf`` from a browser download) are matched to their source by file
    name or alias and get a readable title; unknown PDFs get a title taken from the PDF itself (embedded title or first-page heading).
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    if not data_dir.exists():
        return {}
    manifest = load_manifest(data_dir)
    changed = False
    for source in load_sources(data_dir, include_custom=include_custom):
        path = find_existing_file(data_dir, source)
        if not path:
            continue
        previous = manifest.get(path.name)
        if previous and previous.get("title") == source["title"] and previous.get("pages"):
            continue
        try:
            pages = inspect_pdf(path)
        except ValueError:
            continue
        manifest[path.name] = manifest_entry(source, pages, None, previous)
        changed = True
    known = {name for name in manifest}
    for path in sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".pdf"):
        if path.name in known:
            continue
        try:
            pages = inspect_pdf(path)
        except ValueError:
            continue
        manifest[path.name] = {
            "title": guess_title(path), "publisher": "", "grading": "", "topic": "", "year": None, "url": "",
            "pages": pages, "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "auto_title": True,
        }
        changed = True
    if changed:
        save_manifest(manifest, data_dir)
    return manifest


def select_sources(sources: list[GuidelineSource], only: Optional[str]) -> list[GuidelineSource]:
    if not only:
        return sources
    needle = only.lower()
    return [s for s in sources if needle in s["filename"].lower() or needle in s["title"].lower() or needle in s.get("publisher", "").lower()]


def fetch_all(
    data_dir: Optional[Path] = None, force: bool = False, include_custom: bool = True, only: Optional[str] = None
) -> list[dict[str, Any]]:
    """Download every configured guideline and update ``manifest.json``."""
    data_dir = Path(data_dir or config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = select_sources(load_sources(data_dir, include_custom=include_custom), only)
    manifest = load_manifest(data_dir)
    results: list[dict[str, Any]] = []

    session = build_session()
    try:
        for n, source in enumerate(sources, start=1):
            logger.info("[%d/%d] %s", n, len(sources), source["title"])
            result = download_source(session, source, data_dir, force=force)
            results.append(result)
            if result["status"] in {"downloaded", "cached"}:
                previous = manifest.get(result["filename"], {})
                entry = manifest_entry(source, result.get("pages"), result.get("url"), previous)
                if result["status"] == "downloaded":
                    entry["retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                manifest[result["filename"]] = entry
    finally:
        session.close()

    save_manifest(manifest, data_dir)
    return results


def check_links(include_custom: bool = True, only: Optional[str] = None) -> list[tuple[str, str, str]]:
    """Report HTTP status for every URL without saving anything."""
    rows: list[tuple[str, str, str]] = []
    session = build_session()
    try:
        for source in select_sources(load_sources(include_custom=include_custom), only):
            for entry in source["urls"]:
                if entry.startswith(("pmc:", "doi:")):
                    rows.append((source["filename"], "resolved at download", entry))
                    continue
                is_page = entry.startswith("page:")
                url = entry[5:] if is_page else entry
                try:
                    with session.get(url, stream=True, timeout=(CONNECT_TIMEOUT, 30)) as resp:
                        ctype = resp.headers.get("Content-Type", "?").split(";")[0]
                        extra = ""
                        if is_page and resp.ok:
                            found = extract_pdf_links(resp.text[:MAX_PAGE_BYTES], resp.url)
                            extra = f" ({len(found)} PDF link(s) found)"
                        rows.append((source["filename"], f"{resp.status_code} {ctype}{extra}", entry))
                except requests.RequestException as exc:
                    rows.append((source["filename"], f"ERROR {type(exc).__name__}", entry))
    finally:
        session.close()
    return rows


def sources_markdown(include_custom: bool = True) -> str:
    """Markdown table of every source: handy for downloading files by hand into data/."""
    lines = [
        "| # | Document | Publisher | Year | Direct PDF link | Landing page | Save as (or any alias) |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(load_sources(include_custom=include_custom), start=1):
        aliases = ", ".join(f"`{a}`" for a in s.get("aliases", [])[:2])
        name = f"`{s['filename']}`" + (f" or {aliases}" if aliases else "")
        direct = next((u for u in s["urls"] if u.startswith("https://")), "(found automatically from the landing page)")
        lines.append(
            f"| {i} | {s['title']} | {s.get('publisher', '')} | {s.get('year', '')} | {direct} | {s.get('landing', '')} | {name} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download open-access clinical guideline PDFs into data/.")
    parser.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    parser.add_argument("--check", action="store_true", help="only test that URLs respond; download nothing")
    parser.add_argument("--no-custom", action="store_true", help="ignore data/custom_sources.json")
    parser.add_argument("--only", metavar="TEXT", help="only sources whose title / file name / publisher contains TEXT")
    parser.add_argument("--list", action="store_true", help="print a markdown table of all sources and exit")
    parser.add_argument("--sync-manifest", action="store_true", help="offline: write titles for PDFs already in data/")
    args = parser.parse_args(argv)

    logging.basicConfig(level=config.LOG_LEVEL, format="%(levelname)s  %(message)s")

    if args.list:
        print(sources_markdown(include_custom=not args.no_custom))
        return 0
    if args.sync_manifest:
        manifest = sync_manifest(include_custom=not args.no_custom)
        print(f"manifest.json now describes {len(manifest)} PDF(s).")
        return 0
    if args.check:
        for filename, status, url in check_links(include_custom=not args.no_custom, only=args.only):
            print(f"{status:<40} {filename:<44} {url}")
        return 0

    results = fetch_all(force=args.force, include_custom=not args.no_custom, only=args.only)

    print("\nSummary")
    print("-------")
    for r in results:
        pages = f"{r['pages']} pages" if r.get("pages") else ""
        print(f"[{r['status']:<10}] {r['filename']:<42} {pages}")
        for err in r["errors"] if r["status"] == "failed" else []:
            print(f"             ! {err}")

    usable = sorted(Path(config.DATA_DIR).glob("*.pdf"))
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print(
            f"\n{len(failed)} download(s) failed. The pipeline can still run with the "
            f"{len(usable)} PDF(s) available; fix the URLs in app/fetch_data.py or drop PDFs into data/."
        )
    if not usable:
        print("\nNo usable PDFs in data/. Check your network connection and try again.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
