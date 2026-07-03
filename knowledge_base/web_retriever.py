"""
knowledge_base/web_retriever.py

Federated live retrieval over PubMed, Europe PMC, and ClinicalTrials.gov,
replacing/augmenting the static local FAISS index in knowledge_base/retriever.py.

Design goals (addresses AM-RAG concerns #1 and #2):
  1. Pull real clinical evidence from public sources at query time instead of
     only scraping knowledge_base/documents/*.md.
  2. Return clean, sentence-level EvidenceChunk objects with a direct source_url,
     so the UI can show "[cited sentence] -> [link]" instead of ragged chunks.

All three source APIs are free and require no API key for basic use.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Optional

import httpx


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class EvidenceChunk:
    text: str                  # a single cited sentence (or 2-3 sentence span), never a raw broken chunk
    source_url: str            # direct link to the article / trial record
    title: str
    source_type: str           # "pubmed" | "europepmc" | "ctgov" | "local"
    journal: Optional[str] = None
    year: Optional[int] = None
    pmid: Optional[str] = None
    relevance_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Sentence chunker
# --------------------------------------------------------------------------

# Lightweight sentence splitter -- avoids adding spacy/nltk as a hard dependency
# on a zero-cost deployment. Handles common abbreviations that trip up naive
# splitting on ophthalmology/clinical text (e.g. "Dr.", "Fig.", "et al.", "e.g.").
_ABBREVIATIONS = {
    "dr.", "fig.", "et al.", "e.g.", "i.e.", "vs.", "approx.", "no.",
    "mg.", "ml.", "vol.", "pp.",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_into_sentences(text: str) -> list[str]:
    """Split text into clean sentences, never returning a mid-sentence fragment."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    raw_sentences = _SENTENCE_SPLIT_RE.split(text)

    # Re-merge accidental splits after abbreviations
    sentences: list[str] = []
    buffer = ""
    for s in raw_sentences:
        buffer = f"{buffer} {s}".strip() if buffer else s
        tail = buffer.rsplit(" ", 1)[-1].lower()
        if tail in _ABBREVIATIONS:
            continue
        sentences.append(buffer)
        buffer = ""
    if buffer:
        sentences.append(buffer)

    return [s for s in sentences if len(s.split()) >= 4]  # drop stray fragments


def chunk_abstract(abstract: str, window: int = 2) -> list[str]:
    """Group sentences into small overlapping windows (default: pairs) so each
    chunk carries enough context to be useful as clinical evidence, while still
    being a clean, complete unit rather than an arbitrary character slice."""
    sentences = split_into_sentences(abstract)
    if not sentences:
        return []
    chunks = []
    for i in range(0, len(sentences), window):
        chunks.append(" ".join(sentences[i:i + window]))
    return chunks


# --------------------------------------------------------------------------
# Disk cache (query -> results), TTL-based
# --------------------------------------------------------------------------

class EvidenceCache:
    def __init__(self, cache_dir: str = "knowledge_base/cache", ttl_days: int = 30):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_days * 86400
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, query: str) -> str:
        key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, query: str) -> Optional[list[dict]]:
        path = self._path(query)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if time.time() - payload["cached_at"] > self.ttl_seconds:
            return None
        return payload["results"]

    def set(self, query: str, results: list[dict]) -> None:
        path = self._path(query)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"cached_at": time.time(), "results": results}, f)


# --------------------------------------------------------------------------
# Source clients
# --------------------------------------------------------------------------

class PubMedClient:
    ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 8.0):
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 5) -> list[EvidenceChunk]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            search_resp = await client.get(self.ESEARCH, params=params)
            search_resp.raise_for_status()
            pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }
            if self.api_key:
                fetch_params["api_key"] = self.api_key

            fetch_resp = await client.get(self.EFETCH, params=fetch_params)
            fetch_resp.raise_for_status()
            return self._parse(fetch_resp.text)

    def _parse(self, xml_text: str) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else None
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled"
            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text if journal_el is not None else None
            year_el = article.find(".//PubDate/Year")
            year = int(year_el.text) if year_el is not None and year_el.text.isdigit() else None

            abstract_parts = [
                "".join(el.itertext()) for el in article.findall(".//AbstractText")
            ]
            abstract = " ".join(abstract_parts).strip()
            if not abstract or not pmid:
                continue

            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            for sentence_chunk in chunk_abstract(abstract):
                chunks.append(EvidenceChunk(
                    text=sentence_chunk,
                    source_url=url,
                    title=title,
                    source_type="pubmed",
                    journal=journal,
                    year=year,
                    pmid=pmid,
                ))
        return chunks


class EuropePMCClient:
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 5) -> list[EvidenceChunk]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            params = {
                "query": f"{query} AND (OPEN_ACCESS:Y)",
                "format": "json",
                "pageSize": max_results,
                "resultType": "core",
            }
            resp = await client.get(self.BASE, params=params)
            resp.raise_for_status()
            results = resp.json().get("resultList", {}).get("result", [])

        chunks: list[EvidenceChunk] = []
        for r in results:
            abstract = r.get("abstractText", "")
            if not abstract:
                continue
            title = r.get("title", "Untitled")
            journal = r.get("journalInfo", {}).get("journal", {}).get("title")
            year = r.get("pubYear")
            year = int(year) if year and str(year).isdigit() else None
            source = r.get("source", "MED")
            ext_id = r.get("id", "")
            url = f"https://europepmc.org/article/{source}/{ext_id}"

            for sentence_chunk in chunk_abstract(abstract):
                chunks.append(EvidenceChunk(
                    text=sentence_chunk,
                    source_url=url,
                    title=title,
                    source_type="europepmc",
                    journal=journal,
                    year=year,
                    pmid=r.get("pmid"),
                ))
        return chunks


class ClinicalTrialsClient:
    BASE = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout

    @staticmethod
    def _sanitize_query(query: str, max_words: int = 8) -> str:
        """ClinicalTrials.gov's query.term uses Essie syntax, where colons,
        brackets, and other punctuation are parsed as field-query operators
        (e.g. AREA[Field]:value). A prose sentence like '...presenting
        with: X. Predicted severity level: Y...' gets misparsed by Essie
        and the API returns 400. Strip punctuation and keep it short."""
        stripped = re.sub(r"[^\w\s]", " ", query)
        words = stripped.split()
        return " ".join(words[:max_words])

    async def search(self, query: str, max_results: int = 5) -> list[EvidenceChunk]:
        term = self._sanitize_query(query)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            params = {
                "query.term": term,
                "pageSize": max_results,
                "fields": "NCTId,BriefTitle,BriefSummary,OverallStatus",
            }
            resp = await client.get(self.BASE, params=params)
            resp.raise_for_status()
            studies = resp.json().get("studies", [])

        chunks: list[EvidenceChunk] = []
        for s in studies:
            protocol = s.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            nct_id = ident.get("nctId")
            title = ident.get("briefTitle", "Untitled trial")
            summary = protocol.get("descriptionModule", {}).get("briefSummary", "")
            if not summary or not nct_id:
                continue
            url = f"https://clinicaltrials.gov/study/{nct_id}"

            for sentence_chunk in chunk_abstract(summary):
                chunks.append(EvidenceChunk(
                    text=sentence_chunk,
                    source_url=url,
                    title=title,
                    source_type="ctgov",
                ))
        return chunks


# --------------------------------------------------------------------------
# Federated retriever
# --------------------------------------------------------------------------

class WebEvidenceRetriever:
    """Queries PubMed, Europe PMC, and ClinicalTrials.gov concurrently, with
    disk caching (so a Streamlit demo doesn't re-hit live APIs every rerun)
    and graceful degradation if one source fails or times out."""

    def __init__(
        self,
        cache_dir: str = "knowledge_base/cache",
        ttl_days: int = 30,
        embedder=None,  # optional: pass your existing SentenceTransformer for reranking
    ):
        self.cache = EvidenceCache(cache_dir, ttl_days)
        self.pubmed = PubMedClient()
        self.europepmc = EuropePMCClient()
        self.ctgov = ClinicalTrialsClient()
        self.embedder = embedder

    async def retrieve(self, query: str, top_k: int = 5) -> list[EvidenceChunk]:
        cached = self.cache.get(query)
        if cached is not None:
            return [EvidenceChunk(**c) for c in cached]
        
        results = await asyncio.gather(
            self.pubmed.search(query, max_results=5),
            self.europepmc.search(query, max_results=5),
            self.ctgov.search(query, max_results=3),
            return_exceptions=True,
        )

        all_chunks: list[EvidenceChunk] = []
        for r in results:
            if isinstance(r, Exception):
                # One source failing (timeout, rate limit, schema change) must
                # not take down retrieval -- log and move on.
                print(f"[WebEvidenceRetriever] source failed: {r!r}")
                continue
            all_chunks.extend(r)

        ranked = self._rank(query, all_chunks)[:top_k]
        self.cache.set(query, [c.to_dict() for c in ranked])
        return ranked

    def _rank(self, query: str, chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
        if not chunks:
            return []

        if self.embedder is not None:
            import numpy as np
            texts = [c.text for c in chunks]
            query_vec = self.embedder.encode([query], normalize_embeddings=True)
            chunk_vecs = self.embedder.encode(texts, normalize_embeddings=True)
            scores = (chunk_vecs @ query_vec[0])
            for c, s in zip(chunks, scores):
                c.relevance_score = float(s)
        else:
            # Fallback: simple lexical overlap so relevance_score is still meaningful
            # for the UI badge even without loading the embedding model here.
            query_terms = set(query.lower().split())
            for c in chunks:
                chunk_terms = set(c.text.lower().split())
                overlap = len(query_terms & chunk_terms)
                c.relevance_score = overlap / max(len(query_terms), 1)

        return sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
