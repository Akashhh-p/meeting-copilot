from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

import chromadb
import numpy as np
import ollama
from chromadb.config import Settings

logger = logging.getLogger("meeting-copilot.rag")
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class TranscriptChunk:
    id: str
    start: float
    end: float
    speaker: str
    text: str
    document: str


class TranscriptRAG:
    def __init__(
        self,
        *,
        embed_model: str = "nomic-embed-text",
        llm_model: str = "llama3",
        collection_name: str = "meeting_transcript",
        chunk_seconds: float = 25.0,
        max_chars: int = 1800,
        top_k: int = 5,
        embed_timeout: float = 20.0,
        qa_timeout: float = 45.0,
        summary_timeout: float = 90.0,
        keep_alive: str = "10m",
    ):
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.chunk_seconds = chunk_seconds
        self.max_chars = max_chars
        self.top_k = top_k
        self.embed_timeout = embed_timeout
        self.qa_timeout = qa_timeout
        self.summary_timeout = summary_timeout
        self.keep_alive = keep_alive
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._chunks: dict[str, TranscriptChunk] = {}
        self._embedding_cache: dict[str, list[float]] = {}
        self._last_error = ""
        self._last_mode = "empty"
        self._client = chromadb.Client(Settings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._chunks.clear()
            self._last_error = ""
            self._last_mode = "empty"
            try:
                self._client.delete_collection(self._collection.name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                self._collection.name,
                metadata={"hnsw:space": "cosine"},
            )
        logger.info("RAG index cleared")

    def add_segment(self, segment: dict[str, Any]) -> None:
        text = str(segment.get("text", "")).strip()
        if not text:
            return
        with self._lock:
            self._entries.append(dict(segment))
        self.sync_index()

    def sync_index(self, entries: list[dict[str, Any]] | None = None) -> None:
        if entries is None:
            with self._lock:
                entries = list(self._entries)
        chunks = self._build_chunks(entries)
        new_chunks = [chunk for chunk in chunks if chunk.id not in self._chunks]
        if not new_chunks:
            return
        try:
            embeddings = self._embed([chunk.document for chunk in new_chunks])
            self._collection.upsert(
                ids=[chunk.id for chunk in new_chunks],
                documents=[chunk.document for chunk in new_chunks],
                embeddings=embeddings,
                metadatas=[
                    {
                        "timestamp_start": chunk.start,
                        "timestamp_end": chunk.end,
                        "speaker": chunk.speaker,
                        "text": chunk.text,
                    }
                    for chunk in new_chunks
                ],
            )
            with self._lock:
                for chunk in new_chunks:
                    self._chunks[chunk.id] = chunk
                self._last_mode = "semantic"
                self._last_error = ""
            logger.info("RAG indexed chunks=%d total=%d", len(new_chunks), len(self._chunks))
        except Exception as exc:
            self._last_error = str(exc)
            self._last_mode = "lexical_fallback"
            with self._lock:
                for chunk in new_chunks:
                    self._chunks[chunk.id] = chunk
            logger.exception("RAG semantic indexing failed; lexical fallback active")

    def retrieve(self, question: str, entries: list[dict[str, Any]] | None = None, top_k: int | None = None) -> list[TranscriptChunk]:
        if entries is not None:
            self.sync_index(entries)
        k = top_k or self.top_k
        with self._lock:
            chunks = list(self._chunks.values())
        if not chunks:
            return []

        semantic: list[TranscriptChunk] = []
        try:
            query_embedding = self._embed([question])[0]
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, max(1, len(chunks))),
                include=["metadatas", "documents"],
            )
            ids = result.get("ids", [[]])[0]
            chunk_map = {chunk.id: chunk for chunk in chunks}
            semantic = [chunk_map[item] for item in ids if item in chunk_map]
            self._last_mode = "semantic"
        except Exception as exc:
            self._last_error = str(exc)
            self._last_mode = "lexical_fallback"
            logger.warning("Semantic retrieval failed: %s", exc)

        lexical = self._lexical_retrieve(question, chunks, k)
        merged: list[TranscriptChunk] = []
        seen = set()
        for chunk in semantic + lexical:
            if chunk.id not in seen:
                merged.append(chunk)
                seen.add(chunk.id)
            if len(merged) >= k:
                break
        return sorted(merged, key=lambda chunk: chunk.start)

    def answer(self, question: str, entries: list[dict[str, Any]] | None = None) -> str:
        if not question.strip():
            return "Question:\n\nAnswer: This was not mentioned in the meeting.\nEvidence: None"
        chunks = self.retrieve(question, entries)
        if not chunks:
            return f"Question: {question}\nAnswer: This was not mentioned in the meeting.\nEvidence: None"
        context = "\n\n".join(chunk.document for chunk in chunks)
        prompt = f"""Answer the user question using ONLY the transcript evidence.
If the answer is not directly supported, write exactly: This was not mentioned in the meeting.

Output format:
Question: {question}
Answer: <answer>
Evidence: <speaker and timestamp range>

Transcript evidence:
{context}
"""
        try:
            response = ollama.Client(timeout=self.qa_timeout).chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": int(os.getenv("OLLAMA_QA_NUM_PREDICT", "180"))},
                keep_alive=self.keep_alive,
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            logger.exception("Q&A generation failed")
            evidence = "\n".join(chunk.document for chunk in chunks)
            return f"Question: {question}\nAnswer: This was not mentioned in the meeting.\nEvidence: {evidence}\nError: {exc}"

    def summarize(self, entries: list[dict[str, Any]] | None = None) -> str:
        if entries is not None:
            self.sync_index(entries)
        with self._lock:
            chunks = sorted(self._chunks.values(), key=lambda chunk: chunk.start)
        if not chunks:
            return "Meeting Summary:\n- Not mentioned.\n\nDecisions:\n- Not mentioned.\n\nAction Items:\n- Not mentioned.\n\nOpen Questions:\n- Not mentioned.\n\nNext Steps:\n- Not mentioned."
        selected = self._summary_chunks(chunks)
        context = "\n\n".join(chunk.document for chunk in selected)
        prompt = f"""Create a structured meeting summary using ONLY the transcript below.
Do not infer or invent. If a section has no evidence, write "Not mentioned."

Required format:
Meeting Summary:
- Key points

Decisions:
- Decision list

Action Items:
- Person: Task - Deadline

Open Questions:
- Questions

Next Steps:
- Steps

Transcript:
{context}
"""
        try:
            response = ollama.Client(timeout=self.summary_timeout).chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": int(os.getenv("OLLAMA_SUMMARY_NUM_PREDICT", "420"))},
                keep_alive=self.keep_alive,
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            logger.exception("Summary generation failed")
            return f"Meeting Summary:\n- Summary generation failed: {exc}\n\nDecisions:\n- Not mentioned.\n\nAction Items:\n- Not mentioned.\n\nOpen Questions:\n- Not mentioned.\n\nNext Steps:\n- Not mentioned."

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "segments": len(self._entries),
                "chunks": len(self._chunks),
                "embedding_model": self.embed_model,
                "mode": self._last_mode,
                "last_error": self._last_error,
                "vector_store": "chromadb",
            }

    def _build_chunks(self, entries: list[dict[str, Any]]) -> list[TranscriptChunk]:
        if not entries:
            return []
        chunks = []
        current: list[dict[str, Any]] = []
        current_start = float(entries[0].get("start", entries[0].get("time", 0.0)))
        for entry in entries:
            start = float(entry.get("start", entry.get("time", 0.0)))
            proposed = current + [entry]
            proposed_text = "\n".join(format_entry(item) for item in proposed)
            speaker_changed = current and entry.get("speaker") != current[-1].get("speaker")
            if current and (start - current_start >= self.chunk_seconds or len(proposed_text) > self.max_chars or speaker_changed):
                chunks.append(self._make_chunk(current))
                current = [entry]
                current_start = start
            else:
                current = proposed
        if current:
            chunks.append(self._make_chunk(current))
        return chunks

    def _make_chunk(self, entries: list[dict[str, Any]]) -> TranscriptChunk:
        start = float(entries[0].get("start", entries[0].get("time", 0.0)))
        end = float(entries[-1].get("end", entries[-1].get("time", start)))
        speakers = sorted({str(entry.get("speaker", "Speaker Unknown")) for entry in entries})
        text = " ".join(str(entry.get("text", "")).strip() for entry in entries).strip()
        document = "\n".join(format_entry(entry) for entry in entries)
        digest = hashlib.sha1(document.encode("utf-8")).hexdigest()[:16]
        return TranscriptChunk(
            id=f"{int(start * 1000)}-{int(end * 1000)}-{digest}",
            start=start,
            end=end,
            speaker=", ".join(speakers),
            text=text,
            document=document,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._embedding_cache]
        if missing:
            response = ollama.Client(timeout=self.embed_timeout).embed(
                model=self.embed_model,
                input=missing,
                keep_alive=self.keep_alive,
            )
            for text, embedding in zip(missing, response["embeddings"]):
                self._embedding_cache[text] = list(embedding)
        return [self._embedding_cache[text] for text in texts]

    def _lexical_retrieve(self, question: str, chunks: list[TranscriptChunk], top_k: int) -> list[TranscriptChunk]:
        query_terms = weighted_terms(question)
        if not query_terms:
            return chunks[-top_k:]
        scored = []
        for chunk in chunks:
            terms = weighted_terms(chunk.document)
            overlap = set(query_terms) & set(terms)
            numerator = sum(query_terms[item] * terms[item] for item in overlap)
            denominator = math.sqrt(sum(value * value for value in query_terms.values()))
            denominator *= math.sqrt(sum(value * value for value in terms.values()))
            score = numerator / denominator if denominator else 0.0
            scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].start))
        return [chunk for score, chunk in scored[:top_k] if score > 0] or chunks[-top_k:]

    def _summary_chunks(self, chunks: list[TranscriptChunk], limit: int = 28) -> list[TranscriptChunk]:
        if len(chunks) <= limit:
            return chunks
        head = chunks[:5]
        tail = chunks[-12:]
        middle = chunks[5:-12]
        slots = max(0, limit - len(head) - len(tail))
        stride = max(1, len(middle) // max(1, slots))
        return head + middle[::stride][:slots] + tail


def format_entry(entry: dict[str, Any]) -> str:
    start = float(entry.get("start", entry.get("time", 0.0)))
    end = float(entry.get("end", start))
    speaker = str(entry.get("speaker", "Speaker Unknown"))
    text = str(entry.get("text", "")).strip()
    return f"[{format_timestamp(start)} - {format_timestamp(end)}] {speaker}: {text}"


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def weighted_terms(text: str) -> dict[str, float]:
    terms: dict[str, float] = {}
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) <= 2 or token in STOP_WORDS:
            continue
        terms[token] = terms.get(token, 0.0) + 1.0
    return terms


STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "but",
    "can", "did", "does", "for", "from", "had", "has", "have", "how",
    "into", "meeting", "our", "said", "say", "that", "the", "their",
    "then", "there", "they", "this", "was", "what", "when", "where",
    "which", "who", "will", "with", "you",
}
