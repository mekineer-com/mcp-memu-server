#!/usr/bin/env python3
"""Compare stored OpenAI memory retrieval with Gemini on a disposable soul DB copy."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import sqlite3
import struct
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import numpy as np

ROOT = Path(__file__).resolve().parent
LIVE_DB_DIR = (ROOT.parent / "memu/sqlite").resolve()
CONFIG = ROOT / "config.json"
QUERY_COUNT = 200
DIMENSIONS = 3072
T = TypeVar("T")


def post(url: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def chunks(values: list[T], size: int) -> Iterator[list[T]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def require_disposable_db(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.parent == LIVE_DB_DIR or LIVE_DB_DIR in resolved.parents:
        raise ValueError(f"Refusing live memU database path: {resolved}")
    return resolved


def unpack_embedding(blob: bytes, item_id: str) -> list[float]:
    if len(blob) != DIMENSIONS * 4:
        raise ValueError(
            f"MemoryItem {item_id} has {len(blob) // 4} dimensions, expected {DIMENSIONS}"
        )
    values = list(struct.unpack(f"{DIMENSIONS}f", blob))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"MemoryItem {item_id} has a non-finite embedding")
    return values


def load_memories(path: Path) -> tuple[list[dict], int]:
    uri = f"file:{path}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as con:
        rows = con.execute(
            """
SELECT id, memory_type, summary, embedding
FROM memory_items
WHERE embedding IS NOT NULL
  AND TRIM(summary) <> ''
  AND (confidence IS NULL OR confidence >= 0.6)
  AND (merged_into IS NULL OR TRIM(merged_into) = '')
  AND NOT EXISTS (
    SELECT 1 FROM triples t
    WHERE t.subject_id = memory_items.id
      AND t.predicate = 'evolved_into'
      AND t.valid_to IS NULL
  )
ORDER BY id
"""
        ).fetchall()
        excluded_low_confidence = con.execute(
            """
SELECT COUNT(*) FROM memory_items
WHERE embedding IS NOT NULL
  AND TRIM(summary) <> ''
  AND confidence < 0.6
  AND (merged_into IS NULL OR TRIM(merged_into) = '')
  AND NOT EXISTS (
    SELECT 1 FROM triples t
    WHERE t.subject_id = memory_items.id
      AND t.predicate = 'evolved_into'
      AND t.valid_to IS NULL
  )
"""
        ).fetchone()[0]
    if len(rows) < QUERY_COUNT:
        raise ValueError(
            f"Need at least {QUERY_COUNT} active embedded MemoryItems, found {len(rows)}"
        )
    memories = [
        {
            "id": str(item_id),
            "memory_type": str(memory_type),
            "summary": str(summary),
            "embedding": unpack_embedding(blob, str(item_id)),
        }
        for item_id, memory_type, summary, blob in rows
    ]
    return memories, int(excluded_low_confidence)


def corpus_hash(memories: list[dict]) -> str:
    digest = hashlib.sha256()
    for memory in memories:
        digest.update(memory["id"].encode())
        digest.update(b"\0")
        digest.update(memory["memory_type"].encode())
        digest.update(b"\0")
        digest.update(memory["summary"].encode())
        digest.update(b"\0")
        digest.update(struct.pack(f"{DIMENSIONS}f", *memory["embedding"]))
    return digest.hexdigest()


def selected_targets(memories: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for memory in memories:
        unique.setdefault(" ".join(memory["summary"].casefold().split()), memory)
    ordered = sorted(
        unique.values(), key=lambda row: hashlib.sha256(row["id"].encode()).digest()
    )
    if len(ordered) < QUERY_COUNT:
        raise ValueError(
            f"Need at least {QUERY_COUNT} unique MemoryItem summaries, found {len(ordered)}"
        )
    return ordered[:QUERY_COUNT]


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def four_grams(text: str) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)}


def parse_json_object(text: str) -> dict[str, str]:
    value = text.strip()
    if value.startswith("```json\n") and value.endswith("\n```"):
        value = value[8:-4]
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("expected one JSON object")
    return parsed


def generate_queries(targets: list[dict], *, retry: bool = False) -> dict[str, str]:
    queries: dict[str, str] = {}
    for batch in chunks(targets, 25):
        prompt_rows = [{"id": row["id"], "memory": row["summary"]} for row in batch]
        prompt = (
            "For each memory, write one concise standalone semantic-retrieval query, as if conversational "
            "framing had already been rewritten for vector search. Paraphrase indirectly and do not copy "
            "any sequence of four words. Return only a JSON object mapping each id to its query. Do not use "
            "tools."
            + (
                " These are retries rejected by a mechanical four-word overlap check; use substantially "
                "different wording."
                if retry
                else ""
            )
            + "\n"
            + json.dumps(prompt_rows)
        )
        completed = subprocess.run(
            ["claude-glm", "-p", "--output-format", "text"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            input=prompt,
            text=True,
            timeout=1800,
        )
        if completed.returncode:
            raise RuntimeError(
                f"GLM query batch failed with exit code {completed.returncode}"
            )
        try:
            queries.update(parse_json_object(completed.stdout))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("GLM query batch did not return one JSON object") from exc
    return queries


def freeze_queries(db_path: Path, query_path: Path) -> None:
    if query_path.exists():
        raise FileExistsError(f"Refusing to replace frozen query file: {query_path}")
    memories, excluded_low_confidence = load_memories(require_disposable_db(db_path))
    targets = selected_targets(memories)
    queries = generate_queries(targets)
    expected = {row["id"] for row in targets}
    by_id = {row["id"]: row for row in targets}
    queries = {
        item_id: value.strip()
        for item_id, value in queries.items()
        if item_id in expected and isinstance(value, str) and value.strip()
    }
    for _attempt in range(3):
        missing = expected - set(queries)
        if not missing:
            break
        queries.update(
            generate_queries(
                [by_id[item_id] for item_id in sorted(missing)], retry=True
            )
        )
        queries = {
            item_id: value.strip()
            for item_id, value in queries.items()
            if item_id in expected and isinstance(value, str) and value.strip()
        }
    missing = expected - set(queries)
    if missing:
        raise ValueError(f"GLM omitted {len(missing)} query IDs after retries")
    leaks = [
        item_id
        for item_id, row in by_id.items()
        if four_grams(row["summary"]) & four_grams(queries[item_id])
    ]
    for _attempt in range(3):
        if not leaks:
            break
        queries.update(
            generate_queries([by_id[item_id] for item_id in leaks], retry=True)
        )
        leaks = [
            item_id
            for item_id in leaks
            if four_grams(by_id[item_id]["summary"]) & four_grams(queries[item_id])
        ]
    if leaks:
        raise ValueError(
            f"GLM queries copied a four-word sequence for target IDs: {', '.join(leaks)}"
        )
    payload = {
        "generator": "glm-5.3",
        "query_contract": "standalone retrieval-ready query",
        "corpus_sha256": corpus_hash(memories),
        "excluded_low_confidence": excluded_low_confidence,
        "queries": [
            {"target_id": row["id"], "query": queries[row["id"]]} for row in targets
        ],
    }
    query_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(query_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(
        f"Frozen {len(targets)} validated queries at {query_path}; run score separately."
    )


def openai_embed(config: dict, texts: list[str]) -> tuple[np.ndarray, float]:
    if config["embed_model"] != "text-embedding-3-large":
        raise ValueError(
            f"Expected text-embedding-3-large query model, got {config['embed_model']}"
        )
    vectors: list[list[float]] = []
    started = time.monotonic()
    for batch in chunks(texts, 64):
        data = post(
            f"{config['base_url'].rstrip('/')}/embeddings",
            {"model": config["embed_model"], "input": batch},
            {"Authorization": f"Bearer {config['api_key']}"},
        )
        vectors.extend(
            row["embedding"]
            for row in sorted(data["data"], key=lambda row: row["index"])
        )
    return checked_vectors(vectors, len(texts), "OpenAI"), time.monotonic() - started


def gemini_api_key() -> str:
    key = str(
        json.loads(CONFIG.read_text()).get("mentra", {}).get("gemini_api_key", "")
    ).strip()
    if not key:
        raise ValueError("config.json mentra.gemini_api_key is required")
    return key


def gemini_embed(
    texts: list[str], cache_path: Path | None = None
) -> tuple[np.ndarray, float]:
    api_key = gemini_api_key()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-embedding-2:batchEmbedContents?key={api_key}"
    )
    vectors: list[list[float]] = []
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if (
            cached.ndim != 2
            or cached.shape[1] != DIMENSIONS
            or len(cached) > len(texts)
        ):
            raise ValueError(f"Malformed Gemini checkpoint: {cache_path}")
        vectors = cached.tolist()
    started = time.monotonic()
    for batch in chunks(texts[len(vectors) :], 64):
        payload = {
            "requests": [
                {
                    "model": "models/gemini-embedding-2",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": DIMENSIONS,
                }
                for text in batch
            ]
        }
        for attempt in range(6):
            try:
                data = post(url, payload)
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == 5:
                    raise
                time.sleep(60)
        vectors.extend(row["values"] for row in data["embeddings"])
        if cache_path is not None:
            np.save(cache_path, np.asarray(vectors, dtype=np.float32))
        time.sleep(0.1)
    return checked_vectors(vectors, len(texts), "Gemini"), time.monotonic() - started


def gemini_embed_media(paths: list[Path]) -> tuple[np.ndarray, float]:
    api_key = gemini_api_key()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-embedding-2:embedContent?key={api_key}"
    )
    vectors: list[list[float]] = []
    started = time.monotonic()
    for path in paths:
        mime_type = mimetypes.guess_type(path.name)[0]
        if mime_type not in {"image/jpeg", "image/png"}:
            raise ValueError(f"Mixed-pool fixture must be JPEG or PNG: {path}")
        data = post(
            url,
            {
                "model": "models/gemini-embedding-2",
                "content": {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                            }
                        }
                    ]
                },
                "outputDimensionality": DIMENSIONS,
            },
        )
        vectors.append(data["embedding"]["values"])
    return checked_vectors(vectors, len(paths), "Gemini media"), time.monotonic() - started


def checked_vectors(
    vectors: list[list[float]], expected: int, provider: str
) -> np.ndarray:
    if len(vectors) != expected:
        raise ValueError(
            f"{provider} returned {len(vectors)} vectors for {expected} inputs"
        )
    result = np.asarray(vectors, dtype=np.float32)
    if result.shape != (expected, DIMENSIONS) or not np.isfinite(result).all():
        raise ValueError(
            f"{provider} returned malformed embeddings with shape {result.shape}"
        )
    return result


def ranks(
    documents: np.ndarray, queries: np.ndarray, targets: list[set[int]]
) -> np.ndarray:
    documents = documents / np.linalg.norm(documents, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    order = np.argsort(-(queries @ documents.T), axis=1)
    return np.asarray(
        [
            min(
                int(np.where(order[index] == target)[0][0]) + 1 for target in acceptable
            )
            for index, acceptable in enumerate(targets)
        ]
    )


def exact_sign_pvalue(better: int, worse: int) -> float:
    trials = better + worse
    if not trials:
        return 1.0
    tail = sum(math.comb(trials, value) for value in range(min(better, worse) + 1)) / (
        2**trials
    )
    return min(1.0, 2 * tail)


def rank_metrics(values: np.ndarray) -> dict:
    return {
        "recall_at_1": round(float(np.mean(values <= 1)), 6),
        "recall_at_5": round(float(np.mean(values <= 5)), 6),
        "recall_at_10": round(float(np.mean(values <= 10)), 6),
        "mean_reciprocal_rank": round(float(np.mean(1 / values)), 6),
        "median_rank": int(np.median(values)),
        "maximum_rank": int(np.max(values)),
    }


def mixed_pool_metrics(
    text_documents: np.ndarray,
    text_queries: np.ndarray,
    text_targets: list[set[int]],
    media_documents: np.ndarray,
    media_queries: np.ndarray,
) -> dict:
    combined = np.vstack((text_documents, media_documents))
    text_before = ranks(text_documents, text_queries, text_targets)
    text_after = ranks(combined, text_queries, text_targets)
    offset = len(text_documents)
    media_ranks = ranks(
        combined,
        media_queries,
        [{offset + index} for index in range(len(media_documents))],
    )
    return {
        "text_before_mixed_pool": rank_metrics(text_before),
        "text_after_mixed_pool": rank_metrics(text_after),
        "text_queries_displaced": int(np.sum(text_after > text_before)),
        "media": rank_metrics(media_ranks),
    }


def resource_smoke_metrics(
    media_documents: np.ndarray,
    caption_documents: np.ndarray,
    query_vectors: np.ndarray,
    resource_ids: list[str],
    query_rows: list[dict],
) -> dict:
    def scores(documents: np.ndarray) -> np.ndarray:
        docs = documents / np.linalg.norm(documents, axis=1, keepdims=True)
        queries = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        return queries @ docs.T

    def report(matrix: np.ndarray) -> dict:
        matching: list[float] = []
        distractors: list[float] = []
        target_ranks: list[int] = []
        ambiguous_max: list[float] = []
        unrelated_max: list[float] = []
        by_id = {resource_id: index for index, resource_id in enumerate(resource_ids)}
        for row, row_scores in zip(query_rows, matrix, strict=True):
            group = row["group"]
            if group == "matching":
                target = by_id[row["target"]]
                matching.append(float(row_scores[target]))
                target_ranks.append(int(np.sum(row_scores > row_scores[target])) + 1)
                distractors.extend(float(score) for index, score in enumerate(row_scores) if index != target)
            elif group == "ambiguous":
                ambiguous_max.append(float(np.max(row_scores)))
            elif group == "unrelated":
                unrelated_max.append(float(np.max(row_scores)))
            else:
                raise ValueError(f"Unknown Resource smoke query group: {group}")
        return {
            "matching_targets": distribution(np.asarray(matching)),
            "matching_target_ranks": rank_metrics(np.asarray(target_ranks)),
            "matching_distractors": distribution(np.asarray(distractors)),
            "ambiguous_query_maxima": distribution(np.asarray(ambiguous_max)),
            "unrelated_query_maxima": distribution(np.asarray(unrelated_max)),
        }

    return {
        "resources": len(resource_ids),
        "queries": len(query_rows),
        "media": report(scores(media_documents)),
        "caption": report(scores(caption_documents)),
    }


def neighborhood_scores(
    vectors: np.ndarray, summaries: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = normalized @ normalized.T
    labels = np.asarray([" ".join(summary.casefold().split()) for summary in summaries])
    same = labels[:, None] == labels[None, :]
    exact_duplicates = scores[np.triu(same, k=1)]
    scores[same] = -np.inf
    return exact_duplicates, np.max(scores, axis=1)


def distribution(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "minimum": round(float(np.min(finite)), 6),
        **{
            f"p{percentile}": round(float(np.percentile(finite, percentile)), 6)
            for percentile in (1, 5, 50, 95, 99)
        },
        "maximum": round(float(np.max(finite)), 6),
    }


def calibration_report(
    old_vectors: np.ndarray,
    new_vectors: np.ndarray,
    summaries: list[str],
    old_threshold: float,
) -> dict:
    old_duplicates, old_neighbors = neighborhood_scores(old_vectors, summaries)
    new_duplicates, new_neighbors = neighborhood_scores(new_vectors, summaries)
    exceedance = float(np.mean(old_neighbors >= old_threshold))
    mapped = float(np.quantile(new_neighbors, 1.0 - exceedance))
    return {
        "old_threshold": old_threshold,
        "old_hard_neighbor_exceedance": round(exceedance, 6),
        "candidate_same_exceedance_threshold": round(mapped, 6),
        "openai": {
            "exact_duplicates": distribution(old_duplicates),
            "hardest_nonduplicates": distribution(old_neighbors),
        },
        "gemini": {
            "exact_duplicates": distribution(new_duplicates),
            "hardest_nonduplicates": distribution(new_neighbors),
        },
    }


def score(db_path: Path, query_path: Path, *, runtime_contract: bool = False) -> None:
    memories, excluded_low_confidence, frozen, target_indices = _frozen_inputs(
        db_path, query_path
    )
    query_rows = frozen["queries"]
    query_texts = [row["query"] for row in query_rows]
    config = json.loads(CONFIG.read_text())["llm"]

    openai_queries, openai_seconds = openai_embed(config, query_texts)
    openai_documents = np.asarray(
        [row["embedding"] for row in memories], dtype=np.float32
    )
    if runtime_contract:
        document_texts = [row["summary"] for row in memories]
        gemini_query_texts = query_texts
        document_checkpoint = query_path.with_name("gemini-documents-raw.npy")
        query_checkpoint = query_path.with_name("gemini-queries-raw.npy")
    else:
        document_texts = [f"title: {row['memory_type']} | text: {row['summary']}" for row in memories]
        gemini_query_texts = [f"task: search result | query: {query}" for query in query_texts]
        document_checkpoint = query_path.with_name("gemini-documents.npy")
        query_checkpoint = query_path.with_name("gemini-queries.npy")
    gemini_documents, gemini_doc_seconds = gemini_embed(document_texts, document_checkpoint)
    gemini_queries, gemini_query_seconds = gemini_embed(gemini_query_texts, query_checkpoint)
    old_ranks = ranks(openai_documents, openai_queries, target_indices)
    new_ranks = ranks(gemini_documents, gemini_queries, target_indices)
    gemini_better = int(np.sum(new_ranks < old_ranks))
    openai_better = int(np.sum(new_ranks > old_ranks))
    old_metrics = rank_metrics(old_ranks)
    new_metrics = rank_metrics(new_ranks)
    recall_gap = new_metrics["recall_at_5"] - old_metrics["recall_at_5"]
    sign_pvalue = exact_sign_pvalue(gemini_better, openai_better)
    passed = recall_gap >= -0.02 and not (
        openai_better > gemini_better and sign_pvalue < 0.05
    )
    result = {
        "documents": len(memories),
        "excluded_low_confidence": excluded_low_confidence,
        "queries": len(query_rows),
        "dimensions": DIMENSIONS,
        "gemini_text_contract": "bare runtime text" if runtime_contract else "prefixed diagnostic text",
        "baseline": {
            "model": "text-embedding-3-large",
            "documents": "stored production vectors",
            **old_metrics,
            "query_seconds": round(openai_seconds, 3),
        },
        "candidate": {
            "model": "gemini-embedding-2",
            **new_metrics,
            "document_seconds": round(gemini_doc_seconds, 3),
            "query_seconds": round(gemini_query_seconds, 3),
        },
        "paired": {
            "gemini_better": gemini_better,
            "equal": int(np.sum(new_ranks == old_ranks)),
            "openai_better": openai_better,
            "exact_sign_pvalue": round(sign_pvalue, 6),
            "mean_rank_delta_gemini_minus_openai": round(
                float(np.mean(new_ranks - old_ranks)), 6
            ),
            "median_rank_delta_gemini_minus_openai": round(
                float(np.median(new_ranks - old_ranks)), 6
            ),
        },
        "gate": {
            "maximum_recall_at_5_drop": 0.02,
            "recall_at_5_delta": round(recall_gap, 6),
            "passed": passed,
        },
    }
    print(json.dumps(result, indent=2))


def _frozen_inputs(
    db_path: Path, query_path: Path
) -> tuple[list[dict], int, dict, list[set[int]]]:
    memories, excluded = load_memories(require_disposable_db(db_path))
    frozen = json.loads(query_path.read_text())
    if frozen.get("corpus_sha256") != corpus_hash(memories):
        raise ValueError("Frozen queries do not belong to this disposable database copy")
    by_id = {row["id"]: index for index, row in enumerate(memories)}
    equivalent: dict[str, set[int]] = {}
    for index, row in enumerate(memories):
        equivalent.setdefault(" ".join(row["summary"].casefold().split()), set()).add(index)
    targets = [
        equivalent[
            " ".join(memories[by_id[row["target_id"]]]["summary"].casefold().split())
        ]
        for row in frozen["queries"]
    ]
    return memories, excluded, frozen, targets


def score_mixed(db_path: Path, query_path: Path, manifest_path: Path) -> None:
    memories, _excluded, frozen, targets = _frozen_inputs(db_path, query_path)
    text_documents = checked_vectors(
        np.load(query_path.with_name("gemini-documents.npy")).tolist(),
        len(memories),
        "Cached Gemini documents",
    )
    text_queries = checked_vectors(
        np.load(query_path.with_name("gemini-queries.npy")).tolist(),
        len(frozen["queries"]),
        "Cached Gemini queries",
    )
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("images")
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("Mixed-pool manifest requires at least three fictional images")
    paths = [(manifest_path.parent / str(row["path"])).resolve() for row in rows]
    queries = [str(row["query"]).strip() for row in rows]
    if any(not path.is_file() for path in paths) or any(not query for query in queries):
        raise ValueError("Mixed-pool image paths and queries must be complete")
    media_documents, media_seconds = gemini_embed_media(paths)
    media_queries, query_seconds = gemini_embed(
        [f"task: search result | query: {query}" for query in queries]
    )
    result = mixed_pool_metrics(
        text_documents,
        text_queries,
        targets,
        media_documents,
        media_queries,
    )
    result.update(
        {
            "text_documents": len(memories),
            "media_documents": len(rows),
            "media_embedding_seconds": round(media_seconds, 3),
            "media_query_seconds": round(query_seconds, 3),
        }
    )
    print(json.dumps(result, indent=2))


def score_calibration(db_path: Path, query_path: Path) -> None:
    memories, _excluded, _frozen, _targets = _frozen_inputs(db_path, query_path)
    old_vectors = np.asarray([row["embedding"] for row in memories], dtype=np.float32)
    new_vectors = checked_vectors(
        np.load(query_path.with_name("gemini-documents.npy")).tolist(),
        len(memories),
        "Cached Gemini documents",
    )
    config = json.loads(CONFIG.read_text())
    threshold = float(
        config.get("memorize", {}).get("semantic_dedupe_similarity_threshold", 0.85)
    )
    print(
        json.dumps(
            calibration_report(
                old_vectors,
                new_vectors,
                [row["summary"] for row in memories],
                threshold,
            ),
            indent=2,
        )
    )


def score_resource_smoke(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    resources = manifest.get("resources")
    query_rows = manifest.get("queries")
    if not isinstance(resources, list) or not isinstance(query_rows, list):
        raise ValueError("Resource smoke manifest requires resources and queries lists")
    resource_ids = [str(row["id"]).strip() for row in resources]
    paths = [(manifest_path.parent / str(row["path"])).resolve() for row in resources]
    captions = [str(row["caption"]).strip() for row in resources]
    query_texts = [str(row["text"]).strip() for row in query_rows]
    if (
        len(resource_ids) < 3
        or len(set(resource_ids)) != len(resource_ids)
        or any(not path.is_file() for path in paths)
        or any(not value for value in captions + query_texts)
    ):
        raise ValueError("Resource smoke manifest is incomplete")
    media_documents, media_seconds = gemini_embed_media(paths)
    caption_documents, caption_seconds = gemini_embed(captions)
    query_vectors, query_seconds = gemini_embed(query_texts)
    result = resource_smoke_metrics(
        media_documents,
        caption_documents,
        query_vectors,
        resource_ids,
        query_rows,
    )
    result.update({
        "model": "gemini-embedding-2",
        "dimensions": DIMENSIONS,
        "query_contract": "bare active_query",
        "media_embedding_seconds": round(media_seconds, 3),
        "caption_embedding_seconds": round(caption_seconds, 3),
        "query_embedding_seconds": round(query_seconds, 3),
        "purpose": "smoke evidence only; not production threshold calibration",
    })
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("freeze", "score", "runtime", "mixed", "calibrate", "resource-smoke")
    )
    parser.add_argument(
        "database",
        type=Path,
        nargs="?",
        help="Stopped disposable SQLite backup, never live Siri.db",
    )
    parser.add_argument(
        "queries", type=Path, nargs="?", help="Private frozen query JSON outside version control"
    )
    parser.add_argument(
        "--media-manifest",
        type=Path,
        help="Private fictional image/query manifest required by the mixed phase",
    )
    args = parser.parse_args()
    if args.phase == "resource-smoke":
        if args.media_manifest is None:
            parser.error("resource-smoke requires --media-manifest")
        score_resource_smoke(args.media_manifest)
        return
    if args.database is None or args.queries is None:
        parser.error(f"{args.phase} requires database and queries")
    if args.phase == "freeze":
        freeze_queries(args.database, args.queries)
    elif args.phase == "score":
        score(args.database, args.queries)
    elif args.phase == "runtime":
        score(args.database, args.queries, runtime_contract=True)
    elif args.phase == "calibrate":
        score_calibration(args.database, args.queries)
    else:
        if args.media_manifest is None:
            parser.error("mixed requires --media-manifest")
        score_mixed(args.database, args.queries, args.media_manifest)


if __name__ == "__main__":
    main()
