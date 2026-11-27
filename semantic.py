"""Finding things by meaning: an embedding model turns text into a vector, and texts
that mean similar things get similar vectors ("my exam is on Friday" ~ "when's my test?").

Used for:
- facts about sir: only the ones that matter for this message are given to the model,
  instead of every fact ever saved on every request;
- notes: search_notes finds a note by what it is about, not just by matching words.

The model (config.EMBED_MODEL, nomic-embed-text) runs locally in Ollama like the main model.
If it isn't installed or Ollama is down, everything falls back to the old behaviour
(all facts, keyword search) instead of failing.
"""
import hashlib
import math
import struct
from contextlib import closing

import config
import memory

# nomic-embed-text is trained with these task prefixes; without them matches are worse.
QUERY, DOCUMENT = "search_query: ", "search_document: "


class Unavailable(Exception):
    pass


def init_db() -> None:
    with closing(memory._connect()) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, model TEXT, vector BLOB)")


def _embed(texts: list[str]) -> list[list[float]]:
    try:
        from ollama import embed

        import gamemode

        return [list(v) for v in embed(model=config.EMBED_MODEL, input=texts,
                                       keep_alive=gamemode.keep_alive()).embeddings]
    except Exception as exc:  # model not pulled, Ollama down, ...
        raise Unavailable(f"{type(exc).__name__}: {exc}") from exc


def _key(text: str) -> str:
    return hashlib.sha256((config.EMBED_MODEL + "\0" + text).encode()).hexdigest()


def vectors(texts: list[str]) -> list[list[float]]:
    """Document vectors, cached in SQLite so each text is embedded once."""
    init_db()
    keys = [_key(DOCUMENT + t) for t in texts]
    with closing(memory._connect()) as db:
        cached = {k: v for k, v in db.execute(
            f"SELECT key, vector FROM embeddings WHERE key IN ({','.join('?' * len(keys))})", keys)} if keys else {}
    missing = [i for i, k in enumerate(keys) if k not in cached]
    if missing:
        fresh = _embed([DOCUMENT + texts[i] for i in missing])
        with closing(memory._connect()) as db, db:
            for i, vec in zip(missing, fresh):
                blob = struct.pack(f"{len(vec)}f", *vec)
                cached[keys[i]] = blob
                db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)", (keys[i], config.EMBED_MODEL, blob))
    return [list(struct.unpack(f"{len(cached[k]) // 4}f", cached[k])) for k in keys]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rank(query: str, texts: list[str]) -> list[tuple[float, int]]:
    """(similarity, index) for each text, best first."""
    if not texts:
        return []
    q = _embed([QUERY + query])[0]
    return sorted(((cosine(q, v), i) for i, v in enumerate(vectors(texts))), reverse=True)


def relevant_facts(message: str, facts: list[str]) -> list[str]:
    """The facts worth showing the model for this message, in the order they were saved: only
    the related ones, however few he knows. Every fact sent with every message got worked into
    small talk, and a fact just told is in the conversation already."""
    if not facts:
        return []
    try:
        ranked = rank(message, facts)
    except Unavailable:
        return facts  # no embedding model: the old way, all of them
    keep = {i for score, i in ranked[:config.MEMORY_TOP_K] if score >= config.MEMORY_MIN_SIMILARITY}
    return [f for i, f in enumerate(facts) if i in keep]
