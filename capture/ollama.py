"""The ollama HTTP API on localhost (capture spec section 4). urllib only."""

from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:11434"
PROMPT = "Write an essay of 150-250 words in English on the following task. Task: {prompt}"
SMOKE_PROMPT = "Write a short paragraph (about 80 words) about your favourite season."
TEMPERATURE = 0.7


def essay_prompt(task: str) -> str:
    return PROMPT.format(prompt=task)


def _get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


def _post(path: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def open_stream(body: dict, timeout: float = 600):
    req = urllib.request.Request(BASE + "/api/generate", json.dumps({**body, "stream": True}).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def version() -> str:
    return _get("/api/version")["version"]


def model_record(model: str) -> dict:
    """Name, digest and details. ollama 0.40's /api/show holds no digest, so it is taken from /api/tags (the
    manifest digest), with the GGUF blob named in the modelfile's FROM line beside it."""
    show = _post("/api/show", {"model": model}, timeout=60)
    tags = {m["name"]: m for m in _get("/api/tags")["models"]}
    t = tags.get(model) or tags.get(model + ":latest")
    if t is None:
        raise KeyError(f"{model} is not pulled")
    blob = next((ln.split()[1] for ln in show.get("modelfile", "").splitlines() if ln.startswith("FROM ")), None)
    return {"model": model, "digest": t["digest"], "size": t["size"], "details": show.get("details"),
            "from_blob": blob.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] if blob else None}


def ps() -> list[dict]:
    return [{"name": m["name"], "size": m.get("size"), "size_vram": m.get("size_vram")}
            for m in _get("/api/ps").get("models", [])]


def generate(body: dict, timeout: float = 600) -> dict:
    return _post("/api/generate", {**body, "stream": False}, timeout=timeout)


def unload(model: str) -> None:
    _post("/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
