#!/usr/bin/env python3
"""Unified LLM call layer (anonymized for submission).

A thin OpenAI-compatible chat client used by the optional LLM-as-judge
analyses (extraction / recall / policy-following / policy-alignment).

Endpoint configuration is read from the environment so that no provider URL
or credential is hard-coded:

    OPENAI_BASE_URL   OpenAI-compatible chat-completions base (".../v1")
    OPENAI_API_KEY    bearer token for that endpoint

All chat traffic in the paper used a single answering/judging model,
referred to throughout as ``deepseek-v4-flash`` (see the paper for details).
The zero-cost figure/table scripts do NOT need this module; only the
re-judging utilities (judge_memory.py, a3_policy_following.py,
a4_policy_alignment.py) call it, and they require a configured endpoint.
"""

import os
import re
import json
import time
import requests
import threading
import concurrent.futures

# Global concurrency cap: no matter how many threads/batches, the number of
# in-flight HTTP requests never exceeds this value.
MAX_CONCURRENCY = 128
_global_sem = threading.BoundedSemaphore(MAX_CONCURRENCY)

# OpenAI-compatible endpoint. Set these in your environment before running any
# of the LLM-judge scripts. No default provider is assumed.
BASE_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
if BASE_URL and not BASE_URL.endswith("/chat/completions"):
    BASE_URL = BASE_URL + "/chat/completions"
API_KEY = os.environ.get("OPENAI_API_KEY", "")

_HEADERS = {
    "Content-Type": "application/json",
}
if API_KEY:
    _HEADERS["Authorization"] = f"Bearer {API_KEY}"


class LLMError(Exception):
    """Model call failed (after retries exhausted)."""
    pass


def chat(model: str, messages: list, temperature: float = 0.7,
         max_tokens: int = 2000, timeout: int = 90, max_retry: int = 3,
         **_ignored):
    """Call any OpenAI-compatible model and return the reply text (content)."""
    if not BASE_URL:
        raise LLMError(
            "OPENAI_BASE_URL is not set; configure an OpenAI-compatible "
            "endpoint (and OPENAI_API_KEY) before running LLM-judge scripts."
        )
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(max_retry):
        try:
            # Global semaphore: keep in-flight requests <= MAX_CONCURRENCY.
            with _global_sem:
                resp = requests.post(BASE_URL, headers=_HEADERS, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            if content.strip():
                return content
            # Empty content: try to parse JSON out of any reasoning trace.
            if reasoning.strip():
                fallback = parse_json(reasoning)
                if fallback is not None:
                    return json.dumps(fallback, ensure_ascii=False)
            last_err = f"empty content (attempt {attempt+1})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(min(2 ** attempt, 30))
    raise LLMError(f"model={model} failed after {max_retry} retries: {last_err}")


def chat_json(model: str, messages: list, **kwargs):
    """Call a model and parse its JSON reply. Returns (parsed, raw_text)."""
    raw = chat(model, messages, **kwargs)
    parsed = parse_json(raw)
    if parsed is None:
        raise ValueError(f"could not parse JSON from model output. raw: {raw[:500]}")
    return parsed, raw


# ---- concurrent batch calls ----

def _normalize_request(item, common_kwargs):
    """Normalize one batch item into chat() kwargs.

    Each item may be: a dict (used as chat() kwargs, must contain ``model``
    and ``messages``); a (model, messages) tuple; or a ``messages`` list (then
    ``model`` must be supplied via common_kwargs).
    """
    if isinstance(item, dict):
        kw = {**common_kwargs, **item}
    elif isinstance(item, tuple) and len(item) == 2:
        kw = {**common_kwargs, "model": item[0], "messages": item[1]}
    elif isinstance(item, list):
        kw = {**common_kwargs, "messages": item}
    else:
        raise TypeError(
            "batch item must be dict / (model, messages) tuple / messages list, "
            f"got: {type(item).__name__}"
        )
    if not kw.get("model"):
        raise ValueError("batch request missing model (set it via common_kwargs)")
    if not kw.get("messages"):
        raise ValueError("batch request missing messages")
    return kw


def _run_batch(fn, requests_list, max_workers, return_exceptions, common_kwargs):
    """Generic concurrent executor; preserves input order."""
    n = len(requests_list)
    if n == 0:
        return []
    workers = max(1, min(max_workers, MAX_CONCURRENCY, n))
    results = [None] * n
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_idx = {
            ex.submit(fn, **_normalize_request(item, common_kwargs)): i
            for i, item in enumerate(requests_list)
        }
        for fut in concurrent.futures.as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                if return_exceptions:
                    results[i] = e
                else:
                    raise
    return results


def chat_many(requests_list, *, max_workers: int = MAX_CONCURRENCY,
              return_exceptions: bool = True, **common_kwargs):
    """Concurrent :func:`chat`, up to MAX_CONCURRENCY in flight."""
    return _run_batch(chat, requests_list, max_workers, return_exceptions, common_kwargs)


def chat_json_many(requests_list, *, max_workers: int = MAX_CONCURRENCY,
                   return_exceptions: bool = True, **common_kwargs):
    """Concurrent :func:`chat_json`; returns (parsed, raw) per item."""
    return _run_batch(chat_json, requests_list, max_workers, return_exceptions, common_kwargs)


def parse_json(text: str):
    """Robustly extract a JSON object/array from model output."""
    if not text:
        return None
    t = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences.
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in t:
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1].strip()

    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = t.find(open_c)
        end = t.rfind(close_c)
        if start >= 0 and end > start:
            candidate = t[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    try:
        return json.loads(t)
    except Exception:
        pass

    cleaned = t.replace("`", "")
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_c)
        end = cleaned.rfind(close_c)
        if start >= 0 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

    result = _regex_extract_fields(text)
    if result:
        return result
    return None


def _regex_extract_fields(text):
    """Extract known fields from possibly-broken JSON via regex."""
    known_keys = ["question", "answer", "verification", "verdict", "reason", "reasoning"]
    result = {}
    for key in known_keys:
        pat = r'"' + key + r'"\s*:\s*"((?:[^"\\]|\\.)*)"'
        m = re.search(pat, text, re.DOTALL)
        if m:
            val = m.group(1)
            val = val.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            result[key] = val
    if any(k in result for k in ("question", "answer", "verdict")):
        return result
    return None


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-flash"
    out = chat(m, [{"role": "user", "content": "Reply with one word: OK"}],
               temperature=0, max_tokens=50)
    print(f"[{m}] => {out!r}")
