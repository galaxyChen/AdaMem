#!/usr/bin/env python3
"""Shared helpers for LLM-judge based analyses (built on call_llm.py).

All judging goes through call_llm.chat_json_many (<=128 concurrent). The judge
model is deepseek-v4-flash via the OpenAI-compatible endpoint. Golden loading and the
extraction/recall row schema mirror analysis/run_paper_judge.py so the output
jsonls stay drop-in compatible with build_paper_report.py and the repro
scripts.
"""

import os
import re
import sys
import json

import _common as C

sys.path.insert(0, C.REPO_ROOT)
import call_llm  # noqa: E402

JUDGE_MODEL = "deepseek-v4-flash"

# Ground-truth character key -> aliases that may appear in learned policies.
GT_CHARACTERS = ["Boss Zhang", "Liwei", "Wanghao", "Sister Chen", "Mom", "Jiange"]


def normalize_char(name):
    """Map a policy by_character key onto a canonical ground-truth character."""
    if not name:
        return None
    low = name.strip().lower()
    if "mom" in low or "mother" in low:
        return "Mom"
    for gt in GT_CHARACTERS:
        if gt.lower() in low or low in gt.lower():
            return gt
    return name.strip()


# ---------- batched judge call ----------

def judge_many(prompts, *, max_tokens=8192, temperature=0.0, max_retry=3):
    """Run a list of single-user-message judge prompts; return list of parsed
    dicts (or None for failures), order-preserved."""
    reqs = [{"messages": [{"role": "user", "content": p}]} for p in prompts]
    results = call_llm.chat_json_many(
        reqs, model=JUDGE_MODEL, temperature=temperature,
        max_tokens=max_tokens, max_retry=max_retry, return_exceptions=True,
    )
    out = []
    for r in results:
        if isinstance(r, Exception) or r is None:
            out.append(None)
        else:
            out.append(r[0])  # (parsed, raw) -> parsed
    return out


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_loose(text):
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ---------- dataset golden loading (mirrors run_paper_judge.py) ----------

def load_session_goldens(story_id, num_weeks=10):
    sd = os.path.join(C.DATA_ROOT, f"story_{story_id}")
    session_index = {}
    for w in range(1, num_weeks + 1):
        path = os.path.join(sd, f"week{w}.json")
        if not os.path.exists(path):
            continue
        wd = json.load(open(path, encoding="utf-8"))
        for sess in wd.get("conversations", []) or []:
            sid = sess.get("session_id") or ""
            session_index[(w, sid)] = {
                "date": sess.get("date", ""),
                "focal_character": sess.get("focal_character", ""),
                "golden_memories": list(sess.get("golden_memories", []) or []),
            }
    qa_index = {}
    qa_path = os.path.join(sd, "test_qa.json")
    if os.path.exists(qa_path):
        qd = json.load(open(qa_path, encoding="utf-8"))
        for qa in qd.get("test_questions", []) or []:
            qid = qa.get("question_id")
            if qid:
                qa_index[qid] = {
                    "question": qa.get("question", ""),
                    "target_memory_refs": list(qa.get("target_memory_refs", []) or []),
                }
    return session_index, qa_index


def build_golden_pool(session_index):
    pool = []
    for (w, sid) in sorted(session_index.keys(), key=lambda k: (k[0], k[1])):
        meta = session_index.get((w, sid)) or {}
        for local_idx, gm in enumerate(meta.get("golden_memories", []) or []):
            if not isinstance(gm, str) or not gm.strip():
                continue
            pool.append({
                "global_index": len(pool), "week": w, "session_id": sid,
                "local_index": local_idx, "text": gm, "date": meta.get("date", ""),
            })
    return pool


def load_char_profiles(story_id):
    """canonical char -> preference_desc (ground truth)."""
    fp = os.path.join(C.DATA_ROOT, f"story_{story_id}", "character_profiles.json")
    out = {}
    if os.path.exists(fp):
        d = json.load(open(fp, encoding="utf-8"))
        for name, v in (d.items() if isinstance(d, dict) else []):
            out[normalize_char(name)] = (v or {}).get("preference_desc", "")
    return out


def short(s, n=400):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + " ..."
