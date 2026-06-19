#!/usr/bin/env python3
"""Shared IO helpers for the zero-cost paper analyses (paper/repro/).

All analyses read ONLY from already-saved artifacts under ``exp/paper/`` and
``data/scaling/`` -- no LLM calls, no re-judging. Outputs (CSV / JSON / PNG)
land in ``paper/repro/out/``.

Cell layout:  exp/paper/<method>/<model_alias>/<fbmode>/story_<id>/...
"""

import os
import csv
import glob
import json
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP_ROOT = os.path.join(REPO_ROOT, "exp", "paper")
DATA_ROOT = os.path.join(REPO_ROOT, "data", "scaling")
OUT_DIR = os.path.join(HERE, "out")

# vendor alias -> short label used in tables/figures
MODELS = {
    "deepseek-v4-flash": "deepseek",
    "gemini-3.5-flash": "gemini",
}
METHODS = ["id", "fc", "m0", "adamem"]
FBMODES = ["verbose", "with_gold"]


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR


def short_model(alias):
    return MODELS.get(alias, alias)


def cell_dir(method, model, fbmode):
    return os.path.join(EXP_ROOT, method, model, fbmode)


def list_stories(method, model, fbmode):
    base = cell_dir(method, model, fbmode)
    out = []
    for d in sorted(glob.glob(os.path.join(base, "story_*"))):
        try:
            out.append(int(os.path.basename(d).split("_")[1]))
        except Exception:
            continue
    return sorted(out)


def iter_cells(methods=None, models=None, fbmodes=None):
    """Yield (method, model_alias, fbmode) for every cell dir that exists."""
    methods = methods or METHODS
    models = models or list(MODELS)
    fbmodes = fbmodes or FBMODES
    for m in methods:
        for mm in models:
            for fb in fbmodes:
                if os.path.isdir(cell_dir(m, mm, fb)):
                    yield m, mm, fb


def load_qa_records(method, model, fbmode, story):
    """Return the list of per-QA records for one story (or None if missing)."""
    f = os.path.join(cell_dir(method, model, fbmode), f"story_{story}", "qa_records.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f, encoding="utf-8"))
    if isinstance(d, dict):
        return d.get("qa_records", [])
    return d


def load_qa_meta_blob(method, model, fbmode, story):
    """Return the full qa_records.json dict (for token_usage / accuracy fields)."""
    f = os.path.join(cell_dir(method, model, fbmode), f"story_{story}", "qa_records.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f, encoding="utf-8"))


def iter_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


@lru_cache(maxsize=None)
def load_test_qa_meta(story):
    """question_id -> question meta (info_category / qa_type / supersedes_chain)."""
    f = os.path.join(DATA_ROOT, f"story_{story}", "test_qa.json")
    meta = {}
    if os.path.exists(f):
        d = json.load(open(f, encoding="utf-8"))
        for q in d.get("test_questions", []):
            meta[q.get("question_id")] = q
    return meta


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def pct(num, den):
    return (100.0 * num / den) if den else None
