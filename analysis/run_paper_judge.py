#!/usr/bin/env python3
"""Paper-experiment post-hoc judging pipeline.

Walks every cell under ``exp/paper/<method>/<memory_model>/<fbmode>/story_<id>/``
and produces two judged jsonl artifacts per cell:

* ``extraction_judged.jsonl`` -- one row per add-call x extracted-memory.
  Decides whether the extracted memory is "in golden" for the originating
  session, where "golden" comes from the session's ``golden_memories`` field
  (added by prepare_data.py / requirement 1).

* ``recall_judged.jsonl`` -- one row per QA. Decides whether the QA's
  ``target_memory_refs`` is matched somewhere in the ``retrieved`` list,
  and at what rank, by asking the judge LLM whether each candidate text is
  semantically equivalent to any of the target golden memories.

Both judgments are produced by the ``deepseek-v4-flash`` judge through the
OpenAI-compatible chat endpoint configured via OPENAI_BASE_URL.

Resumable: if a cell already has BOTH judged jsonls (with the same row count
as the source), it is skipped unless ``--force`` is passed.

Usage:
    python AdaMem/analysis/run_paper_judge.py
    python AdaMem/analysis/run_paper_judge.py --parallel 16
    python run_paper_judge.py --methods adamem --memory-models deepseek-v4-flash
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (  # noqa: E402
    log, JUDGE_MODEL,
    PAPER_EXP_ROOT, paper_story_dir,
    make_chat_call, append_jsonl,
    print_effective_config, sanity_check_connectivity,
    DATA_DIR,
    MEMORY_MODEL_CHOICES,
)


# ---------- LLM judge helpers ----------

EXTRACTION_PROMPT = """You are judging whether an extracted "memory" matches
ANY of the GOLDEN memories that an ideal preference-aware memory system
should keep for the WHOLE story (across all sessions of all weeks).

The candidate may have come from any add-call in the story; do NOT restrict
to a particular session.

## Golden memory pool (the ideal target list; rank starts at 0)
{golden_block}

## Candidate extracted memory
"{candidate}"

## Decision
Decide whether the candidate matches ONE of the golden memories above.
"Match" means the candidate carries the same factual content as the golden
memory (same subject, same fact, same time anchor at the granularity stated
in the golden memory). Phrasing / tense / extra inert wording may differ;
the substance must match. If a golden item has finer details (e.g. a clock
time or a specific person) that the candidate lacks, it is NOT a match.

Reply with a single line of JSON only:
{{"is_in_golden": true|false, "matched_golden_index": <int or null>, "reason": "<<= 30 words>"}}

If multiple golden items could match, return the smallest index. If none
matches, return ``null``."""

RECALL_PROMPT = """You are judging whether a memory-retrieval system surfaced the
target golden memory(ies) when answering a question.

## Question
"{question}"

## Target golden memories (any one of these is acceptable; rank starts at 0)
{golden_block}

## Retrieved candidates (rank starts at 1, top-k order preserved)
{retrieved_block}

## Decision
For each candidate (in rank order), say YES iff it conveys the same factual
content as ANY of the target golden memories. Same-substance matching as
described above (subject + fact + time anchor must align; finer details in
the golden memory must be present in the candidate, but extra non-conflicting
phrasing is fine).

Return ONE line of JSON only:
{{"target_in_recall": true|false,
  "target_rank": <1-based rank int or null>,
  "matched_memory_id": "<the candidate id at that rank, or null>",
  "matched_target_index": <int or null>,
  "reason": "<<= 30 words>"}}

``target_rank`` must be the SMALLEST rank that matches; ``matched_memory_id``
must be the candidate's id field (a string). If nothing matches, return
``null`` for the three optional fields."""


_JSON_LINE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_line(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences if any.
    if text.startswith("```"):
        # remove first fence line
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()
    m = _JSON_LINE_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ---------- Dataset loading helpers ----------

def _load_session_goldens(story_id: int, num_weeks: int):
    """Return:

      session_index: {(week, session_id): {"date": str, "golden_memories": [...] }}
      qa_index:      {question_id: {"question":..., "target_memory_refs":[{week,session_id,index}, ...] }}
    """
    sd = os.path.join(DATA_DIR, f"story_{story_id}")
    session_index: Dict[Tuple[int, str], Dict[str, Any]] = {}
    weeks_seen = 0
    for w in range(1, num_weeks + 1):
        path = os.path.join(sd, f"week{w}.json")
        if not os.path.exists(path):
            continue
        weeks_seen = w
        with open(path) as f:
            wd = json.load(f)
        for sess in wd.get("conversations", []) or []:
            sid = sess.get("session_id") or ""
            session_index[(w, sid)] = {
                "date": sess.get("date", ""),
                "day": sess.get("day", ""),
                "topic": sess.get("topic", ""),
                "focal_character": sess.get("focal_character", ""),
                "golden_memories": list(sess.get("golden_memories", []) or []),
            }
    qa_index: Dict[str, Dict[str, Any]] = {}
    qa_path = os.path.join(sd, "test_qa.json")
    if os.path.exists(qa_path):
        with open(qa_path) as f:
            qd = json.load(f)
        for qa in qd.get("test_questions", []) or []:
            qid = qa.get("question_id")
            if not qid:
                continue
            qa_index[qid] = {
                "question": qa.get("question", ""),
                "gold_answer": qa.get("gold_answer", ""),
                "target_memory": qa.get("target_memory", ""),
                "target_memory_refs": list(qa.get("target_memory_refs", []) or []),
            }
    return session_index, qa_index, weeks_seen


# ---------- Cell discovery ----------

def _discover_cells(out_root: str,
                    methods: Optional[List[str]] = None,
                    memory_models: Optional[List[str]] = None,
                    fbmodes: Optional[List[str]] = None,
                    stories: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Walk ``out_root`` and return per-cell dicts.

    Each dict has ``method``, ``memory_model``, ``fbmode``, ``story_id``,
    ``cell_dir``. Filters are applied if supplied.
    """
    found = []
    if not os.path.isdir(out_root):
        return found
    for method in sorted(os.listdir(out_root)):
        method_dir = os.path.join(out_root, method)
        if not os.path.isdir(method_dir) or method.startswith("_"):
            continue
        if methods and method not in methods:
            continue
        for mm in sorted(os.listdir(method_dir)):
            mm_dir = os.path.join(method_dir, mm)
            if not os.path.isdir(mm_dir):
                continue
            if memory_models and mm not in memory_models:
                continue
            for fb in sorted(os.listdir(mm_dir)):
                fb_dir = os.path.join(mm_dir, fb)
                if not os.path.isdir(fb_dir):
                    continue
                if fbmodes and fb not in fbmodes:
                    continue
                for sd in sorted(os.listdir(fb_dir)):
                    if not sd.startswith("story_"):
                        continue
                    cell_dir = os.path.join(fb_dir, sd)
                    if not os.path.isdir(cell_dir):
                        continue
                    try:
                        sid = int(sd.split("_", 1)[1])
                    except Exception:
                        continue
                    if stories and sid not in stories:
                        continue
                    found.append({
                        "method": method,
                        "memory_model": mm,
                        "fbmode": fb,
                        "story_id": sid,
                        "cell_dir": cell_dir,
                    })
    return found


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


# ---------- Per-row judging ----------

def _build_story_golden_pool(session_index) -> List[Dict[str, Any]]:
    """Flatten every session's ``golden_memories`` into one story-level pool.

    Each entry: ``{"global_index": int, "week": int, "session_id": str,
    "local_index": int, "text": str, "date": str}``. ``global_index`` is the
    candidate's rank inside the pool (0-based) -- this is what the judge LLM
    sees and reports back.
    """
    pool: List[Dict[str, Any]] = []
    # Stable ordering: by week, then by session_id, then by local_index.
    for (w, sid) in sorted(session_index.keys(), key=lambda k: (k[0], k[1])):
        meta = session_index.get((w, sid)) or {}
        for local_idx, gm in enumerate(meta.get("golden_memories", []) or []):
            if not isinstance(gm, str) or not gm.strip():
                continue
            pool.append({
                "global_index": len(pool),
                "week": w,
                "session_id": sid,
                "local_index": local_idx,
                "text": gm,
                "date": meta.get("date", ""),
            })
    return pool


def _judge_one_extracted_memory(*, candidate_row: Dict[str, Any],
                                added_item: Dict[str, Any],
                                golden_pool: List[Dict[str, Any]],
                                golden_block: str,
                                story_id: int) -> Dict[str, Any]:
    """Judge ONE extracted memory against the story-level golden pool.

    Returns a single judged row dict. The candidate's provenance (week / date
    / source) is preserved on the row so consumers can audit which add-call
    produced the hit.
    """
    week = candidate_row.get("week") or 0
    date = candidate_row.get("date", "")
    source = candidate_row.get("source", "")
    candidate = (added_item.get("memory") or "").strip()
    mem_id = str(added_item.get("id", ""))

    if not candidate:
        return {
            "story_id": story_id, "week": week, "date": date, "source": source,
            "memory_id": mem_id, "memory_text": "",
            "is_in_golden": False,
            "matched_golden_global_index": None,
            "matched_session_key": None,
            "matched_golden_index": None,
            "reason": "empty extracted memory",
        }
    if not golden_pool:
        return {
            "story_id": story_id, "week": week, "date": date, "source": source,
            "memory_id": mem_id, "memory_text": candidate,
            "is_in_golden": False,
            "matched_golden_global_index": None,
            "matched_session_key": None,
            "matched_golden_index": None,
            "reason": "story has no golden memories",
        }

    prompt = EXTRACTION_PROMPT.format(
        golden_block=golden_block, candidate=candidate,
    )
    tag_id = (mem_id or "noid")[:32]
    tag = f"judge_extract_s{story_id}_w{week}_{tag_id}"
    text = make_chat_call(
        JUDGE_MODEL, prompt,
        max_tokens=8192, temperature=0, timeout=60,
        bucket="judge",
        tag=tag, story_id=story_id, week=week, kind="judge",
    )
    parsed = _parse_json_line(text or "") or {}
    is_in = bool(parsed.get("is_in_golden", False))
    g_idx = parsed.get("matched_golden_index", None)
    try:
        g_idx = int(g_idx) if g_idx is not None else None
    except Exception:
        g_idx = None

    matched_session_key = None
    matched_local_idx = None
    matched_global_idx = None
    if is_in and g_idx is not None and 0 <= g_idx < len(golden_pool):
        gm = golden_pool[g_idx]
        matched_session_key = {"week": gm["week"], "session_id": gm["session_id"]}
        matched_local_idx = gm["local_index"]
        matched_global_idx = gm["global_index"]
    elif is_in:
        # Judge said yes but gave a bogus index; treat as no-match for safety.
        is_in = False

    return {
        "story_id": story_id, "week": week, "date": date, "source": source,
        "memory_id": mem_id, "memory_text": candidate,
        "is_in_golden": is_in,
        "matched_golden_global_index": matched_global_idx,
        "matched_session_key": matched_session_key,
        "matched_golden_index": matched_local_idx,
        "reason": str(parsed.get("reason", ""))[:200],
    }


def _judge_recall_row(*, recall_row: Dict[str, Any],
                      session_index,
                      qa_index,
                      story_id: int):
    qid = recall_row.get("question_id") or ""
    week = recall_row.get("week") or 0
    qa = qa_index.get(qid) or {}
    refs = qa.get("target_memory_refs") or recall_row.get("target_memory_refs") or []
    question = qa.get("question") or recall_row.get("query") or ""
    if not refs:
        return {
            "story_id": story_id, "week": week, "question_id": qid,
            "question": question,
            "target_memory_refs": [],
            "target_in_recall": False,
            "target_rank": None,
            "matched_memory_id": None,
            "matched_target_index": None,
            "reason": "skipped: QA has no target_memory_refs",
            "skipped": True,
        }
    target_texts = []
    for ref in refs:
        try:
            w = int(ref.get("week"))
            sid = str(ref.get("session_id") or "")
            i = int(ref.get("index"))
        except Exception:
            continue
        meta = session_index.get((w, sid))
        if not meta:
            continue
        gms = meta.get("golden_memories") or []
        if 0 <= i < len(gms):
            target_texts.append({
                "ref": {"week": w, "session_id": sid, "index": i},
                "text": gms[i],
            })
    if not target_texts:
        return {
            "story_id": story_id, "week": week, "question_id": qid,
            "question": question,
            "target_memory_refs": refs,
            "target_in_recall": False,
            "target_rank": None,
            "matched_memory_id": None,
            "matched_target_index": None,
            "reason": "skipped: target_memory_refs do not resolve to any golden",
            "skipped": True,
        }
    retrieved = recall_row.get("retrieved") or []
    if not retrieved:
        return {
            "story_id": story_id, "week": week, "question_id": qid,
            "question": question,
            "target_memory_refs": refs,
            "target_in_recall": False,
            "target_rank": None,
            "matched_memory_id": None,
            "matched_target_index": None,
            "reason": "no retrieval candidates",
            "skipped": False,
        }

    # Truncate very long candidate texts so the judge prompt stays small.
    def _short(s: str, n: int = 600) -> str:
        s = (s or "").replace("\n", " ").strip()
        return s if len(s) <= n else s[:n] + " ..."

    golden_block = "\n".join(
        f"[{i}] (week={t['ref']['week']} session={t['ref']['session_id']} idx={t['ref']['index']}) "
        f"{_short(t['text'], 400)}"
        for i, t in enumerate(target_texts)
    )
    retrieved_block = "\n".join(
        f"#{r.get('rank', i+1)} [id={r.get('id','')}] {_short(r.get('text',''), 400)}"
        for i, r in enumerate(retrieved)
    )
    prompt = RECALL_PROMPT.format(
        question=question,
        golden_block=golden_block,
        retrieved_block=retrieved_block,
    )
    tag = f"judge_recall_s{story_id}_w{week}_{qid}"
    text = make_chat_call(
        JUDGE_MODEL, prompt,
        max_tokens=8192, temperature=0, timeout=90,
        bucket="judge",
        tag=tag, story_id=story_id, week=week, kind="judge",
    )
    parsed = _parse_json_line(text or "") or {}
    in_recall = bool(parsed.get("target_in_recall", False))
    try:
        rank = int(parsed["target_rank"]) if parsed.get("target_rank") is not None else None
    except Exception:
        rank = None
    mid = parsed.get("matched_memory_id")
    try:
        mt_idx = int(parsed["matched_target_index"]) if parsed.get("matched_target_index") is not None else None
    except Exception:
        mt_idx = None
    return {
        "story_id": story_id, "week": week, "question_id": qid,
        "question": question,
        "target_memory_refs": refs,
        "target_in_recall": in_recall,
        "target_rank": rank if in_recall else None,
        "matched_memory_id": (str(mid) if (in_recall and mid is not None) else None),
        "matched_target_index": mt_idx if in_recall else None,
        "reason": str(parsed.get("reason", ""))[:200],
        "skipped": False,
    }


# ---------- Per-cell driver ----------

def _judge_one_cell(cell: Dict[str, Any], *, num_weeks: int, force: bool,
                    parallel: int) -> Dict[str, Any]:
    method = cell["method"]
    memory_model = cell["memory_model"]
    fbmode = cell["fbmode"]
    story_id = cell["story_id"]
    cell_dir = cell["cell_dir"]
    extracted_path = os.path.join(cell_dir, "extracted_memories.jsonl")
    recall_path = os.path.join(cell_dir, "recall.jsonl")
    extraction_judged_path = os.path.join(cell_dir, "extraction_judged.jsonl")
    recall_judged_path = os.path.join(cell_dir, "recall_judged.jsonl")
    judge_done_path = os.path.join(cell_dir, "judge_done.json")

    # Resume: use an explicit marker file so empty inputs are not mistaken
    # for "not yet judged".
    if not force and os.path.exists(judge_done_path):
        return {"cell": cell, "skipped": True,
                "extracted": _count_lines(extraction_judged_path),
                "recall": _count_lines(recall_judged_path)}

    # Force or first-run: wipe judged outputs so we don't leave stale rows
    # on top of a fresh run.
    for p in (extraction_judged_path, recall_judged_path, judge_done_path):
        if os.path.exists(p):
            os.remove(p)

    session_index, qa_index, weeks_seen = _load_session_goldens(story_id, num_weeks)
    golden_pool = _build_story_golden_pool(session_index)
    golden_block = "\n".join(
        f"[{gm['global_index']}] (week={gm['week']} session={gm['session_id']} idx={gm['local_index']}) {gm['text']}"
        for gm in golden_pool
    )

    # ---- Extraction judging: one judged row per extracted memory item ----
    extraction_inputs: List[Dict[str, Any]] = list(_iter_jsonl(extracted_path))
    # Flatten: list of (candidate_row, added_item) pairs.
    extraction_jobs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for row in extraction_inputs:
        for item in row.get("added", []) or []:
            extraction_jobs.append((row, item))

    extracted_count = 0
    if extraction_jobs:
        if parallel <= 1:
            for row, item in extraction_jobs:
                jr = _judge_one_extracted_memory(
                    candidate_row=row, added_item=item,
                    golden_pool=golden_pool, golden_block=golden_block,
                    story_id=story_id,
                )
                append_jsonl(extraction_judged_path, jr)
                extracted_count += 1
        else:
            with ThreadPoolExecutor(max_workers=parallel) as ex:
                futs = [
                    ex.submit(_judge_one_extracted_memory,
                              candidate_row=row, added_item=item,
                              golden_pool=golden_pool, golden_block=golden_block,
                              story_id=story_id)
                    for row, item in extraction_jobs
                ]
                for fut in as_completed(futs):
                    try:
                        append_jsonl(extraction_judged_path, fut.result())
                        extracted_count += 1
                    except Exception as e:
                        log.error(f"[judge] extraction error: {e}")

    # ---- Recall judging ----
    recall_inputs: List[Dict[str, Any]] = list(_iter_jsonl(recall_path))
    recall_count = 0
    if recall_inputs:
        if parallel <= 1:
            for row in recall_inputs:
                jr = _judge_recall_row(
                    recall_row=row, session_index=session_index,
                    qa_index=qa_index, story_id=story_id,
                )
                append_jsonl(recall_judged_path, jr)
                recall_count += 1
        else:
            with ThreadPoolExecutor(max_workers=parallel) as ex:
                futs = [
                    ex.submit(_judge_recall_row,
                              recall_row=row, session_index=session_index,
                              qa_index=qa_index, story_id=story_id)
                    for row in recall_inputs
                ]
                for fut in as_completed(futs):
                    try:
                        append_jsonl(recall_judged_path, fut.result())
                        recall_count += 1
                    except Exception as e:
                        log.error(f"[judge] recall error: {e}")

    # Drop a done marker so resumable runs see this cell as judged even when
    # one of the inputs (e.g. ID's empty extracted_memories.jsonl) is empty.
    with open(judge_done_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": method,
            "memory_model": memory_model,
            "fbmode": fbmode,
            "story_id": story_id,
            "extracted_jobs": len(extraction_jobs),
            "recall_jobs": len(recall_inputs),
            "extracted_judged_rows": extracted_count,
            "recall_judged_rows": recall_count,
            "golden_pool_size": len(golden_pool),
            "ts": int(time.time()),
        }, f, ensure_ascii=False, indent=2)

    return {"cell": cell, "skipped": False,
            "extracted": extracted_count, "recall": recall_count}


def _iter_jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Paper-experiment LLM-judge pipeline")
    parser.add_argument("--out-dir", default=PAPER_EXP_ROOT,
                        help="Root of paper experiments (default: AdaMem/exp/paper)")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated subset (id,fc,m0,adamem). Default: all.")
    parser.add_argument("--memory-models", default=None,
                        help="Comma-separated subset of vendor aliases.")
    parser.add_argument("--fbmodes", default=None,
                        help="Comma-separated subset (with_gold,verbose).")
    parser.add_argument("--stories", default=None,
                        help="Comma-separated story ids; default: every story under each cell.")
    parser.add_argument("--weeks", type=int, default=10,
                        help="Number of weeks present in the dataset (default: 10).")
    parser.add_argument("--parallel", type=int, default=32,
                        help="Within-cell judge concurrency.")
    parser.add_argument("--parallel-cell", type=int, default=1,
                        help="Across-cell concurrency.")
    parser.add_argument("--force", action="store_true",
                        help="Re-judge even if judged jsonl already exists.")
    parser.add_argument("--skip-sanity-check", action="store_true")
    args = parser.parse_args()

    methods = [x.strip() for x in (args.methods or "").split(",") if x.strip()] or None
    memory_models = [x.strip() for x in (args.memory_models or "").split(",") if x.strip()] or None
    fbmodes = [x.strip() for x in (args.fbmodes or "").split(",") if x.strip()] or None
    stories = None
    if args.stories:
        stories = [int(x) for x in args.stories.split(",") if x.strip()]
    if memory_models:
        for mm in memory_models:
            if mm not in MEMORY_MODEL_CHOICES:
                # still allow -- the directory tree may carry slugs that
                # were created with non-canonical aliases.
                log.warning(f"--memory-models: {mm} not in MEMORY_MODEL_CHOICES")

    print_effective_config({
        "judge_root": args.out_dir,
        "filters": f"methods={methods} memory_models={memory_models} fbmodes={fbmodes} stories={stories}",
        "parallel": args.parallel,
        "parallel_cell": args.parallel_cell,
        "force": args.force,
    })
    if not args.skip_sanity_check:
        try:
            sanity_check_connectivity(require_embedding=False)
        except Exception as e:
            print(f"sanity_check FAILED: {e}", file=sys.stderr)
            sys.exit(5)

    cells = _discover_cells(args.out_dir, methods, memory_models, fbmodes, stories)
    print(f"discovered {len(cells)} cells")
    if not cells:
        return

    t0 = time.time()
    if args.parallel_cell <= 1:
        for cell in cells:
            r = _judge_one_cell(cell, num_weeks=args.weeks, force=args.force,
                                parallel=args.parallel)
            _print_cell_result(r)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel_cell) as ex:
            futs = [
                ex.submit(_judge_one_cell, cell,
                          num_weeks=args.weeks, force=args.force,
                          parallel=args.parallel)
                for cell in cells
            ]
            for fut in as_completed(futs):
                try:
                    _print_cell_result(fut.result())
                except Exception as e:
                    print(f"cell crashed: {e}", file=sys.stderr)
    elapsed = (time.time() - t0) / 60.0
    print(f"\njudge done in {elapsed:.1f} min")


def _print_cell_result(r):
    cell = r["cell"]
    tag = f"{cell['method']}/{cell['memory_model']}/{cell['fbmode']}/story_{cell['story_id']}"
    if r.get("skipped"):
        print(f"  [skip] {tag} (already judged)")
    else:
        print(f"  [done] {tag} extracted={r['extracted']} recall={r['recall']}")


if __name__ == "__main__":
    main()
