#!/usr/bin/env python3
"""Complete the extraction / recall judging for cells that were never judged.

Produces ``extraction_judged.jsonl`` + ``recall_judged.jsonl`` (+ ``judge_done.json``)
with the SAME schema as analysis/run_paper_judge.py, but routed through
call_llm.py (deepseek-v4-flash, <=128 concurrent). Resumable: cells that
already have judge_done.json are skipped unless --force.

Default targets the currently-missing cells (m0/adamem that lack judging):
  m0/deepseek/with_gold, adamem/deepseek/with_gold,
  m0|adamem / gemini / {verbose,with_gold}

Usage:
  python3 judge_memory.py                 # all missing cells, all stories
  python3 judge_memory.py --force         # re-judge
  python3 judge_memory.py --methods m0 --models gemini-3.5-flash
"""

import os
import json
import time
import argparse

import _common as C
import _judge_common as J

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
content as ANY of the target golden memories. Subject + fact + time anchor
must align; finer details in the golden memory must be present in the
candidate, but extra non-conflicting phrasing is fine.

Return ONE line of JSON only:
{{"target_in_recall": true|false,
  "target_rank": <1-based rank int or null>,
  "matched_memory_id": "<the candidate id at that rank, or null>",
  "matched_target_index": <int or null>,
  "reason": "<<= 30 words>"}}

``target_rank`` must be the SMALLEST rank that matches. If nothing matches,
return ``null`` for the three optional fields."""

DEFAULT_MISSING = [
    ("m0", "deepseek-v4-flash", "with_gold"),
    ("adamem", "deepseek-v4-flash", "with_gold"),
    ("m0", "gemini-3.5-flash", "verbose"),
    ("m0", "gemini-3.5-flash", "with_gold"),
    ("adamem", "gemini-3.5-flash", "verbose"),
    ("adamem", "gemini-3.5-flash", "with_gold"),
]


def _judge_extraction(cell_path, story_id, pool, golden_block):
    rows = list(C.iter_jsonl(os.path.join(cell_path, "extracted_memories.jsonl")))
    jobs = []  # (meta, candidate_text)
    for row in rows:
        for item in row.get("added", []) or []:
            cand = (item.get("memory") or "").strip()
            jobs.append(({
                "week": row.get("week") or 0,
                "date": row.get("date", ""),
                "source": row.get("source", ""),
                "memory_id": str(item.get("id", "")),
            }, cand))
    prompts = [EXTRACTION_PROMPT.format(golden_block=golden_block, candidate=c)
               for _, c in jobs]
    parsed = J.judge_many(prompts) if prompts else []
    out = []
    for (meta, cand), p in zip(jobs, parsed):
        base = {"story_id": story_id, "week": meta["week"], "date": meta["date"],
                "source": meta["source"], "memory_id": meta["memory_id"],
                "memory_text": cand, "is_in_golden": False,
                "matched_golden_global_index": None, "matched_session_key": None,
                "matched_golden_index": None, "reason": ""}
        if not cand:
            base["reason"] = "empty extracted memory"
            out.append(base); continue
        if not pool:
            base["reason"] = "story has no golden memories"
            out.append(base); continue
        if not p:
            base["reason"] = "judge parse failed"
            out.append(base); continue
        is_in = bool(p.get("is_in_golden", False))
        g_idx = p.get("matched_golden_index")
        try:
            g_idx = int(g_idx) if g_idx is not None else None
        except Exception:
            g_idx = None
        if is_in and g_idx is not None and 0 <= g_idx < len(pool):
            gm = pool[g_idx]
            base.update(is_in_golden=True,
                        matched_golden_global_index=gm["global_index"],
                        matched_session_key={"week": gm["week"], "session_id": gm["session_id"]},
                        matched_golden_index=gm["local_index"])
        base["reason"] = str(p.get("reason", ""))[:200]
        out.append(base)
    return out


def _judge_recall(cell_path, story_id, session_index, qa_index):
    rows = list(C.iter_jsonl(os.path.join(cell_path, "recall.jsonl")))
    judged = [None] * len(rows)
    prompts = []
    prompt_map = []  # (row_index, skeleton_row) for each prompt
    for i, row in enumerate(rows):
        qid = row.get("question_id") or ""
        week = row.get("week") or 0
        qa = qa_index.get(qid) or {}
        refs = qa.get("target_memory_refs") or row.get("target_memory_refs") or []
        question = qa.get("question") or row.get("query") or ""
        skel = {"story_id": story_id, "week": week, "question_id": qid,
                "question": question, "target_memory_refs": refs,
                "target_in_recall": False, "target_rank": None,
                "matched_memory_id": None, "matched_target_index": None}
        if not refs:
            skel.update(reason="skipped: QA has no target_memory_refs", skipped=True)
            judged[i] = skel; continue
        targets = []
        for ref in refs:
            try:
                w = int(ref.get("week")); sid = str(ref.get("session_id") or "")
                idx = int(ref.get("index"))
            except Exception:
                continue
            meta = session_index.get((w, sid))
            if meta and 0 <= idx < len(meta.get("golden_memories") or []):
                targets.append({"ref": {"week": w, "session_id": sid, "index": idx},
                                "text": meta["golden_memories"][idx]})
        if not targets:
            skel.update(reason="skipped: target_memory_refs do not resolve to any golden",
                        skipped=True)
            judged[i] = skel; continue
        retrieved = row.get("retrieved") or []
        if not retrieved:
            skel.update(reason="no retrieval candidates", skipped=False)
            judged[i] = skel; continue
        golden_block = "\n".join(
            f"[{j}] (week={t['ref']['week']} session={t['ref']['session_id']} idx={t['ref']['index']}) "
            f"{J.short(t['text'])}" for j, t in enumerate(targets))
        retrieved_block = "\n".join(
            f"#{r.get('rank', k+1)} [id={r.get('id','')}] {J.short(r.get('text',''))}"
            for k, r in enumerate(retrieved))
        prompts.append(RECALL_PROMPT.format(question=question, golden_block=golden_block,
                                            retrieved_block=retrieved_block))
        prompt_map.append((i, skel))
    parsed = J.judge_many(prompts) if prompts else []
    for (i, skel), p in zip(prompt_map, parsed):
        if not p:
            skel.update(reason="judge parse failed", skipped=False)
            judged[i] = skel; continue
        in_recall = bool(p.get("target_in_recall", False))
        try:
            rank = int(p["target_rank"]) if p.get("target_rank") is not None else None
        except Exception:
            rank = None
        mid = p.get("matched_memory_id")
        try:
            mt = int(p["matched_target_index"]) if p.get("matched_target_index") is not None else None
        except Exception:
            mt = None
        skel.update(target_in_recall=in_recall,
                    target_rank=rank if in_recall else None,
                    matched_memory_id=(str(mid) if (in_recall and mid is not None) else None),
                    matched_target_index=mt if in_recall else None,
                    reason=str(p.get("reason", ""))[:200], skipped=False)
        judged[i] = skel
    return [r for r in judged if r is not None]


def judge_cell(method, model, fbmode, story, force=False):
    cell_path = os.path.join(C.cell_dir(method, model, fbmode), f"story_{story}")
    done_path = os.path.join(cell_path, "judge_done.json")
    ext_path = os.path.join(cell_path, "extraction_judged.jsonl")
    rec_path = os.path.join(cell_path, "recall_judged.jsonl")
    if not force and os.path.exists(done_path):
        return "skip", 0, 0
    for p in (ext_path, rec_path, done_path):
        if os.path.exists(p):
            os.remove(p)

    session_index, qa_index = J.load_session_goldens(story)
    pool = J.build_golden_pool(session_index)
    golden_block = "\n".join(
        f"[{g['global_index']}] (week={g['week']} session={g['session_id']} idx={g['local_index']}) {g['text']}"
        for g in pool)

    ext_rows = _judge_extraction(cell_path, story, pool, golden_block)
    with open(ext_path, "w", encoding="utf-8") as f:
        for r in ext_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rec_rows = _judge_recall(cell_path, story, session_index, qa_index)
    with open(rec_path, "w", encoding="utf-8") as f:
        for r in rec_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(done_path, "w", encoding="utf-8") as f:
        json.dump({"method": method, "memory_model": model, "fbmode": fbmode,
                   "story_id": story, "extracted_judged_rows": len(ext_rows),
                   "recall_judged_rows": len(rec_rows), "golden_pool_size": len(pool),
                   "judged_by": "call_llm/deepseek-v4-flash", "ts": int(time.time())},
                  f, ensure_ascii=False, indent=2)
    return "done", len(ext_rows), len(rec_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=None, help="CSV subset; default missing cells")
    ap.add_argument("--models", default=None, help="CSV vendor aliases")
    ap.add_argument("--fbmodes", default=None, help="CSV fbmodes")
    ap.add_argument("--stories", default=None, help="CSV story ids")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.methods or a.models or a.fbmodes:
        methods = (a.methods or "id,fc,m0,adamem").split(",")
        models = (a.models or ",".join(C.MODELS)).split(",")
        fbmodes = (a.fbmodes or "verbose,with_gold").split(",")
        cells = [(m, mm, fb) for m in methods for mm in models for fb in fbmodes
                 if os.path.isdir(C.cell_dir(m, mm, fb))]
    else:
        cells = [c for c in DEFAULT_MISSING if os.path.isdir(C.cell_dir(*c))]

    story_filter = set(int(x) for x in a.stories.split(",")) if a.stories else None
    print(f"[judge_memory] {len(cells)} cells via {J.JUDGE_MODEL}")
    t0 = time.time()
    for method, model, fbmode in cells:
        stories = C.list_stories(method, model, fbmode)
        if story_filter:
            stories = [s for s in stories if s in story_filter]
        for s in stories:
            st = time.time()
            status, ne, nr = judge_cell(method, model, fbmode, s, force=a.force)
            tag = f"{method}/{C.short_model(model)}/{fbmode}/story_{s}"
            print(f"  [{status}] {tag} extracted={ne} recall={nr} ({time.time()-st:.0f}s)",
                  flush=True)
    print(f"[judge_memory] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
