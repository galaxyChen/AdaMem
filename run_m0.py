#!/usr/bin/env python3
"""AdaMem run_m0: Mem0 baseline.

Per-week pipeline:
  1. Add all of this week's raw dialogues to mem0 (per-day batches, with the
     conversation's observation_date set so mem0 anchors the facts correctly).
  2. Answer this week's QA via independent per-question test snippets. Each
     question sees a system prompt containing the generic answer guide plus
     its own retrieved memories, then a user turn containing only that question.
     All questions in the same week are requested in parallel and do not share
     cross-question chat state.
  3. Add the actual QA test snippets of this week (system, user Q, assistant A,
     user feedback, assistant OK) into mem0 as one extra add() call so future
     weeks can recall what was asked and how the user reacted.

For the paper matrix, M0 is run **independently in every (memory_model,
fbmode) cell** so each cell has its own qa_records / recall.jsonl /
extracted_memories.jsonl artifacts. Every mem0.add call appends one row to
``extracted_memories.jsonl`` (input dialogue + returned memory items); every
mem0.search call appends one row to ``recall.jsonl`` (question + retrieved
memory list with scores).

Usage:
    python run_m0.py --story all --fbmode with_gold
    python run_m0.py --story 1 --fbmode verbose --memory-model gemini-3.5-flash
"""

import argparse, gc, json, os, sys, time, traceback, subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, call_answer_messages, judge_answer, load_progress, update_progress,
    reset_token_usage, get_token_usage_snapshot,
    build_mem0_client, mem0_safe_call, close_mem0, purge_mem0_collection,
    set_mem0_observation_date, clear_mem0_observation_date,
    set_mem0_client_metadata, clear_mem0_client_metadata,
    load_story_bundle, story_exp_dir, format_feedback,
    NUM_STORIES_DEFAULT, NUM_WEEKS_DEFAULT, DEFAULT_PARALLEL,
    set_memory_model, get_memory_model_alias, MEMORY_MODEL,
    paper_story_dir, qdrant_collection_name, append_jsonl,
    is_paper_cell_complete,
    make_qa_tag, make_memory_tag, print_effective_config,
    sanity_check_connectivity, mark_paper_cell_done,
)


def run_m0_for_story(story_id, *, num_weeks, fbmode, out_dir):
    out_path = os.path.join(out_dir, "qa_records.json")
    extracted_path = os.path.join(out_dir, "extracted_memories.jsonl")
    recall_path = os.path.join(out_dir, "recall.jsonl")
    if is_paper_cell_complete(out_dir):
        print(f"  [M0/{fbmode}] story {story_id} ✓ (cached: {out_dir})")
        with open(out_path) as f:
            return json.load(f)

    for p in (out_path, extracted_path, recall_path, os.path.join(out_dir, "done.json")):
        if os.path.exists(p):
            os.remove(p)

    weeks_data, qa_data, ideal_data, profiles = load_story_bundle(story_id, num_weeks)
    questions_by_week = {}
    for q in qa_data["test_questions"]:
        questions_by_week.setdefault(q["test_week"], []).append(q)

    user_id = qdrant_collection_name("m0", fbmode, story_id)
    collection = user_id
    # P0: hard-wipe the qdrant collection on disk BEFORE building the client.
    # delete_all alone leaks vectors across runs (see debug note 2026-06-17);
    # we now refuse to start the run if the wipe fails.
    purge_mem0_collection(collection)
    client = build_mem0_client(collection)

    qa_records = []
    qa_turns_global = []   # actual test snippets across the whole story

    def _persist_add(*, source, week, date, msgs, ret, err):
        results = []
        if isinstance(ret, dict):
            results = ret.get("results", []) or []
        append_jsonl(extracted_path, {
            "story_id": story_id,
            "week": week,
            "date": date,
            "source": source,
            "method": "M0",
            "fbmode": fbmode,
            "memory_model": get_memory_model_alias(),
            "input_messages": msgs,
            "input_message_count": len(msgs),
            "added": [
                {
                    "id": str(r.get("id", "")),
                    "memory": r.get("memory", ""),
                    "event": r.get("event", ""),
                    "metadata": r.get("metadata") or {},
                } for r in results if isinstance(r, dict)
            ],
            "error": err,
        })

    def _qa_system_prompt(context):
        return (
            "You are Linchen's AI assistant. Answer concisely using only the memories below. "
            "If the memories don't contain the answer, say \"I don't remember.\"\n\n"
            f"Memories:\n{context}"
        )

    def _answer_one_qa(q, retrieved):
        qid = q["question_id"]
        context = "\n".join(f"- {t}" for t, _, _, _ in retrieved) if retrieved else "(no memory)"
        system_prompt = _qa_system_prompt(context)
        qa_tag = make_qa_tag(story_id, week, qid)
        ai = call_answer_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q["question"]},
            ],
            tag=qa_tag, story_id=story_id, week=week, kind="qa",
        )
        ok = judge_answer(q["question"], q["gold_answer"], ai,
                          story_id=story_id, week=week, qid=qid)
        log.info(f"[M0] story={story_id} week={week} qa qid={qid} "
                f"retrieved={len(retrieved)} correct={ok}")
        feedback = format_feedback(q, ok, fbmode=fbmode, character_profiles=profiles)
        test_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q["question"]},
            {"role": "assistant", "content": ai},
            {"role": "user", "content": feedback},
            {"role": "assistant", "content": "OK"},
        ]
        record = {
            "week": week, "question_id": qid,
            "character": q.get("character", ""),
            "topic_anchor": q.get("topic_anchor", ""),
            "qa_type": q.get("qa_type", ""),
            "question": q["question"], "gold_answer": q["gold_answer"],
            "golden_feedback": q.get("golden_feedback", ""),
            "ai_answer": ai, "correct": ok,
            "feedback": feedback,
            "test_messages": test_messages,
            "retrieved": [{"text": t, "score": round(s, 3) if isinstance(s, (int, float)) else s,
                           "id": str(mid)}
                          for t, s, mid, _meta in retrieved],
        }
        return record, test_messages

    print(f"  [M0/{fbmode}] story {story_id} -> {out_dir}: ", end="", flush=True)
    for week in range(1, num_weeks + 1):
        # 1) Add this week's dialogues, batched by date.
        by_date = {}
        for conv in weeks_data[week]["conversations"]:
            d = conv.get("date", f"w{week}")
            by_date.setdefault(d, []).extend(conv.get("messages", []))
        for d in sorted(by_date.keys()):
            msgs = by_date[d]
            if not msgs:
                continue
            set_mem0_observation_date(d)
            set_mem0_client_metadata({
                "tag": make_memory_tag(story_id, week, d),
                "story_id": story_id,
                "week": week,
                "kind": "memory_add",
                "source": "dialogue",
            })
            add_ret = None
            add_err = None
            try:
                add_ret = mem0_safe_call(client.add, msgs, user_id=user_id,
                               metadata={"date": d, "week": week, "source": "dialogue"})
            except Exception as e:
                add_err = repr(e)
                log.error(f"[M0] add dialogue failed week={week} date={d}: {e}")
            finally:
                clear_mem0_observation_date()
                clear_mem0_client_metadata()
            _added_n = len((add_ret or {}).get("results", []) or []) if isinstance(add_ret, dict) else 0
            log.info(f"[M0] story={story_id} week={week} day={d} "
                    f"add_memory source=dialogue msgs={len(msgs)} added={_added_n}"
                    + (f" err={add_err}" if add_err else ""))
            _persist_add(source="dialogue", week=week, date=d, msgs=msgs,
                         ret=add_ret, err=add_err)

        # 2) Answer this week's QA independently; LLM answer requests are parallel.
        week_qa_messages = []   # actual test snippets to be written back as memory after the week
        week_records = []
        qa_jobs = []
        for q in questions_by_week.get(week, []):
            qid = q["question_id"]
            try:
                res = mem0_safe_call(client.search, q["question"],
                                     filters={"user_id": user_id}, top_k=10)
                items = res.get("results", []) if isinstance(res, dict) else []
            except Exception as e:
                log.error(f"[M0] search failed: {e}")
                items = []
            retrieved = [
                (it.get("memory", ""), it.get("score", 0.0), it.get("id", ""), it.get("metadata") or {})
                for it in items
            ]
            append_jsonl(recall_path, {
                "story_id": story_id, "week": week, "question_id": qid,
                "method": "M0", "fbmode": fbmode,
                "memory_model": get_memory_model_alias(),
                "query": q["question"],
                "target_memory": q.get("target_memory", ""),
                "target_memory_refs": q.get("target_memory_refs", []),
                "retrieved": [
                    {
                        "rank": idx + 1,
                        "id": str(mid),
                        "text": t,
                        "score": round(s, 4) if isinstance(s, (int, float)) else s,
                        "source_day": (meta or {}).get("date"),
                        "source_week": (meta or {}).get("week"),
                    }
                    for idx, (t, s, mid, meta) in enumerate(retrieved)
                ],
            })
            qa_jobs.append((len(qa_jobs), q, retrieved))

        if qa_jobs:
            max_workers = min(len(qa_jobs), max(1, int(os.environ.get("ADAMEM_WEEK_QA_PARALLEL", "8"))))
            results_by_idx = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(_answer_one_qa, q, retrieved): idx
                    for idx, q, retrieved in qa_jobs
                }
                for fut in as_completed(futs):
                    idx = futs[fut]
                    results_by_idx[idx] = fut.result()
            for idx in sorted(results_by_idx):
                rec, test_messages = results_by_idx[idx]
                qa_records.append(rec)
                week_records.append(rec)
                week_qa_messages.extend(test_messages)
                qa_turns_global.extend([
                    {**m, "week": week, "question_id": rec["question_id"]}
                    for m in test_messages
                ])

        # 3) Write back this week's actual QA test snippets as memory.
        if week_qa_messages:
            last_date = sorted(by_date.keys())[-1] if by_date else f"w{week}"
            set_mem0_observation_date(last_date)
            set_mem0_client_metadata({
                "tag": make_memory_tag(story_id, week, f"{last_date}_qa_writeback"),
                "story_id": story_id,
                "week": week,
                "kind": "memory_add",
                "source": "qa_writeback",
            })
            add_ret = None
            add_err = None
            try:
                add_ret = mem0_safe_call(client.add, week_qa_messages, user_id=user_id,
                               metadata={"date": last_date, "week": week,
                                         "source": "qa_writeback"})
            except Exception as e:
                add_err = repr(e)
                log.error(f"[M0] QA-writeback failed week={week}: {e}")
            finally:
                clear_mem0_observation_date()
                clear_mem0_client_metadata()
            _added_n = len((add_ret or {}).get("results", []) or []) if isinstance(add_ret, dict) else 0
            log.info(f"[M0] story={story_id} week={week} day={last_date} "
                    f"add_memory source=qa_writeback msgs={len(week_qa_messages)} added={_added_n}"
                    + (f" err={add_err}" if add_err else ""))
            _persist_add(source="qa_writeback", week=week, date=last_date,
                         msgs=week_qa_messages, ret=add_ret, err=add_err)

        print(f"W{week}", end=" ", flush=True)

    acc = sum(1 for r in qa_records if r["correct"]) / max(len(qa_records), 1)
    print(f"= {acc*100:.0f}%")
    close_mem0(client)
    del client
    gc.collect()

    out = {
        "method": "M0",
        "story_id": story_id,
        "fbmode": fbmode,
        "memory_model": get_memory_model_alias(),
        "memory_model_wire": MEMORY_MODEL,
        "accuracy": acc,
        "qa_records": qa_records,
        "qa_turns": qa_turns_global,
        "token_usage": get_token_usage_snapshot(),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    mark_paper_cell_done(out_dir, method="M0", story_id=story_id, fbmode=fbmode)
    return out


def _run_subprocess(story_id, num_weeks, fbmode, memory_model, out_root, exp_tag):
    env = os.environ.copy()
    if memory_model:
        env['ADAMEM_MEMORY_MODEL'] = memory_model
    if exp_tag:
        env['ADAMEM_EXP_TAG'] = exp_tag
    rc = subprocess.run(
        [sys.executable, '-u', __file__,
         '--story', str(story_id),
         '--weeks', str(num_weeks),
         '--fbmode', fbmode,
         '--memory-model', memory_model,
         '--out-dir', out_root,
         '--exp-tag', exp_tag,
         '--no-parallel'],
        env=env, cwd=os.path.dirname(os.path.abspath(__file__)),
    ).returncode
    return story_id, rc


def main():
    parser = argparse.ArgumentParser(description="AdaMem M0 (mem0 baseline) experiment")
    parser.add_argument('--story', default='all', help='Story id (int) or "all"')
    parser.add_argument('--stories', type=int, default=NUM_STORIES_DEFAULT)
    parser.add_argument('--weeks', type=int, default=NUM_WEEKS_DEFAULT)
    parser.add_argument('--fbmode', choices=['with_gold', 'verbose'], required=True)
    parser.add_argument('--memory-model', default=None,
                        help='Vendor alias for MEMORY_MODEL (drives mem0 fact extraction).')
    parser.add_argument('--exp-tag', default=None)
    parser.add_argument('--out-dir', default=None,
                        help='Override the paper-experiment root directory (default: AdaMem/exp/paper).')
    parser.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL)
    parser.add_argument('--no-parallel', action='store_true')
    parser.add_argument('--skip-sanity-check', action='store_true')
    args = parser.parse_args()

    if args.memory_model:
        set_memory_model(args.memory_model)
    if args.exp_tag:
        os.environ['ADAMEM_EXP_TAG'] = args.exp_tag

    print_effective_config({"method": "M0", "fbmode": args.fbmode,
                            "out_root": args.out_dir or "<default exp/paper>"})
    if not args.skip_sanity_check:
        try:
            sanity_check_connectivity()
        except Exception as e:
            print(f"sanity_check FAILED: {e}", file=sys.stderr)
            sys.exit(5)

    def _cell_dir(sid):
        return paper_story_dir(
            "m0", args.fbmode, sid,
            memory_model_alias=get_memory_model_alias(),
            out_root=args.out_dir,
        )

    if args.story != 'all':
        sid = int(args.story)
        reset_token_usage()
        try:
            run_m0_for_story(sid, num_weeks=args.weeks, fbmode=args.fbmode,
                             out_dir=_cell_dir(sid))
            update_progress(lambda p: p.setdefault("stories", {}).setdefault(f"story_{sid}", {}).update({
                f"paper_m0_{args.fbmode}_{get_memory_model_alias()}": "done",
                f"tokens_m0_{args.fbmode}": get_token_usage_snapshot(),
            }))
        except Exception as e:
            print(f"[Story {sid}] M0 error: {e}")
            traceback.print_exc()
        return

    print(f"AdaMem M0 experiment | fbmode={args.fbmode}")
    print(f"  Stories: {args.stories} | Weeks: {args.weeks} | Parallel: {args.parallel}")
    print("=" * 60)
    pending = []
    for sid in range(1, args.stories + 1):
        if is_paper_cell_complete(_cell_dir(sid)):
            print(f"[Story {sid}] M0/{args.fbmode}/{get_memory_model_alias()} ✓")
            continue
        pending.append(sid)
    if not pending:
        print("All stories M0 complete.")
        return
    print(f"Pending: {pending}")
    start = time.time()
    if args.parallel <= 1 or args.no_parallel:
        for sid in pending:
            reset_token_usage()
            try:
                run_m0_for_story(sid, num_weeks=args.weeks, fbmode=args.fbmode,
                                 out_dir=_cell_dir(sid))
            except Exception as e:
                print(f"[Story {sid}] M0 error: {e}")
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(_run_subprocess, sid, args.weeks, args.fbmode,
                          get_memory_model_alias(), args.out_dir or "",
                          os.environ.get('ADAMEM_EXP_TAG', 'scaling')): sid
                for sid in pending
            }
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    sid, rc = fut.result()
                    print(f"[Story {sid}] {'Done ✓' if rc == 0 else f'Failed ✗ (exit {rc})'}")
                except Exception as e:
                    print(f"[Story {sid}] Worker exception: {e}")
    elapsed = (time.time() - start) / 60
    print(f"\nALL DONE in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
