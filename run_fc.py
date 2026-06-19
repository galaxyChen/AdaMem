#!/usr/bin/env python3
"""AdaMem run_fc: Full-Context baseline.

Each test question is evaluated as an independent snippet:
    system: generic answer guide + the full exposed dialogue history
    user:   <Q>
    assistant: <A>
    user:   <feedback(A)>
    assistant: OK

All questions in the same week are requested in parallel and do not share
cross-question chat state. The weekly QA history used for analysis/writeback is
formed by concatenating these actual snippets.

The feedback string follows AdaMem.common.format_feedback (with_gold | verbose).

For the paper matrix, FC is run **independently in every (memory_model,
fbmode) cell** so each cell has its own qa_records / recall.jsonl /
extracted_memories.jsonl artifacts. FC does not run mem0 fact extraction,
so ``extracted_memories.jsonl`` is created empty as a placeholder; the
"recall" written per QA is the entire dialogue context that was actually
fed to the answerer (rank order = chronological).

Usage:
    python run_fc.py --story all --fbmode with_gold
    python run_fc.py --story 1 --fbmode verbose --memory-model deepseek-v4-flash
"""

import argparse, gc, json, os, sys, time, traceback, subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, call_answer_messages, judge_answer, load_progress, update_progress,
    reset_token_usage, get_token_usage_snapshot,
    load_story_bundle, story_exp_dir, format_feedback, format_dialogue_as_text,
    NUM_STORIES_DEFAULT, NUM_WEEKS_DEFAULT, DEFAULT_PARALLEL,
    set_memory_model, get_memory_model_alias, MEMORY_MODEL,
    paper_story_dir, append_jsonl, is_paper_cell_complete,
    make_qa_tag, print_effective_config, sanity_check_connectivity,
    mark_paper_cell_done,
)


def _build_system_prompt(dialogue_blocks, past_qa_blocks=None):
    """Compose the independent FC system prompt for one question.

    ``past_qa_blocks`` is the chronological list of QA history snippets that
    were exposed in *previous* weeks (NEVER includes any QA from the same week
    as the current question). When empty/None, the ``## Past QA Q&A turns``
    section is omitted entirely so the prompt stays clean for early weeks.
    """
    history_text = "\n\n---\n\n".join(dialogue_blocks) if dialogue_blocks else "(no prior dialogue)"
    parts = [
        "You are Linchen's AI assistant. Answer concisely using only the conversation "
        "history below. If the history does not contain the answer, say \"I don't remember.\"\n\n"
        f"## Past dialogue history (week by week):\n{history_text}"
    ]
    if past_qa_blocks:
        qa_text = "\n\n---\n\n".join(past_qa_blocks)
        parts.append(
            "\n\n## Past QA Q&A turns (only previous weeks):\n"
            f"{qa_text}"
        )
    return "".join(parts)


def run_fc_for_story(story_id, *, num_weeks, fbmode, out_dir):
    out_path = os.path.join(out_dir, "qa_records.json")
    extracted_path = os.path.join(out_dir, "extracted_memories.jsonl")
    recall_path = os.path.join(out_dir, "recall.jsonl")
    if is_paper_cell_complete(out_dir):
        print(f"  [FC/{fbmode}] story {story_id} ✓ (cached: {out_dir})")
        with open(out_path) as f:
            return json.load(f)

    for p in (out_path, extracted_path, recall_path, os.path.join(out_dir, "done.json")):
        if os.path.exists(p):
            os.remove(p)
    # FC does no fact extraction; create the empty placeholder so the judge
    # pipeline can iterate every cell uniformly.
    open(extracted_path, "a", encoding="utf-8").close()

    weeks_data, qa_data, ideal_data, profiles = load_story_bundle(story_id, num_weeks)
    questions_by_week = {}
    for q in qa_data["test_questions"]:
        questions_by_week.setdefault(q["test_week"], []).append(q)

    dialogue_blocks = []   # accumulating list[str], one block per session (chronological)
    block_meta = []        # parallel list with (week, session_id, date, day) for recall logging
    past_qa_blocks = []    # QA history exposed by *previous* weeks only
    past_qa_block_meta = []  # parallel list with {week, question_id} per QA block
    qa_turns = []          # actual test snippets concatenated chronologically
    qa_records = []

    # ``frozen_past_qa_blocks`` is rebound once per week BEFORE answering that
    # week's QA, so all questions in the same week see exactly the same
    # ``past_qa_blocks`` snapshot (i.e. <= week-1 QA history). Without this
    # snapshotting, parallel threads in the same week could race against
    # late-arriving appends. We do not append within the same week anyway --
    # that only happens at end-of-week below -- but the snapshot makes the
    # invariant explicit.
    frozen_past_qa_blocks = []

    def _answer_one_qa(q):
        qid = q["question_id"]
        system_prompt = _build_system_prompt(dialogue_blocks, frozen_past_qa_blocks)
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
        log.info(f"[FC] story={story_id} week={week} qa qid={qid} "
                f"ctx_chars={len(system_prompt) + len(q['question'])} correct={ok}")
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
            "context_chars": len(system_prompt) + len(q["question"]),
        }
        return record, test_messages

    print(f"  [FC/{fbmode}] story {story_id} -> {out_dir}: ", end="", flush=True)
    for week in range(1, num_weeks + 1):
        # Append this week's dialogues.
        for conv in weeks_data[week]["conversations"]:
            block = f"[{conv.get('date','')} {conv.get('day','')}]\n"
            block += format_dialogue_as_text(conv.get("messages", []))
            dialogue_blocks.append(block)
            block_meta.append({
                "week": week,
                "session_id": conv.get("session_id", ""),
                "date": conv.get("date", ""),
                "day": conv.get("day", ""),
            })
            log.info(f"[FC] story={story_id} week={week} day={conv.get('date','')} "
                    f"add_memory source=dialogue session={conv.get('session_id','')} "
                    f"msgs={len(conv.get('messages', []))} total_blocks={len(dialogue_blocks)}")

        # Snapshot the QA history that this week's questions are allowed to
        # see (only previous weeks). This guarantees same-week QA do not see
        # each other's test snippets.
        frozen_past_qa_blocks = list(past_qa_blocks)

        qa_jobs = []
        for q in questions_by_week.get(week, []):
            qid = q["question_id"]
            # The retrieved row is the entire context fed to the answerer:
            # dialogue blocks first (rank 1..N), then frozen QA history
            # (rank N+1..N+M). FC stays "--" in the recall matrix downstream;
            # we still log the full context for auditability.
            dialogue_rows = [
                {
                    "rank": idx + 1,
                    "id": f"sess::{m['week']}::{m['session_id']}",
                    "text": dialogue_blocks[idx],
                    "score": None,
                    "source_day": m.get("date"),
                    "session_id": m.get("session_id"),
                    "week": m.get("week"),
                    "kind": "dialogue",
                }
                for idx, m in enumerate(block_meta)
            ]
            qa_rows = [
                {
                    "rank": len(dialogue_rows) + idx + 1,
                    "id": f"qa::{m['week']}::{m['question_id']}",
                    "text": frozen_past_qa_blocks[idx],
                    "score": None,
                    "source_day": None,
                    "session_id": None,
                    "week": m.get("week"),
                    "question_id": m.get("question_id"),
                    "kind": "qa_history",
                }
                for idx, m in enumerate(past_qa_block_meta)
            ]
            append_jsonl(recall_path, {
                "story_id": story_id, "week": week, "question_id": qid,
                "method": "FC", "fbmode": fbmode,
                "target_memory": q.get("target_memory", ""),
                "target_memory_refs": q.get("target_memory_refs", []),
                "retrieved": dialogue_rows + qa_rows,
            })
            qa_jobs.append((len(qa_jobs), q))

        if qa_jobs:
            max_workers = min(len(qa_jobs), max(1, int(os.environ.get("ADAMEM_WEEK_QA_PARALLEL", "8"))))
            results_by_idx = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_answer_one_qa, q): idx for idx, q in qa_jobs}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    results_by_idx[idx] = fut.result()
            # After all questions in this week have answered, append their
            # actual test snippets (Q/A/feedback/OK) to ``past_qa_blocks`` so
            # *future* weeks can see them. Order by question_id for a stable
            # chronological view.
            week_qa_appendix = []
            for idx in sorted(results_by_idx):
                rec, test_messages = results_by_idx[idx]
                qa_records.append(rec)
                qa_turns.extend([
                    {**m, "week": week, "question_id": rec["question_id"]}
                    for m in test_messages
                ])
                week_qa_appendix.append((rec["question_id"], test_messages))
            week_qa_appendix.sort(key=lambda x: x[0])
            for qid, test_messages in week_qa_appendix:
                # Render the 5-message snippet using the same role mapping as
                # ``format_dialogue_as_text`` (Linchen / AI), but skip the
                # system message since its content is mostly the (already
                # exposed) dialogue history. We retain Q, A, feedback, OK.
                lines = [f"[Week {week} QA {qid}]"]
                for m in test_messages:
                    if m["role"] == "system":
                        continue
                    role = "Linchen" if m["role"] == "user" else "AI"
                    lines.append(f"{role}: {m['content']}")
                past_qa_blocks.append("\n".join(lines))
                past_qa_block_meta.append({"week": week, "question_id": qid})
        print(f"W{week}", end=" ", flush=True)

    acc = sum(1 for r in qa_records if r["correct"]) / max(len(qa_records), 1)
    print(f"= {acc*100:.0f}%")

    out = {
        "method": "FC",
        "story_id": story_id,
        "fbmode": fbmode,
        "memory_model": get_memory_model_alias(),
        "memory_model_wire": MEMORY_MODEL,
        "accuracy": acc,
        "qa_records": qa_records,
        "qa_turns": qa_turns,
        "token_usage": get_token_usage_snapshot(),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    mark_paper_cell_done(out_dir, method="FC", story_id=story_id, fbmode=fbmode)
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
    parser = argparse.ArgumentParser(description="AdaMem FC (Full-Context) experiment")
    parser.add_argument('--story', default='all', help='Story id (int) or "all"')
    parser.add_argument('--stories', type=int, default=NUM_STORIES_DEFAULT)
    parser.add_argument('--weeks', type=int, default=NUM_WEEKS_DEFAULT)
    parser.add_argument('--fbmode', choices=['with_gold', 'verbose'], required=True)
    parser.add_argument('--memory-model', default=None,
                        help='Vendor alias for MEMORY_MODEL (FC does not actually use it for '
                             'extraction, but each cell still keeps independent artifacts).')
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

    print_effective_config({"method": "FC", "fbmode": args.fbmode,
                            "out_root": args.out_dir or "<default exp/paper>"})
    if not args.skip_sanity_check:
        try:
            sanity_check_connectivity()
        except Exception as e:
            print(f"sanity_check FAILED: {e}", file=sys.stderr)
            sys.exit(5)

    def _cell_dir(sid):
        return paper_story_dir(
            "fc", args.fbmode, sid,
            memory_model_alias=get_memory_model_alias(),
            out_root=args.out_dir,
        )

    if args.story != 'all':
        sid = int(args.story)
        reset_token_usage()
        try:
            run_fc_for_story(sid, num_weeks=args.weeks, fbmode=args.fbmode,
                             out_dir=_cell_dir(sid))
            update_progress(lambda p: p.setdefault("stories", {}).setdefault(f"story_{sid}", {}).update({
                f"paper_fc_{args.fbmode}_{get_memory_model_alias()}": "done",
                f"tokens_fc_{args.fbmode}": get_token_usage_snapshot(),
            }))
        except Exception as e:
            print(f"[Story {sid}] FC error: {e}")
            traceback.print_exc()
        return

    print(f"AdaMem FC experiment | fbmode={args.fbmode}")
    print(f"  Stories: {args.stories} | Weeks: {args.weeks} | Parallel: {args.parallel}")
    print("=" * 60)
    pending = []
    for sid in range(1, args.stories + 1):
        if is_paper_cell_complete(_cell_dir(sid)):
            print(f"[Story {sid}] FC/{args.fbmode}/{get_memory_model_alias()} ✓")
            continue
        pending.append(sid)
    if not pending:
        print("All stories FC complete.")
        return
    print(f"Pending: {pending}")
    start = time.time()
    if args.parallel <= 1 or args.no_parallel:
        for sid in pending:
            reset_token_usage()
            try:
                run_fc_for_story(sid, num_weeks=args.weeks, fbmode=args.fbmode,
                                 out_dir=_cell_dir(sid))
            except Exception as e:
                print(f"[Story {sid}] FC error: {e}")
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
