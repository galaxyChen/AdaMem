#!/usr/bin/env python3
"""AdaMem run_adamem: AdaMem (Adaptive Memory) method.

Core idea
---------
Maintain a `policy` describing what to focus on when extracting memory:

    policy = {
        "general_policy": "<generic, non-character-specific rules, may stay empty>",
        "by_character":   { "<character name>": "<focused-extraction rule>", ... }
    }

Per-week loop (aligned with M0 on writeback timing)
---------------------------------------------------
For each week N:

  1. WRITE week N's dialogues into mem0 using the policy reflected at the
     end of week N-1 as `custom_instructions` (the non-empty default general
     policy is used before the first reflection).

  2. ANSWER week N's QA via independent per-question snippets. Each question
     sees only a system prompt containing the generic answer guide plus its own
     retrieved memories, then a user turn containing that question. Questions in
     the same week are requested in parallel and do not share chat state.

  3. REFLECT on (old_policy, week N's actual QA test snippets) and obtain
     new_policy. The reflection prompt sees the same surface history that is
     later written back: system, user Q, assistant A, user feedback, assistant OK.

  4. WRITE this week's actual QA test snippets into mem0 using `new_policy` as
     `custom_instructions`. This is the same writeback phase M0 uses.

Answering uses ONLY mem0.search (same as M0). Cross-question history is
not re-fed via in-context; the policy + memory are the only state.

Policy initialization: NOT empty. We start with a non-empty
`general_policy` (DEFAULT_GENERAL_POLICY) so the fact-extractor LLM has
a strong baseline focus rule even on week 1, before any reflect step.
`by_character` still starts empty and is populated only when reflection
sees clear evidence to add a per-person rule.

Usage:
    python run_adamem.py --story all --fbmode with_gold
    python run_adamem.py --story 1 --fbmode verbose
"""

import argparse, copy, gc, json, os, sys, time, traceback, subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, call_answer_messages, call_reflect, judge_answer, parse_json_from_llm,
    load_progress, update_progress, reset_token_usage, get_token_usage_snapshot,
    build_mem0_client, mem0_safe_call, close_mem0, purge_mem0_collection,
    set_mem0_observation_date, clear_mem0_observation_date,
    set_mem0_custom_instructions, clear_mem0_custom_instructions,
    set_mem0_client_metadata, clear_mem0_client_metadata,
    load_story_bundle, story_exp_dir, format_feedback,
    NUM_STORIES_DEFAULT, NUM_WEEKS_DEFAULT, DEFAULT_PARALLEL,
    set_memory_model, get_memory_model_alias, MEMORY_MODEL,
    paper_story_dir, qdrant_collection_name, append_jsonl,
    is_paper_cell_complete,
    make_qa_tag, make_memory_tag, make_policy_tag,
    print_effective_config, sanity_check_connectivity,
    mark_paper_cell_done,
)


REFLECT_PROMPT = """You are an editor for adaptive memory strategies.

You are responsible for maintaining a JSON-formatted policy that guides a
downstream memory-extraction LLM -- specifically, determining which details
to prioritize when extracting facts. The policy consists of two parts:
  - "general_policy": a brief, universal rule (string).
  - "by_character":   rules specific to certain characters (object: Name -> brief rule).

## Current policy (for reference only -- do NOT output as-is)
```json
{old_policy}
```

## Recent context (one week of Q&A logs, formatted as ordinary dialogue)
The transcript below is a continuous dialogue between the user and the AI
assistant. Each of the user's turns may be either a new question or a piece
of natural-language feedback responding to the AI's previous answer
("You are right." / "You are wrong, ..."). We do NOT tell you which person
each question is about -- infer it from the text.

{qa_history}

## Your job: emit a PATCH, not the full policy

Return a JSON object describing ONLY the changes required. Unmentioned
parts will remain unchanged.

Schema (every field is optional; emit `{{}}` if no changes are needed):
  {{
    "general_policy": "<new value>"   // optional. omit to keep current.
                                       // "" means clear it.
    "set":    {{ "Name": "new rule", ... }} // optional. add/overwrite.
                                            // To drop a character's rule,
                                            // set its value to "".
  }}

Policy update requirements:
- When the user's feedback explicitly requests focus on specific content,
  record that requirement. If the current policy does not yet include the
  relevant character, add both the character and the corresponding
  extraction preference.
- When the AI assistant answers a question incorrectly, you MUST learn
  from the mistake and add a relevant memory preference so the missing
  information is captured correctly next time.
- When the user's question is answered correctly, that confirms the user
  cares about this topic for that character. You SHOULD add or refine a
  rule for that character so future weeks keep capturing the same kind of
  fact. Treat each character that appeared in any question this week
  (whether the AI answered right or wrong) as a candidate that probably
  deserves a `by_character` entry; only skip a character if there is no
  identifiable topic of interest for them yet.

Phrasing requirements (CRITICAL -- the downstream LLM follows your wording
literally):
- Phrase every rule as a POSITIVE PRIORITY, not an exclusive filter.
  Use verbs like "pay attention to", "capture", "track", "remember",
  "note". Do NOT use "only", "just", "exclusively", "ignore", "skip",
  "not", "rather than", "instead of". Those words make the downstream
  extractor drop legitimate facts.
  - Bad:  "Remember only Boss Zhang's final decisions, skip the discussion."
  - Good: "Pay extra attention to Boss Zhang's final decisions and
           conclusions."
  - Bad:  "Track Liwei's promises, not casual chatter."
  - Good: "Capture Liwei's promises and how she is feeling."
- Each rule must be a short positive sentence (one clause is fine).

Output rules:
- Output ONLY the JSON object. No markdown code fences. No explanatory text.
- Use the exact names as they appear in the questions/feedback
  (e.g. "Boss Zhang", not "boss").
- If absolutely nothing in this week is worth recording, return `{{}}`.
- Do NOT repeat existing rules just to preserve them. Existing rules are
  automatically retained unless explicitly cleared (set value to "").

Examples
--------
* No changes:                     {{}}
* New rule for Liwei only:        {{"set": {{"Liwei": "Capture Liwei's commitments and emotional reactions."}}}}
* Two characters this week:       {{"set": {{"Boss Zhang": "Pay attention to Boss Zhang's final decisions and conclusions.", "Wanghao": "Capture Wanghao's technical conclusions and key numbers."}}}}
* Clear Wanghao's rule + update general:
                                  {{"general_policy": "Focus on dates and decisions.", "set": {{"Wanghao": ""}}}}

Now, return only the JSON object:
"""


# Default general policy used when reflection has not yet produced a
# specialized one. Per the new design, policy is NOT initialized empty:
# we always feed mem0 a baseline focus rule so the fact-extractor has a
# strong directive even on week 1 (before any reflect step).
DEFAULT_GENERAL_POLICY = (
    "Track the user's relationships with each named person -- their stated "
    "preferences, promises (made or received), recurring topics, and "
    "emotional dynamics. Always attribute facts to the specific person "
    "involved."
)


def _empty_policy():
    return {"general_policy": DEFAULT_GENERAL_POLICY, "by_character": {}}


def _format_qa_history_for_reflect(week, week_records, fbmode, profiles):
    """Compose the actual weekly QA test snippets as dialogue history."""
    lines = [f"Week {week} actual QA test snippets:"]
    for r in week_records:
        qid = r.get("question_id", "")
        lines.append(f"\n### {qid}")
        for m in r.get("test_messages", []) or []:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                role_name = "System"
            elif role == "assistant":
                role_name = "AI"
            else:
                role_name = "User"
            lines.append(f"{role_name}: {content}")
    return "\n".join(lines)


def _render_policy_as_instructions(policy):
    """Convert the policy dict into the natural-language string that mem0
    will splice into its fact-extraction system prompt as `custom_instructions`.

    Uses the V3d "preference-list" wording. The per-character rules are
    rendered as a Preference List that the extractor LLM should treat as
    the user's stated focus for those individuals; for anyone NOT in the
    list, the LLM falls back to the default extraction method.

    Layout rules:
      * If `by_character` is non-empty, the Preference List enumerates
        each character with its focus rule.
      * If `by_character` is empty, the Preference List is rendered as
        an explicit "(none)" so the LLM knows to use the default method
        for everyone.
      * `general_policy` is intentionally not surfaced in the rendered
        custom_instructions: empirically the V3d framing is more reliable
        when it stays focused on the per-character preferences and the
        default-extraction fallback. The general_policy is still tracked
        in the policy dict and continues to be visible to the reflect
        LLM as part of the policy state.
    """
    by = {k: (v or "").strip()
          for k, v in (policy.get("by_character") or {}).items()
          if isinstance(v, str) and (v or "").strip()}

    if by:
        pref_lines = ["Preference List:"]
        for char in sorted(by.keys()):
            pref_lines.append(f"- {char}: {by[char]}")
        pref_block = "\n".join(pref_lines)
    else:
        pref_block = (
            "Preference List: (none -- use the default extraction "
            "method for everyone mentioned in the chat history.)"
        )

    body = (
        "Users have specific extraction preferences regarding certain "
        "individuals. If a specific person appears in the chat history "
        "(identified by name or inferred from context) and is included "
        "in the preference list below, priority must be given to "
        "identifying information of interest to the user and extracting "
        "the corresponding memories in accordance with the preference "
        "list's requirements.\n"
        "For individuals subject to specific extraction preferences, "
        "ensure that:\n"
        "- Content of interest to the user (items explicitly designated "
        "for recording in the preference list) is correctly extracted.\n"
        "- Other information about the same individual that is purely "
        "about that individual and unrelated to the user's life, "
        "schedule, decisions, plans, commitments, or preferences may "
        "be excluded. However, any fact whose substance concerns the "
        "user (e.g. a recurring schedule the user follows, a plan or "
        "promise the user is part of, a preference the user holds) "
        "must still be extracted, even if the sentence's grammatical "
        "subject is that other individual.\n"
        "Note: If an individual mentioned in the chat history does not "
        "appear in the preference list, proceed with extraction using "
        "the default method, ensuring relevant information is not "
        "overlooked."
    )

    # GLOBAL TIME RULE -- applies to every extracted memory regardless of
    # whether the speaker / subject is in the preference list. Anchored to
    # the `## Observation Date` section already present in mem0's user
    # prompt, so relative phrases like "yesterday" / "next Wednesday" can
    # always be resolved to a concrete date.
    time_rule = (
        "Time Anchoring (MANDATORY for every extracted memory):\n"
        "- Treat the `## Observation Date` value provided in the user "
        "prompt as 'today' for this conversation. All relative time "
        "expressions (e.g. 'today', 'yesterday', 'this morning', "
        "'last Friday', 'next Wednesday', 'tomorrow', 'this weekend') "
        "MUST be resolved into a concrete absolute date (YYYY-MM-DD) "
        "or, when a specific clock time is mentioned, an absolute "
        "datetime (e.g. 'YYYY-MM-DD HH:MM').\n"
        "- Every extracted memory MUST explicitly carry a time anchor "
        "in its text. Acceptable forms include: a specific date "
        "('on 2025-07-21'), a date range ('from 2025-07-21 to "
        "2025-07-23'), a recurring cadence with an effective date "
        "('every Thursday, as of 2025-07-24'), or a future commitment "
        "('scheduled for 2025-07-30').\n"
        "- If the conversation gives an explicit absolute date, use "
        "that date verbatim. Otherwise compute the absolute date from "
        "the `## Observation Date` value. Do NOT leave the time as a "
        "bare relative expression ('recently', 'soon', 'lately', "
        "'the other day') -- always resolve it.\n"
        "- If, after careful reading, no time anchor can be inferred "
        "for a candidate fact, prefer NOT to extract it as a standalone "
        "memory; instead fold it into another memory that does have a "
        "time anchor, or omit it."
    )

    return f"{time_rule}\n\n{body}\n\n{pref_block}".strip()


def _apply_policy_patch(old_policy, patch):
    """Apply a {general_policy?, set?, remove?} patch to old_policy and return
    a NEW policy dict. Unmentioned fields/characters are kept verbatim.

    Backwards compat: if `patch` instead looks like a full policy
    ({general_policy, by_character}), treat it as a full overwrite (matches
    the legacy schema the LLM occasionally still emits).
    """
    new_policy = copy.deepcopy(old_policy)
    if not isinstance(patch, dict):
        return None

    # Legacy full-policy fallback (no patch keys present, but has by_character).
    legacy_keys = {"general_policy", "by_character"}
    patch_keys = {"general_policy", "set", "remove"}
    if ("by_character" in patch and not (set(patch.keys()) & {"set", "remove"})):
        # full overwrite -- but if the LLM did not include general_policy,
        # keep the existing one (which is non-empty by default) instead of
        # silently clearing it to "".
        if "general_policy" in patch and isinstance(patch.get("general_policy"), str):
            gp = patch["general_policy"].strip()
        else:
            gp = old_policy.get("general_policy", "")
        return {
            "general_policy": gp,
            "by_character": {str(k): str(v).strip()
                             for k, v in (patch.get("by_character") or {}).items()
                             if isinstance(v, str)},
        }

    # Patch path. All fields optional -- {} is the legitimate "no-op".
    has_any_patch_key = bool(set(patch.keys()) & patch_keys) or not patch
    if not has_any_patch_key:
        # unknown shape -> caller treats as bad_shape
        return None

    if "general_policy" in patch:
        gp = patch.get("general_policy")
        if isinstance(gp, str):
            new_policy["general_policy"] = gp.strip()
        # if not string (e.g. null), keep current

    by = new_policy.setdefault("by_character", {})
    for k, v in (patch.get("set") or {}).items():
        if isinstance(v, str):
            by[str(k)] = v.strip()
    for k in (patch.get("remove") or []):
        by.pop(str(k), None)
    return new_policy

def _update_policy(old_policy, week, week_records, fbmode, profiles, *,
                   story_id=None):
    """Reflect on this week's QA records and return (new_policy, meta).

    The reflector is asked to emit a PATCH (only what should change). On
    any LLM/parse/shape error, return old_policy unchanged. ``meta`` carries
    the reflect_input_qa_history (rendered transcript), the reflect_raw_output
    (the LLM's raw text), the parsed patch, and any error string -- callers
    persist this verbatim into ``policy_snapshots.json``.
    """
    if not week_records:
        return copy.deepcopy(old_policy), {
            "skipped": "no qa records",
            "reflect_input_qa_history": "",
            "reflect_raw_output": "",
            "patch": None,
        }
    qa_history = _format_qa_history_for_reflect(week, week_records, fbmode, profiles)
    prompt = REFLECT_PROMPT.format(
        old_policy=json.dumps(old_policy, ensure_ascii=False, indent=2),
        qa_history=qa_history,
    )
    raw = call_reflect(
        prompt, max_tokens=8192, temperature=0,
        force_json=True,
        tag=make_policy_tag(story_id, week) if story_id is not None else None,
        story_id=story_id, week=week, kind="policy_update",
    )
    parsed = None
    try:
        parsed = parse_json_from_llm(raw or "")
    except Exception as e:
        log.warning(f"[AdaMem] reflect parse failed: {e}")
    new_policy = _apply_policy_patch(old_policy, parsed) if parsed is not None else None
    if new_policy is None:
        log.warning(f"[AdaMem] reflect bad shape, keeping old policy: {raw!r}")
        return copy.deepcopy(old_policy), {
            "reflect_input_qa_history": qa_history,
            "reflect_raw_output": raw or "",
            "patch": parsed,
            "error": "bad_shape",
        }
    return new_policy, {
        "reflect_input_qa_history": qa_history,
        "reflect_raw_output": raw or "",
        "patch": parsed,
    }


def run_adamem_for_story(story_id, *, num_weeks, fbmode, out_dir):
    out_path = os.path.join(out_dir, "qa_records.json")
    extracted_path = os.path.join(out_dir, "extracted_memories.jsonl")
    recall_path = os.path.join(out_dir, "recall.jsonl")
    policy_history_path = os.path.join(out_dir, "policy_snapshots.json")
    if is_paper_cell_complete(out_dir):
        print(f"  [AdaMem/{fbmode}] story {story_id} ✓ (cached: {out_dir})")
        with open(out_path) as f:
            return json.load(f)

    for p in (out_path, extracted_path, recall_path, policy_history_path, os.path.join(out_dir, "done.json")):
        if os.path.exists(p):
            os.remove(p)

    weeks_data, qa_data, ideal_data, profiles = load_story_bundle(story_id, num_weeks)
    questions_by_week = {}
    for q in qa_data["test_questions"]:
        questions_by_week.setdefault(q["test_week"], []).append(q)

    user_id = qdrant_collection_name("adamem", fbmode, story_id)
    collection = user_id
    # P0: hard-wipe the qdrant collection on disk BEFORE building the client.
    # delete_all alone leaks vectors across runs (see debug note 2026-06-17);
    # we now refuse to start the run if the wipe fails.
    purge_mem0_collection(collection)
    client = build_mem0_client(collection)

    policy = _empty_policy()
    policy_history = [{"week": 0, "phase": "init", "policy": copy.deepcopy(policy)}]

    qa_records = []
    qa_turns_global = []

    def _persist_add(*, source, week, date, msgs, ret, err, custom_instructions):
        results = []
        if isinstance(ret, dict):
            results = ret.get("results", []) or []
        append_jsonl(extracted_path, {
            "story_id": story_id,
            "week": week,
            "date": date,
            "source": source,
            "method": "AdaMem",
            "fbmode": fbmode,
            "memory_model": get_memory_model_alias(),
            "custom_instructions": custom_instructions or "",
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
        log.info(f"[AdaMem] story={story_id} week={week} qa qid={qid} "
                f"retrieved={len(retrieved)} correct={ok}")
        feedback = format_feedback(q, ok, fbmode=fbmode, character_profiles=profiles)
        test_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q["question"]},
            {"role": "assistant", "content": ai},
            {"role": "user", "content": feedback},
            {"role": "assistant", "content": "OK"},
        ]
        rec = {
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
        return rec, test_messages

    print(f"  [AdaMem/{fbmode}] story {story_id} -> {out_dir}: ", end="", flush=True)
    for week in range(1, num_weeks + 1):
        # ----------------------------------------------------------------
        # 1) WRITE week N's dialogues into mem0 BEFORE answering week N's QA.
        #    The policy used here is the one reflected at the end of week N-1
        #    (the DEFAULT general policy on week 1 -- AdaMem is no longer
        #    initialized empty, see _empty_policy). The dialogue itself is
        #    contemporaneous (model isn't forced to answer week N's QA
        #    without ever having seen week N).
        # ----------------------------------------------------------------
        old_instructions = _render_policy_as_instructions(policy)

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
            if old_instructions:
                set_mem0_custom_instructions(old_instructions)
            add_ret = None
            add_err = None
            try:
                add_ret = mem0_safe_call(client.add, msgs, user_id=user_id,
                               metadata={"date": d, "week": week, "source": "dialogue"})
            except Exception as e:
                add_err = repr(e)
                log.error(f"[AdaMem] add dialogue failed week={week} date={d}: {e}")
            finally:
                clear_mem0_observation_date()
                clear_mem0_custom_instructions()
                clear_mem0_client_metadata()
            _added_n = len((add_ret or {}).get("results", []) or []) if isinstance(add_ret, dict) else 0
            log.info(f"[AdaMem] story={story_id} week={week} day={d} "
                    f"add_memory source=dialogue msgs={len(msgs)} added={_added_n}"
                    + (f" err={add_err}" if add_err else ""))
            _persist_add(source="dialogue", week=week, date=d, msgs=msgs,
                         ret=add_ret, err=add_err,
                         custom_instructions=old_instructions)

        # ----------------------------------------------------------------
        # 2) ANSWER this week's QA via independent mem0.search snippets.
        # ----------------------------------------------------------------
        week_records = []
        qa_jobs = []
        for q in questions_by_week.get(week, []):
            qid = q["question_id"]
            try:
                res = mem0_safe_call(client.search, q["question"],
                                     filters={"user_id": user_id}, top_k=10)
                items = res.get("results", []) if isinstance(res, dict) else []
            except Exception as e:
                log.error(f"[AdaMem] search failed: {e}")
                items = []
            retrieved = [
                (it.get("memory", ""), it.get("score", 0.0), it.get("id", ""), it.get("metadata") or {})
                for it in items
            ]
            append_jsonl(recall_path, {
                "story_id": story_id, "week": week, "question_id": qid,
                "method": "AdaMem", "fbmode": fbmode,
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
                qa_turns_global.extend([
                    {**m, "week": week, "question_id": rec["question_id"]}
                    for m in test_messages
                ])

        # ----------------------------------------------------------------
        # 3) REFLECT on this week's actual QA test snippets to update policy.
        #    If reflect/parse fails after all retries, _update_policy
        #    returns the old policy unchanged.
        # ----------------------------------------------------------------
        new_policy, reflect_meta = _update_policy(
            policy, week, week_records, fbmode, profiles, story_id=story_id,
        )
        _changed = (new_policy != policy)
        log.info(f"[AdaMem] story={story_id} week={week} policy_update "
                f"changed={_changed} qa_count={len(week_records)}"
                + (f" err={reflect_meta.get('error')}" if reflect_meta.get('error') else ""))
        policy_history.append({
            "week": week,
            "phase": "after_reflect",
            "old_policy": copy.deepcopy(policy),
            "new_policy": copy.deepcopy(new_policy),
            "reflect_input_qa_history": reflect_meta.get("reflect_input_qa_history", ""),
            "reflect_raw_output": reflect_meta.get("reflect_raw_output", ""),
            "patch": reflect_meta.get("patch"),
            "error": reflect_meta.get("error"),
        })
        policy = new_policy

        # ----------------------------------------------------------------
        # 4) WRITE this week's actual QA test snippets into mem0 using the NEW policy
        #    as custom_instructions. Aligns with M0's "writeback at week-end"
        #    timing; the only AdaMem-specific knob is the policy-driven
        #    custom_instructions on mem0.add.
        # ----------------------------------------------------------------
        if week_records:
            new_instructions = _render_policy_as_instructions(policy)
            qa_messages = []
            for r in week_records:
                qa_messages.extend(r.get("test_messages", []) or [])
            wb_date = sorted(by_date.keys())[-1] if by_date else f"w{week}"
            set_mem0_observation_date(wb_date)
            set_mem0_client_metadata({
                "tag": make_memory_tag(story_id, week, f"{wb_date}_qa_writeback"),
                "story_id": story_id,
                "week": week,
                "kind": "memory_add",
                "source": "qa_writeback",
            })
            if new_instructions:
                set_mem0_custom_instructions(new_instructions)
            add_ret = None
            add_err = None
            try:
                add_ret = mem0_safe_call(client.add, qa_messages, user_id=user_id,
                               metadata={"date": wb_date, "week": week,
                                         "source": "qa_writeback"})
            except Exception as e:
                add_err = repr(e)
                log.error(f"[AdaMem] QA-writeback failed week={week}: {e}")
            finally:
                clear_mem0_observation_date()
                clear_mem0_custom_instructions()
                clear_mem0_client_metadata()
            _added_n = len((add_ret or {}).get("results", []) or []) if isinstance(add_ret, dict) else 0
            log.info(f"[AdaMem] story={story_id} week={week} day={wb_date} "
                    f"add_memory source=qa_writeback msgs={len(qa_messages)} added={_added_n}"
                    + (f" err={add_err}" if add_err else ""))
            _persist_add(source="qa_writeback", week=week, date=wb_date,
                         msgs=qa_messages, ret=add_ret, err=add_err,
                         custom_instructions=new_instructions)

        print(f"W{week}", end=" ", flush=True)

    acc = sum(1 for r in qa_records if r["correct"]) / max(len(qa_records), 1)
    print(f"= {acc*100:.0f}%")
    close_mem0(client)
    del client
    gc.collect()

    out = {
        "method": "AdaMem",
        "story_id": story_id,
        "fbmode": fbmode,
        "memory_model": get_memory_model_alias(),
        "memory_model_wire": MEMORY_MODEL,
        "accuracy": acc,
        "qa_records": qa_records,
        "qa_turns": qa_turns_global,
        "final_policy": policy,
        "token_usage": get_token_usage_snapshot(),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(policy_history_path, "w") as f:
        json.dump({
            "method": "AdaMem",
            "story_id": story_id,
            "fbmode": fbmode,
            "memory_model": get_memory_model_alias(),
            "history": policy_history,
            "final_policy": policy,
        }, f, ensure_ascii=False, indent=2)
    mark_paper_cell_done(out_dir, method="AdaMem", story_id=story_id, fbmode=fbmode,
                         extra={"artifacts": [
                             "qa_records.json",
                             "extracted_memories.jsonl",
                             "recall.jsonl",
                             "policy_snapshots.json",
                         ]})
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
    parser = argparse.ArgumentParser(description="AdaMem (Adaptive Memory) experiment")
    parser.add_argument('--story', default='all', help='Story id (int) or "all"')
    parser.add_argument('--stories', type=int, default=NUM_STORIES_DEFAULT)
    parser.add_argument('--weeks', type=int, default=NUM_WEEKS_DEFAULT)
    parser.add_argument('--fbmode', choices=['with_gold', 'verbose'], required=True)
    parser.add_argument('--memory-model', default=None,
                        help='Vendor alias for MEMORY_MODEL (drives mem0 fact extraction '
                             'and AdaMem policy reflection).')
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

    print_effective_config({"method": "AdaMem", "fbmode": args.fbmode,
                            "out_root": args.out_dir or "<default exp/paper>"})
    if not args.skip_sanity_check:
        try:
            sanity_check_connectivity()
        except Exception as e:
            print(f"sanity_check FAILED: {e}", file=sys.stderr)
            sys.exit(5)

    def _cell_dir(sid):
        return paper_story_dir(
            "adamem", args.fbmode, sid,
            memory_model_alias=get_memory_model_alias(),
            out_root=args.out_dir,
        )

    if args.story != 'all':
        sid = int(args.story)
        reset_token_usage()
        try:
            run_adamem_for_story(sid, num_weeks=args.weeks, fbmode=args.fbmode,
                                 out_dir=_cell_dir(sid))
            update_progress(lambda p: p.setdefault("stories", {}).setdefault(f"story_{sid}", {}).update({
                f"paper_adamem_{args.fbmode}_{get_memory_model_alias()}": "done",
                f"tokens_adamem_{args.fbmode}": get_token_usage_snapshot(),
            }))
        except Exception as e:
            print(f"[Story {sid}] AdaMem error: {e}")
            traceback.print_exc()
        return

    print(f"AdaMem experiment | fbmode={args.fbmode}")
    print(f"  Stories: {args.stories} | Weeks: {args.weeks} | Parallel: {args.parallel}")
    print("=" * 60)
    pending = []
    for sid in range(1, args.stories + 1):
        if is_paper_cell_complete(_cell_dir(sid)):
            print(f"[Story {sid}] AdaMem/{args.fbmode}/{get_memory_model_alias()} ✓")
            continue
        pending.append(sid)
    if not pending:
        print("All stories AdaMem complete.")
        return
    print(f"Pending: {pending}")
    start = time.time()
    if args.parallel <= 1 or args.no_parallel:
        for sid in pending:
            reset_token_usage()
            try:
                run_adamem_for_story(sid, num_weeks=args.weeks, fbmode=args.fbmode,
                                     out_dir=_cell_dir(sid))
            except Exception as e:
                print(f"[Story {sid}] AdaMem error: {e}")
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
