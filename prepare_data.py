#!/usr/bin/env python3
"""AdaMem data preparation -- outline-first pipeline.

Pipeline (per story, fully idempotent at file level):

  Step 1  preference_feedback x 6          -> character_profiles.json
  Step 2  outline x N                       -> outline.json
            (week w sees ALL key_info from weeks 1..w-1)
  Step 3  (dialogue + golden_memories) x (N x S)
                                            -> week<W>.json
            (session i sees ALL key_info / golden_memories so far)
  Step 4  (QA + golden_feedback) x N        -> test_qa.json
            (each week's QA is generated only from THAT week's
             golden_memories, with target_memory_refs strictly aligned
             to (week, session_id, index))

CLI sub-modes (semantics preserved):
  * default       -- full pipeline; idempotent.
  * --validate    -- schema/ref-integrity check; no LLM calls.
  * --backfill-golden -- legacy patcher for old datasets that were generated
                         under the previous "scenarios + post-hoc GM" flow.
"""

import argparse, json, os, sys, time, traceback, subprocess, math
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    log, call_gen, parse_json_from_llm,
    load_progress, save_progress, update_progress, reset_token_usage,
    get_token_usage_snapshot, story_data_dir, DATA_DIR,
    PREFERENCES, STORY_THEMES, ANCHOR_POOL,
    NUM_STORIES_DEFAULT, NUM_WEEKS_DEFAULT, CONVS_PER_WEEK_DEFAULT,
    QA_PER_WEEK_DEFAULT, DEFAULT_PARALLEL, MAX_RETRY,
    format_dialogue_as_text,
)

# Hard fail-fast: if anyone ever reduces ANCHOR_POOL below threshold or empties
# it, abort before issuing any LLM call.
if not (
    isinstance(ANCHOR_POOL, list)
    and len(ANCHOR_POOL) >= 30
    and all(isinstance(a, str) and a.strip() for a in ANCHOR_POOL)
):
    raise RuntimeError(
        "ANCHOR_POOL is missing or invalid; expected >= 30 non-empty strings."
    )


# ============================================================================
# PROMPTS
# ============================================================================

PREFERENCE_FEEDBACK_PROMPT = """You are helping label what a user (Linchen) would naturally say to his AI assistant
to nudge the assistant's memory focus for ONE specific person in his life.

## The person
Name: {character}
Linchen's stated memory preference for this person: {preference}

## Story background
{theme}

## Task
Write ONE short, natural sentence in spoken English that Linchen would say to his AI
to express the above preference -- as feedback after the AI got something wrong about
this person. It must:
- Address the AI in second person ("you" / "I want you to" / "I prefer that you").
- Reference {character} by name OR by an unambiguous role tag (e.g. "my boss" for Boss Zhang).
- Express the *focus* (what to remember more of), not the literal preference word-for-word.
- Be at most 25 words. No quotes, no markdown, no leading/trailing whitespace.

Examples (different people / preferences):
- "I'd rather you only keep the final decisions Boss Zhang makes, not every comment he tosses out."
- "Please focus on what Liwei truly commits to and how she's feeling -- skip the small talk."
- "When Wanghao talks tech, I want you to remember the conclusions and the numbers, nothing else."

Return ONLY the sentence."""


OUTLINE_PROMPT = """You are an expert dialogue-data designer. You are designing scenarios that will
STRESS-TEST a memory system whose weakness is that it stores too much.

## Story background
{theme}

## Cast (the protagonist is Linchen)
- Boss Zhang  : Linchen's manager. Preference = ONLY remember Boss Zhang's final decisions / conclusions.
- Liwei       : Linchen's girlfriend. Preference = ONLY remember serious commitments + emotional shifts.
- Wanghao     : Linchen's engineering colleague. Preference = ONLY remember technical conclusions + key numbers.
- Sister Chen : Linchen's close friend / coworker. Preference = ONLY remember emotional changes + their triggers.
- Mom         : Linchen's mother. Preference = ONLY remember Linchen's promises + major family events.
- Jiange      : Linchen's workout buddy. Preference = ONLY remember schedule / time arrangements.

## This is week {week} of {total_weeks}. Date range: {date_range}
We are designing the WHOLE multi-week story one week at a time. After this week
there are still {weeks_remaining} more weeks to design, so leave room for
follow-ups, status updates, and decisions that get refined later.

## Topic anchor pool (you MUST pick from this list -- do not invent new anchors unless absolutely necessary)
{anchor_pool}

## What has already happened earlier in this story (cumulative key_info from weeks 1..{prev_week})
Each item below is a fact that the IDEAL preference-aware memory system already
holds going into week {week}. Treat them as canon -- you may reference them,
update them (use SUPERSEDES), or build on them. DO NOT contradict them silently.

{memory_context}

## Design philosophy

The dataset's whole purpose is to expose this failure mode:
"a memory system stores everything topically related, so the relevant fact gets
buried under semantically-similar but irrelevant items."

To do that, every scenario is built around a **topic anchor** (chosen from the
pool). Anchors must be REUSED across multiple scenarios in this week, ideally
across weeks too (you can see prior weeks via memory_context above).

For every scenario, you produce two kinds of items:

### `key_info` -- things the focal character's preference DOES want to keep
Strictly aligned with that character's preference. Each is a complete sentence
(subject-verb-object), and either embeds the date "{example_date}" or starts with
"On {example_date}" so it is unambiguously placed in time.

### `noise_info` -- things TOPICALLY around the anchor but the preference DROPS
The trap: noise_info must look retrieval-relevant for a future query, but a
preference-aware system would correctly drop it. Each noise item MUST:
1. Mention the same topic anchor as key_info.
2. Belong to an information TYPE the focal character's preference rejects.
3. Be plausible chit-chat someone would say in that conversation.

## SUPERSEDES updates
At least {supersedes_min} scenarios this week MUST update an earlier fact from
memory_context. For each such scenario, add a field
`supersedes_key_info_ref: {{"week": <prior W>, "session_index": <0-based index>}}`
pointing at the earlier scenario whose key_info you are overriding. The new
key_info should make the OLD fact wrong (e.g. previously "TBD next Friday", now
"confirmed for Saturday 9 AM").

## Hard requirements for this week ({sessions} scenarios)

R1. Pick 3-5 topic anchors for THIS week from the pool. Each must be reused in
    at least 2 different scenarios this week.
R2. Produce exactly {sessions} scenarios. Each scenario:
    - 1-3 `key_info` items, all matching the focal character's preference.
    - 3-6 `noise_info` items, all referencing the anchor.
    - `topic_anchor` field naming the anchor (must come from the pool above).
    - `focal_character` field naming the ONE character whose preference governs.
    - `day` (Monday..Sunday) and `date` (YYYY-MM-DD inside {date_range}).
R3. Across the WHOLE story (counting all prior weeks visible in memory_context
    plus this week), every one of the 6 cast characters must appear as
    `focal_character` at least once. Use this week to fill any character gaps.
R4. The same anchor word/phrase must literally appear in BOTH key_info and
    noise_info of the same scenario (so naive vector RAG cannot tell them apart).
R5. {supersedes_min}+ scenarios carry `supersedes_key_info_ref` (rule above).

## Output schema
Return ONLY a JSON object (no prose, no markdown fences):
{{
  "topic_anchors": ["the morning coffee chat", "the Shenzhen trip"],
  "scenarios": [
    {{
      "day": "Monday",
      "date": "{example_date}",
      "focal_character": "Boss Zhang",
      "topic_anchor": "the morning coffee chat",
      "topic": "<short scene description>",
      "characters_mentioned": ["Boss Zhang"],
      "key_info": ["On {example_date} Boss Zhang approved Linchen's PTO for next Friday."],
      "noise_info": ["Someone joked the espresso machine is broken again."],
      "supersedes_key_info_ref": null
    }}
  ]
}}"""


DIALOGUE_PROMPT = """You are an expert dialogue writer. Expand the scenario outline below into a
realistic multi-turn conversation between Linchen and his personal AI assistant,
AND simultaneously emit the IDEAL set of memories the assistant should keep.

## Story background
{theme}

## This session
- Date: {date} ({day})
- Topic: {topic}
- Focal character (the ONE whose preference governs this session): {focal_character}
- Topic anchor: {topic_anchor}
- Other characters mentioned: {characters}

## Focal character's memory preference
{preference_desc}

## What Linchen's AI already knows up to this session
This is the cumulative state going into this session (from earlier sessions).
Do NOT contradict it; you may reference or update it through the dialogue.

{memory_context_so_far}

## Information that MUST appear naturally in the dialogue
Key info (to be naturally surfaced by Linchen -- don't quote verbatim):
{key_info_list}

Value-layered noise info (surface as chit-chat / details / background):
{noise_info_list}

## Requirements for the dialogue
1. 4-8 turns (each turn = 1 user + 1 assistant message), strictly alternating
   user/assistant, starting with user.
2. user = Linchen, in natural spoken English. assistant = the AI: concise,
   friendly, like a buddy.
3. Every key_info AND every noise_info item must be semantically covered in
   Linchen's messages.
4. The conversation must feel real (contractions, fragments OK).

## Requirements for golden_memories
After the dialogue, output the IDEAL minimum atomic facts to remember.
Each item MUST:
1. Be a single self-contained sentence (subject-verb-object).
2. Carry a time anchor: prefix or embed "{date}" (e.g. "On {date} ..." or
   "on {day} {date} ..."). The date must be inside the sentence, not in a
   separate field.
3. Match the focal character's preference (drop chit-chat, drop noise items).
4. Be a fact Linchen or the focal character ACTUALLY said / committed to /
   decided in THIS dialogue -- do NOT invent.
5. If nothing preference-aligned was said, return an empty list.

## Output schema
Return ONLY a JSON object (no prose, no markdown fences):
{{
  "messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "golden_memories": [
    "On {date} Boss Zhang approved Linchen's PTO request for next Friday."
  ]
}}"""


QA_PROMPT = """You are an expert dataset annotator for a memory-system benchmark.

It is the end of week {week}. Generate exactly {qa_per_week} evaluation
questions Linchen would ask his AI assistant.

## This week's golden memories (the ONLY oracles you may target)
Each entry below is a fact stored at the end of week {week}. You MUST pick the
target memory for every question from THIS list -- never invent a target.

{week_memory_list}

## Cast preferences (so you understand WHY each memory was kept)
{cast_preferences}

## Question design rules

For EVERY question you must output a `question` field in the colloquial,
DILUTION-STRESS form. It must:
- Keep the topic anchor (or a clear synonym) so semantic-RAG retrieves many
  competing memories on that anchor.
- NOT name the focal character.
- NOT name the preference category as a literal word (avoid "final decision",
  "latest commitment", "technical conclusion", "emotional reaction", "schedule").
- NOT leak distinctive content from the gold answer (no specific numbers,
  proper nouns, or quoted phrases that appear only in the gold).
- Read like spoken English (contractions, fragments, casual register).

For EVERY question you must also output `golden_feedback`: a short spoken-style
statement that the user would say to *correct* the AI when it answers wrong.
It restates the gold answer naturally so it can be appended after
"You are wrong, ". Examples:
  gold_answer "9:00 AM" -> golden_feedback "the actual time is 9:00 AM"
  gold_answer "Marriott" -> golden_feedback "we agreed to stay at the Marriott"

## SELF-ANCHORED date/time answers (HARD RULE)

If the question asks about a date, day-of-week, time, or schedule (i.e.
`info_category == "Schedule/Time"`, OR the natural answer involves a weekday /
"next ..." / "tomorrow" / clock time / calendar date), the `gold_answer` MUST
be self-anchored: include BOTH an absolute calendar date AND the relative /
weekday phrasing, joined by " -- i.e. ".

The absolute date MUST be copied from the chosen target memory's "On
YYYY-MM-DD ..." prefix or from the body of the memory; do not invent a date.
If the target memory only contains a relative phrase, look at the memory's
session date (the "On YYYY-MM-DD" prefix) and resolve the weekday accordingly.

Format:
  "<YYYY-MM-DD> (<Weekday>) at <HH:MM AM/PM> -- i.e. <relative phrase>"

Examples:
  target memory: "On 2025-08-01 Jiange scheduled a hospital visit next Wednesday at 8 AM."
    gold_answer    -> "2025-08-06 (Wednesday) at 8 AM -- i.e. next Wednesday at 8 AM"
    golden_feedback -> "the actual time is 2025-08-06 (Wednesday) at 8 AM, i.e. next Wednesday at 8 AM"

  target memory: "On 2025-07-09 Jiange rescheduled the badminton session to Friday at 8 PM."
    gold_answer    -> "2025-07-11 (Friday) at 8 PM -- i.e. Friday at 8 PM"

If the question is purely date-only (no time-of-day in the source), drop the
"at HH:MM" part:
  "<YYYY-MM-DD> (<Weekday>) -- i.e. <relative phrase>"

This rule does NOT apply to non-time answers (names, places, decisions, etc.).

## target_memory_refs (REQUIRED)
For every question, set `target_memory_refs` to a JSON array of exactly one
ref: {{"week": {week}, "session_id": "<wW_sN>", "index": <int>}}, copied
EXACTLY from the `[REF: ...]` tag of the chosen memory. The ref must point at
a memory in the list above whose content fully justifies your `gold_answer`.

## Distribution
Spread the {qa_per_week} questions across as many distinct memories / focal
characters / anchors as possible (ideally one question per memory if there are
enough memories).

## Output schema
Return a JSON array of exactly {qa_per_week} objects:
[
  {{
    "qa_type": "within_pref",
    "question": "...",
    "golden_feedback": "...",
    "gold_answer": "...",
    "target_memory": "<copied verbatim from the list above>",
    "target_memory_refs": [{{"week": {week}, "session_id": "wX_sY", "index": 0}}],
    "character": "...",
    "topic_anchor": "...",
    "info_category": "Decision/Conclusion|Fact/Number|Agreement/Promise|Emotion/Attitude|Schedule/Time",
    "supersedes_chain": []
  }}
]
Return JSON only -- no prose, no markdown fences."""


# Legacy prompt kept ONLY for --backfill-golden against old datasets.
GOLDEN_MEMORY_PROMPT = """You are an expert memory-curator labeling the IDEAL set of memories
that should be extracted from ONE conversation session.

## Story background
{theme}

## This session
- session_id: {session_id}
- date: {observation_date}
- day: {day}
- topic: {topic}
- focal_character: {focal_character}
- characters_mentioned: {characters_mentioned}
- topic_anchor: {topic_anchor}

## Focal character's memory preference (the user's `preference_focus`)
{preference_focus}

## The conversation (verbatim)
{dialogue_text}

## Task
Extract the COMPLETE list of `golden_memories` for this session: the *minimum,
atomic* facts that an ideal preference-aware memory system should store after
reading this dialogue.

Each golden memory MUST satisfy ALL:
1. It is a single, self-contained sentence (subject-verb-object).
2. It carries a time anchor when relevant: prefix or embed the date "{observation_date}"
   (or a clear relative anchor like "on Monday {observation_date}") so the fact
   is unambiguously placed in time.
3. It is aligned with the focal character's preference focus above.
4. It is a fact LINCHEN OR THE FOCAL CHARACTER actually said / committed to /
   decided in THIS session -- do NOT invent.
5. If the dialogue contains nothing preference-aligned, return [].

## Output
Return JSON only, no prose, no markdown fences. Format:
[
  "On {observation_date} Boss Zhang approved Linchen's PTO request for next Friday.",
  "..."
]"""


# ============================================================================
# UTILITY HELPERS (legacy & shared)
# ============================================================================

def _normalise_target_memory_refs(raw):
    """Coerce arbitrary LLM output into a list of {week,session_id,index} dicts."""
    if not isinstance(raw, list):
        return []
    out = []
    seen = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            w = int(entry.get("week"))
            s = str(entry.get("session_id") or "").strip()
            i = int(entry.get("index"))
        except Exception:
            continue
        if not s or w <= 0 or i < 0:
            continue
        key = (w, s, i)
        if key in seen:
            continue
        seen.add(key)
        out.append({"week": w, "session_id": s, "index": i})
    return out


def _build_content_to_refs(story_dir, *, num_weeks):
    """Map session-level golden memory text -> list of (week, session_id, index)."""
    out = {}
    for w in range(1, num_weeks + 1):
        path = os.path.join(story_dir, f"week{w}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                wd = json.load(f)
        except Exception:
            continue
        for sess in wd.get("conversations", []) or []:
            sid = sess.get("session_id") or ""
            for idx, gm in enumerate(sess.get("golden_memories", []) or []):
                if not isinstance(gm, str):
                    continue
                key = gm.strip()
                if not key:
                    continue
                out.setdefault(key, []).append((w, sid, idx))
    return out


def _collect_golden_memory_bank(story_dir, *, num_weeks):
    """Read-only aggregator over weekN.json. Used by validate / backfill ONLY."""
    bank = []
    for w in range(1, num_weeks + 1):
        path = os.path.join(story_dir, f"week{w}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            wd = json.load(f)
        for sess in wd.get("conversations", []) or []:
            sid = sess.get("session_id") or ""
            char = sess.get("focal_character") or ""
            anchor = sess.get("topic_anchor") or "misc"
            for idx, gm in enumerate(sess.get("golden_memories", []) or []):
                if not isinstance(gm, str) or not gm.strip():
                    continue
                bank.append({
                    "content": " ".join(gm.split()),
                    "week": w,
                    "character": char,
                    "topic_anchor": anchor,
                    "session_id": sid,
                    "index": idx,
                    "date": sess.get("date", ""),
                })
    return bank


# ============================================================================
# STEP 1 -- preference_feedback (per character, prefixed)
# ============================================================================

def _generate_preference_feedback(character, preference, theme, *, story_id=None):
    prompt = PREFERENCE_FEEDBACK_PROMPT.format(
        character=character, preference=preference, theme=theme,
    )
    safe_char = "".join(c if c.isalnum() else "_" for c in (character or "unk"))[:32]
    tag = f"data_prep_pref_fb_s{story_id or '?'}_{safe_char}"
    text = (
        call_gen(
            prompt, max_tokens=8192, temperature=0.5,
            tag=tag, story_id=story_id, kind="data_prep",
        )
        or ""
    ).strip()
    text = text.strip().strip('`').strip('"').strip("'").strip()
    text = " ".join(s.strip() for s in text.splitlines() if s.strip())
    return text or f"I want you to focus more on {preference} when it comes to {character}."


def _ensure_character_profiles(story_dir, theme, *, story_id=None):
    """Idempotent: write character_profiles.json with one entry per cast member."""
    cp_path = os.path.join(story_dir, "character_profiles.json")
    profiles = {}
    if os.path.exists(cp_path):
        try:
            with open(cp_path) as f:
                profiles = json.load(f)
        except Exception:
            profiles = {}
    changed = False
    for char, meta in PREFERENCES.items():
        cur = profiles.get(char) or {}
        if not (cur.get("preference_feedback") or "").strip():
            fb = _generate_preference_feedback(
                char, meta["desc"], theme, story_id=story_id,
            )
            cur["preference_feedback"] = fb
            cur["preference_id"] = meta["id"]
            cur["preference_desc"] = meta["desc"]
            profiles[char] = cur
            changed = True
        else:
            cur.setdefault("preference_id", meta["id"])
            cur.setdefault("preference_desc", meta["desc"])
            profiles[char] = cur
    if changed:
        with open(cp_path, "w") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
    return profiles


# ============================================================================
# STEP 2 -- outline (one LLM call per week, but key_info is cumulative)
# ============================================================================

def _render_memory_context_from_key_infos(accumulated):
    """Render `accumulated` (list of dicts) into the OUTLINE_PROMPT memory_context.

    Each dict carries: week, session_index, focal_character, topic_anchor,
    date, text. We render grouped by week so the model sees temporal order.
    """
    if not accumulated:
        return "(week 1 -- empty memory)"
    by_week = {}
    for ki in accumulated:
        by_week.setdefault(int(ki["week"]), []).append(ki)
    lines = []
    for w in sorted(by_week):
        lines.append(f"### Week {w}")
        for ki in by_week[w]:
            lines.append(
                f"  - [{ki['date']}] [{ki['focal_character']}] "
                f"[anchor: {ki['topic_anchor']}] "
                f"[ref: w{w}_s{ki['session_index']+1}] {ki['text']}"
            )
    return "\n".join(lines)


def _generate_one_week_outline(
    *, theme, week, total_weeks, base_date, sessions_per_week,
    accumulated_key_infos, story_id,
):
    """Call the outline LLM once for a single week. Returns the parsed dict."""
    week_end = base_date + timedelta(days=6)
    date_range = f"{base_date} ~ {week_end}"
    weeks_remaining = total_weeks - week
    prev_week = week - 1
    memory_context = _render_memory_context_from_key_infos(accumulated_key_infos)
    anchor_pool_str = "\n".join(f"- {a}" for a in ANCHOR_POOL)
    supersedes_min = 1 if week >= 2 else 0
    prompt = OUTLINE_PROMPT.format(
        theme=theme, week=week, total_weeks=total_weeks,
        date_range=date_range, weeks_remaining=weeks_remaining,
        anchor_pool=anchor_pool_str,
        prev_week=prev_week, memory_context=memory_context,
        example_date=str(base_date), sessions=sessions_per_week,
        supersedes_min=supersedes_min,
    )
    last_err = None
    for attempt in range(MAX_RETRY):
        resp = call_gen(
            prompt,
            tag=f"data_prep_outline_s{story_id}_w{week}",
            story_id=story_id, week=week, kind="data_prep",
        )
        if not resp:
            time.sleep(2)
            continue
        try:
            parsed = parse_json_from_llm(resp)
        except Exception as e:
            last_err = e
            time.sleep(1)
            continue
        scenarios = None
        topic_anchors = []
        if isinstance(parsed, dict):
            scenarios = parsed.get("scenarios")
            topic_anchors = parsed.get("topic_anchors") or []
        elif isinstance(parsed, list):
            scenarios = parsed
        if not isinstance(scenarios, list):
            time.sleep(1)
            continue
        # Validate required fields per session.
        ok = True
        for sc in scenarios:
            if not isinstance(sc, dict):
                ok = False
                break
            for fld in ("focal_character", "topic_anchor", "key_info", "noise_info"):
                if fld not in sc:
                    ok = False
                    break
            if not ok:
                break
            if sc.get("focal_character") not in PREFERENCES:
                ok = False
                break
            if not isinstance(sc.get("key_info"), list) or not sc["key_info"]:
                ok = False
                break
            if not isinstance(sc.get("noise_info"), list):
                ok = False
                break
        if not ok:
            time.sleep(1)
            continue
        if len(scenarios) < max(3, sessions_per_week - 2):
            time.sleep(1)
            continue
        scenarios = scenarios[:sessions_per_week]
        # Light normalization: ensure date / day defaults.
        for j, sc in enumerate(scenarios):
            sc.setdefault("date", str(base_date + timedelta(days=min(j, 6))))
            sc.setdefault("day", (base_date + timedelta(days=min(j, 6))).strftime("%A"))
            sc.setdefault("topic", "")
            sc.setdefault("characters_mentioned", [sc["focal_character"]])
        return {
            "week": week,
            "date_range": date_range,
            "topic_anchors": topic_anchors,
            "scenarios": scenarios,
        }
    raise RuntimeError(
        f"outline LLM failed for story={story_id} week={week} after "
        f"{MAX_RETRY} retries (last_err={last_err})"
    )


def _validate_outline(outline, *, total_weeks):
    """Return list of human-readable issues; empty list = OK.

    Used to optionally trigger per-week regeneration after the full N-week
    outline is assembled.
    """
    issues = []
    focal_seen = set()
    anchor_uses = {}  # anchor -> set of weeks where used
    supersedes_count = 0
    for wk in outline["weeks"]:
        for j, sc in enumerate(wk["scenarios"]):
            focal_seen.add(sc.get("focal_character"))
            a = sc.get("topic_anchor") or "misc"
            anchor_uses.setdefault(a, set()).add(wk["week"])
            if sc.get("supersedes_key_info_ref"):
                supersedes_count += 1
    missing_focal = [c for c in PREFERENCES if c not in focal_seen]
    if missing_focal:
        issues.append(f"focal_character missing for: {missing_focal}")
    sup_min = max(2, total_weeks // 3)
    if supersedes_count < sup_min:
        issues.append(
            f"supersedes scenarios = {supersedes_count}, need >= {sup_min}"
        )
    cross_week_anchors = sum(1 for a, ws in anchor_uses.items() if len(ws) >= 2)
    cross_min = math.ceil(3 * total_weeks / 5)
    if total_weeks >= 2 and cross_week_anchors < cross_min:
        issues.append(
            f"cross-week-reused anchors = {cross_week_anchors}, need >= {cross_min}"
        )
    return issues


def _generate_outline_for_story(
    story_dir, *, story_id, theme, num_weeks, sessions_per_week, base_date_start,
):
    """Outline-first: emit one outline.json covering all N weeks. Idempotent."""
    out_path = os.path.join(story_dir, "outline.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)

    weeks_out = []
    accumulated = []  # list of dicts: week, session_index, focal_character, topic_anchor, date, text

    for w in range(1, num_weeks + 1):
        base_date = base_date_start + timedelta(weeks=w - 1)
        wk = _generate_one_week_outline(
            theme=theme, week=w, total_weeks=num_weeks,
            base_date=base_date, sessions_per_week=sessions_per_week,
            accumulated_key_infos=accumulated, story_id=story_id,
        )
        weeks_out.append(wk)
        # Append this week's key_infos to the accumulated context.
        for j, sc in enumerate(wk["scenarios"]):
            for ki in sc.get("key_info", []) or []:
                if not isinstance(ki, str) or not ki.strip():
                    continue
                accumulated.append({
                    "week": w,
                    "session_index": j,
                    "focal_character": sc.get("focal_character") or "",
                    "topic_anchor": sc.get("topic_anchor") or "",
                    "date": sc.get("date") or str(base_date),
                    "text": ki.strip(),
                })

    outline = {
        "theme": theme,
        "num_weeks": num_weeks,
        "sessions_per_week": sessions_per_week,
        "weeks": weeks_out,
    }

    # Global validation; if violated, regenerate the worst-offending weeks.
    for retry in range(MAX_RETRY):
        issues = _validate_outline(outline, total_weeks=num_weeks)
        if not issues:
            break
        log.warning(
            f"outline validation failed (story={story_id}, attempt={retry+1}): {issues}"
        )
        # Pick the worst week heuristically: the latest week, since later
        # weeks have most context and are cheapest to redo.
        target_week = num_weeks
        # Roll back accumulated to before target_week.
        accumulated = [k for k in accumulated if k["week"] < target_week]
        base_date = base_date_start + timedelta(weeks=target_week - 1)
        wk = _generate_one_week_outline(
            theme=theme, week=target_week, total_weeks=num_weeks,
            base_date=base_date, sessions_per_week=sessions_per_week,
            accumulated_key_infos=accumulated, story_id=story_id,
        )
        # Replace.
        outline["weeks"][target_week - 1] = wk
        for j, sc in enumerate(wk["scenarios"]):
            for ki in sc.get("key_info", []) or []:
                if not isinstance(ki, str) or not ki.strip():
                    continue
                accumulated.append({
                    "week": target_week,
                    "session_index": j,
                    "focal_character": sc.get("focal_character") or "",
                    "topic_anchor": sc.get("topic_anchor") or "",
                    "date": sc.get("date") or str(base_date),
                    "text": ki.strip(),
                })
    else:
        raise RuntimeError(
            f"outline validation failed for story={story_id} after {MAX_RETRY} retries: "
            f"{_validate_outline(outline, total_weeks=num_weeks)}"
        )

    with open(out_path, "w") as f:
        json.dump(outline, f, ensure_ascii=False, indent=2)
    return outline


# ============================================================================
# STEP 3 -- dialogue + golden_memories (single LLM call per session)
# ============================================================================

def _render_memory_context_so_far(accumulated_key_infos, accumulated_gms):
    parts = []
    if accumulated_key_infos:
        parts.append("### Earlier sessions' key_info (planned facts)")
        for ki in accumulated_key_infos:
            parts.append(
                f"  - [{ki['date']}] [{ki['focal_character']}] "
                f"[anchor: {ki['topic_anchor']}] {ki['text']}"
            )
    if accumulated_gms:
        parts.append("\n### Earlier sessions' golden_memories (already remembered)")
        for gm in accumulated_gms:
            parts.append(f"  - {gm['content']}")
    return "\n".join(parts) if parts else "(this is the first session of the story)"


def _generate_session_dialogue_with_gm(
    *, theme, scenario, accumulated_key_infos, accumulated_gms, story_id, week, session_index,
):
    """Single LLM call producing both messages and golden_memories."""
    focal = scenario.get("focal_character") or ""
    if focal not in PREFERENCES:
        raise RuntimeError(
            f"session w{week}_s{session_index+1} has invalid focal_character={focal!r}"
        )
    pref_desc = PREFERENCES[focal]["desc"]
    key_info_list = "\n".join(f"- {ki}" for ki in scenario.get("key_info", []) or [])
    noise_info_list = "\n".join(f"- {ni}" for ni in scenario.get("noise_info", []) or [])
    memory_context_so_far = _render_memory_context_so_far(
        accumulated_key_infos, accumulated_gms,
    )
    prompt = DIALOGUE_PROMPT.format(
        theme=theme,
        date=scenario.get("date") or "",
        day=scenario.get("day") or "",
        topic=scenario.get("topic") or "",
        focal_character=focal,
        topic_anchor=scenario.get("topic_anchor") or "",
        characters=", ".join(scenario.get("characters_mentioned") or [focal]),
        preference_desc=pref_desc,
        memory_context_so_far=memory_context_so_far,
        key_info_list=key_info_list or "(none)",
        noise_info_list=noise_info_list or "(none)",
    )
    tag = f"data_prep_dialgm_s{story_id}_w{week}_s{session_index+1}"
    last_err = None
    for attempt in range(MAX_RETRY):
        resp = call_gen(
            prompt, max_tokens=8192,
            tag=tag, story_id=story_id, week=week, kind="data_prep",
        )
        if not resp:
            time.sleep(1)
            continue
        try:
            parsed = parse_json_from_llm(resp)
        except Exception as e:
            last_err = e
            time.sleep(1)
            continue
        if not isinstance(parsed, dict):
            time.sleep(1)
            continue
        msgs = parsed.get("messages")
        gms = parsed.get("golden_memories")
        if not isinstance(msgs, list) or len(msgs) < 4:
            time.sleep(1)
            continue
        ok = True
        for k, m in enumerate(msgs):
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                ok = False
                break
            expected = "user" if k % 2 == 0 else "assistant"
            if m["role"] != expected:
                ok = False
                break
        if not ok:
            time.sleep(1)
            continue
        if not isinstance(gms, list) or not gms:
            time.sleep(1)
            continue
        cleaned_gms = []
        for g in gms:
            if not isinstance(g, str):
                continue
            g_norm = " ".join(g.split())
            if not g_norm:
                continue
            cleaned_gms.append(g_norm)
        if not cleaned_gms:
            time.sleep(1)
            continue
        return msgs, cleaned_gms
    raise RuntimeError(
        f"dialogue+GM LLM failed for story={story_id} w{week}_s{session_index+1} "
        f"after {MAX_RETRY} retries (last_err={last_err})"
    )


def _generate_week_dialogues(
    *, story_dir, theme, week_outline, story_id,
    accumulated_key_infos, accumulated_gms,
):
    """For ONE week: emit weekN.json. SESSION-level idempotent.

    Each completed session is flushed immediately to weekN.json with a
    ``partial`` flag, so an interrupted run can resume from the next
    missing session instead of restarting the week from scratch.
    """
    week = week_outline["week"]
    week_path = os.path.join(story_dir, f"week{week}.json")
    scenarios = week_outline.get("scenarios", []) or []
    expected_n = len(scenarios)

    on_disk_payload = {}
    existing_sessions = []
    if os.path.exists(week_path):
        try:
            with open(week_path) as f:
                on_disk_payload = json.load(f) or {}
            existing_sessions = list(
                on_disk_payload.get("conversations", []) or []
            )
        except Exception:
            on_disk_payload = {}
            existing_sessions = []

    by_sid = {
        s.get("session_id"): s for s in existing_sessions
        if isinstance(s, dict) and s.get("session_id")
    }

    # Fully-complete week: refresh accumulators and return early.
    if len(by_sid) >= expected_n and expected_n > 0:
        for sess in existing_sessions:
            for gm in sess.get("golden_memories", []) or []:
                if isinstance(gm, str) and gm.strip():
                    accumulated_gms.append({
                        "week": week,
                        "session_id": sess.get("session_id"),
                        "content": " ".join(gm.split()),
                    })
        for j, sc in enumerate(scenarios):
            for ki in sc.get("key_info", []) or []:
                if not isinstance(ki, str) or not ki.strip():
                    continue
                accumulated_key_infos.append({
                    "week": week,
                    "session_index": j,
                    "focal_character": sc.get("focal_character") or "",
                    "topic_anchor": sc.get("topic_anchor") or "",
                    "date": sc.get("date") or "",
                    "text": ki.strip(),
                })
        if on_disk_payload.get("partial"):
            with open(week_path, "w") as f:
                json.dump(
                    {"week": week, "conversations": existing_sessions},
                    f, ensure_ascii=False, indent=2,
                )
        return {"week": week, "conversations": existing_sessions}

    # Partial / fresh path.
    conversations = []
    for j, sc in enumerate(scenarios):
        sid = f"w{week}_s{j+1}"
        if sid in by_sid:
            session_obj = by_sid[sid]
            print(
                f"    [resume] W{week} {sid}: reuse existing session",
                flush=True,
            )
        else:
            msgs, cleaned_gms = _generate_session_dialogue_with_gm(
                theme=theme, scenario=sc,
                accumulated_key_infos=accumulated_key_infos,
                accumulated_gms=accumulated_gms,
                story_id=story_id, week=week, session_index=j,
            )
            session_obj = {
                "session_id": sid,
                "day": sc.get("day"),
                "date": sc.get("date"),
                "topic": sc.get("topic"),
                "focal_character": sc.get("focal_character"),
                "topic_anchor": sc.get("topic_anchor"),
                "characters_mentioned": (
                    sc.get("characters_mentioned")
                    or [sc.get("focal_character")]
                ),
                "messages": msgs,
                "golden_memories": cleaned_gms,
            }
        conversations.append(session_obj)

        for ki in sc.get("key_info", []) or []:
            if not isinstance(ki, str) or not ki.strip():
                continue
            accumulated_key_infos.append({
                "week": week,
                "session_index": j,
                "focal_character": sc.get("focal_character") or "",
                "topic_anchor": sc.get("topic_anchor") or "",
                "date": sc.get("date") or "",
                "text": ki.strip(),
            })
        for gm in session_obj.get("golden_memories", []) or []:
            if isinstance(gm, str) and gm.strip():
                accumulated_gms.append({
                    "week": week,
                    "session_id": session_obj["session_id"],
                    "content": " ".join(gm.split()),
                })

        # Atomic per-session flush.
        is_last = (j == expected_n - 1)
        payload = {"week": week, "conversations": conversations}
        if not is_last:
            payload["partial"] = True
        tmp_path = week_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, week_path)

    return {"week": week, "conversations": conversations}


# ============================================================================
# STEP 4 -- QA + golden_feedback (single LLM call per week)
# ============================================================================

def _generate_week_qa_with_feedback(
    *, story_dir, week, week_data, qa_per_week, story_id,
):
    """Generate qa_per_week QAs for THIS week's golden_memories. Returns list."""
    # Build the per-week memory list (the only oracle source for this week).
    week_mems = []  # list of (content, focal, anchor, session_id, index, date)
    for sess in week_data.get("conversations", []) or []:
        sid = sess.get("session_id") or ""
        focal = sess.get("focal_character") or ""
        anchor = sess.get("topic_anchor") or ""
        date_s = sess.get("date") or ""
        for idx, gm in enumerate(sess.get("golden_memories", []) or []):
            if not isinstance(gm, str) or not gm.strip():
                continue
            week_mems.append({
                "content": " ".join(gm.split()),
                "focal": focal, "anchor": anchor,
                "session_id": sid, "index": idx, "date": date_s,
            })
    if not week_mems:
        raise RuntimeError(
            f"week {week} has zero golden_memories; cannot generate QA"
        )

    # Render the memory list with [REF: ...] tags.
    lines = []
    for m in week_mems:
        lines.append(
            f"- {m['content']}  [REF: week={week} session={m['session_id']} index={m['index']}] "
            f"[focal: {m['focal']}] [anchor: {m['anchor']}] [date: {m['date']}]"
        )
    week_memory_list = "\n".join(lines)

    cast_prefs = "\n".join(
        f"- {char}: {meta['desc']}" for char, meta in PREFERENCES.items()
    )

    valid_keys = {(m["session_id"], m["index"]) for m in week_mems}
    # (session_id, index) -> authoritative golden_memory text (used to
    # overwrite LLM-written target_memory so it never carries [REF: ...]
    # metadata leakage from the prompt).
    gm_text_by_key = {
        (m["session_id"], m["index"]): m["content"] for m in week_mems
    }

    prompt = QA_PROMPT.format(
        week=week, qa_per_week=qa_per_week,
        week_memory_list=week_memory_list, cast_preferences=cast_prefs,
    )
    tag = f"data_prep_qagf_s{story_id}_w{week}"
    last_err = None
    for attempt in range(MAX_RETRY):
        resp = call_gen(
            prompt, temperature=0.7, max_tokens=8192,
            tag=tag, story_id=story_id, week=week, kind="data_prep",
        )
        if not resp:
            time.sleep(2)
            continue
        try:
            parsed = parse_json_from_llm(resp)
        except Exception as e:
            last_err = e
            time.sleep(2)
            continue
        if not isinstance(parsed, list):
            time.sleep(2)
            continue
        accepted = []
        for r in parsed:
            if not isinstance(r, dict):
                continue
            q = (r.get("question") or "").strip()
            ga = (r.get("gold_answer") or "").strip()
            gf = (r.get("golden_feedback") or "").strip()
            if not q or not ga or not gf:
                continue
            refs = _normalise_target_memory_refs(r.get("target_memory_refs"))
            # Keep only refs pointing at THIS week's golden memories.
            refs = [
                ref for ref in refs
                if ref["week"] == week and (ref["session_id"], ref["index"]) in valid_keys
            ]
            if not refs:
                continue
            # Authoritative target_memory: always rebuilt from refs, ignoring
            # whatever the LLM wrote. This prevents [REF: ...] / [focal: ...]
            # metadata residues from the QA prompt leaking into the dataset.
            tm_parts = [
                gm_text_by_key[(ref["session_id"], ref["index"])]
                for ref in refs
            ]
            target_memory_text = "\n".join(tm_parts)
            accepted.append({
                "qa_type": r.get("qa_type") or "within_pref",
                "question": q,
                "golden_feedback": gf,
                "gold_answer": ga,
                "target_memory": target_memory_text,
                "target_memory_refs": refs,
                "character": r.get("character") or "",
                "topic_anchor": r.get("topic_anchor") or "",
                "info_category": r.get("info_category") or "",
                "supersedes_chain": r.get("supersedes_chain") or [],
            })
        if len(accepted) >= max(3, qa_per_week - 2):
            return accepted[:qa_per_week]
        time.sleep(2)
    raise RuntimeError(
        f"QA+GF LLM failed for story={story_id} week={week} after "
        f"{MAX_RETRY} retries (last_err={last_err})"
    )


# ============================================================================
# MAIN GENERATOR
# ============================================================================

def generate_story_data(story_id, theme, *, num_weeks, convs_per_week, qa_per_week):
    """End-to-end. File-level idempotent: existing artifacts are reused."""
    sd = story_data_dir(story_id)
    os.makedirs(sd, exist_ok=True)
    base_date_start = date(2025, 7, 7)

    # ------ STEP 1: preference_feedback (prefixed before outline) ------
    print(f"  [story {story_id}] STEP 1: preference_feedback ...", flush=True)
    _ensure_character_profiles(sd, theme, story_id=story_id)

    # ------ STEP 2: outline (one outline.json for all N weeks) ------
    print(f"  [story {story_id}] STEP 2: outline x {num_weeks} ...", flush=True)
    outline = _generate_outline_for_story(
        sd, story_id=story_id, theme=theme, num_weeks=num_weeks,
        sessions_per_week=convs_per_week, base_date_start=base_date_start,
    )
    print(
        f"    outline ok: {sum(len(w['scenarios']) for w in outline['weeks'])} sessions, "
        f"{len(set(a for w in outline['weeks'] for a in w.get('topic_anchors', []) or []))} unique anchors",
        flush=True,
    )

    # ------ STEP 3 + 4 interleaved per week ------
    # QA is flushed per-week so a crash mid-story does not lose finished QA.
    qa_path = os.path.join(sd, "test_qa.json")
    all_questions = []
    done_weeks = set()
    if os.path.exists(qa_path):
        try:
            with open(qa_path) as f:
                prev = json.load(f)
            for q in prev.get("test_questions", []) or []:
                if isinstance(q, dict) and q.get("question_id"):
                    all_questions.append(q)
                    if isinstance(q.get("test_week"), int):
                        done_weeks.add(q["test_week"])
            if all_questions:
                print(
                    f"  [story {story_id}] resume QA: "
                    f"{len(all_questions)} questions across weeks "
                    f"{sorted(done_weeks)}",
                    flush=True,
                )
        except Exception:
            all_questions = []
            done_weeks = set()

    accumulated_key_infos = []
    accumulated_gms = []
    for wk in outline["weeks"]:
        week = wk["week"]
        # Step 3: dialogues + golden memories for this week.
        week_data = _generate_week_dialogues(
            story_dir=sd, theme=theme, week_outline=wk, story_id=story_id,
            accumulated_key_infos=accumulated_key_infos,
            accumulated_gms=accumulated_gms,
        )
        msg_count = sum(len(c.get("messages", [])) for c in week_data["conversations"])
        gm_count = sum(len(c.get("golden_memories", [])) for c in week_data["conversations"])
        print(
            f"    W{week}: {len(week_data['conversations'])} sessions, "
            f"{msg_count} messages, {gm_count} golden_memories",
            flush=True,
        )

        # Step 4: QA + golden_feedback for this week (skip if already on disk).
        if week in done_weeks:
            existing = sum(1 for q in all_questions if q.get("test_week") == week)
            print(
                f"    QA W{week}: reuse {existing} questions from disk",
                flush=True,
            )
            continue

        qas = _generate_week_qa_with_feedback(
            story_dir=sd, week=week, week_data=week_data,
            qa_per_week=qa_per_week, story_id=story_id,
        )
        for i, r in enumerate(qas):
            all_questions.append({
                "test_week": week, "question_id": f"w{week}_q{i+1}",
                "qa_type": r.get("qa_type", "within_pref"),
                "character": r.get("character", ""),
                "topic_anchor": r.get("topic_anchor", ""),
                "question": r["question"],
                "golden_feedback": r["golden_feedback"],
                "gold_answer": r["gold_answer"],
                "target_memory": r.get("target_memory", ""),
                "target_memory_refs": r["target_memory_refs"],
                "supersedes_chain": r.get("supersedes_chain", []),
                "info_category": r.get("info_category", ""),
            })
        done_weeks.add(week)
        # Atomic incremental flush of test_qa.json after every week.
        qa_out = {
            "story_id": story_id,
            "total": len(all_questions),
            "test_questions": all_questions,
        }
        tmp_qa = qa_path + ".tmp"
        with open(tmp_qa, "w") as f:
            json.dump(qa_out, f, ensure_ascii=False, indent=2)
        os.replace(tmp_qa, qa_path)
        print(
            f"    QA W{week}: {len(qas)} questions (cum={len(all_questions)})",
            flush=True,
        )

    # Final consistency rewrite (covers the all-skipped case).
    qa_out = {
        "story_id": story_id,
        "total": len(all_questions),
        "test_questions": all_questions,
    }
    with open(qa_path, "w") as f:
        json.dump(qa_out, f, ensure_ascii=False, indent=2)

    total_gms = 0
    for w in range(1, num_weeks + 1):
        wp = os.path.join(sd, f"week{w}.json")
        if os.path.exists(wp):
            with open(wp) as f:
                wd = json.load(f)
            for sess in wd.get("conversations", []) or []:
                total_gms += len(sess.get("golden_memories", []) or [])
    print(
        f"    Data done: {total_gms} golden_memories, {len(all_questions)} QA",
        flush=True,
    )


# ============================================================================
# LEGACY BACKFILL (only used by --backfill-golden against pre-refactor data)
# ============================================================================

def _generate_session_golden_memories(session, theme, *, story_id=None, week=None):
    """Legacy: post-hoc GM extraction for old datasets (--backfill-golden)."""
    focal = session.get("focal_character") or ""
    if focal in PREFERENCES:
        pref_focus = PREFERENCES[focal]["desc"]
    else:
        pref_focus = "(no specific preference focus for this session)"
    dialogue_text = format_dialogue_as_text(session.get("messages", []) or [])
    prompt = GOLDEN_MEMORY_PROMPT.format(
        theme=theme,
        session_id=session.get("session_id", ""),
        observation_date=session.get("date", ""),
        day=session.get("day", ""),
        topic=session.get("topic", ""),
        focal_character=focal,
        characters_mentioned=", ".join(session.get("characters_mentioned", []) or []),
        topic_anchor=session.get("topic_anchor", ""),
        preference_focus=pref_focus,
        dialogue_text=dialogue_text or "(empty dialogue)",
    )
    sid = session.get("session_id") or "unknown"
    tag = f"data_prep_golden_s{story_id or '?'}_w{week or '?'}_{sid}"
    for attempt in range(MAX_RETRY):
        resp = call_gen(
            prompt, max_tokens=8192, temperature=0.2,
            tag=tag, story_id=story_id, week=week, kind="data_prep",
        )
        if not resp:
            time.sleep(1)
            continue
        try:
            parsed = parse_json_from_llm(resp)
        except Exception:
            time.sleep(1)
            continue
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return [" ".join(x.split()) for x in parsed if (x or "").strip()]
    log.warning(
        f"golden_memories fallback empty for story={story_id} week={week} session={sid}"
    )
    return []


def _generate_golden_feedback(qa, theme, *, story_id=None, week=None):
    """Legacy: 1-shot feedback fallback for old QAs missing the field."""
    prompt = f"""You are labeling a memory benchmark.
Story background: {theme}

A user asked their AI: "{qa.get('question') or ''}"
Gold answer: "{qa.get('gold_answer','')}"

Write ONE short spoken-style statement (no quotes, no markdown) that a user would say to
correct the AI when it answered wrong. It should restate the gold answer naturally so it
can be appended after "You are wrong, ".

Examples:
  gold_answer "9:00 AM" -> "the actual time is 9:00 AM"
  gold_answer "Marriott" -> "we agreed to stay at the Marriott"

Return ONLY the statement."""
    text = (
        call_gen(
            prompt, max_tokens=8192, temperature=0.3,
            tag=f"data_prep_goldenfb_s{story_id or '?'}_w{week or '?'}",
            story_id=story_id, week=week, kind="data_prep",
        )
        or ""
    ).strip()
    text = text.strip().strip('`').strip('"').strip("'").strip()
    text = " ".join(s.strip() for s in text.splitlines() if s.strip())
    return text or f"the answer is {qa.get('gold_answer','')}"


def _backfill_golden_feedback(story_dir, theme, *, story_id=None):
    qa_path = os.path.join(story_dir, "test_qa.json")
    if not os.path.exists(qa_path):
        return
    with open(qa_path) as f:
        data = json.load(f)
    changed = False
    for qa in data.get("test_questions", []):
        if not (qa.get("golden_feedback") or "").strip():
            qa["golden_feedback"] = _generate_golden_feedback(
                qa, theme, story_id=story_id, week=qa.get("test_week"),
            )
            changed = True
    if changed:
        with open(qa_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _backfill_session_golden_memories(story_dir, theme, *, story_id=None, num_weeks=None):
    if num_weeks is None:
        num_weeks = 0
        while os.path.exists(os.path.join(story_dir, f"week{num_weeks + 1}.json")):
            num_weeks += 1
    for w in range(1, num_weeks + 1):
        path = os.path.join(story_dir, f"week{w}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            wd = json.load(f)
        changed = False
        for sess in wd.get("conversations", []) or []:
            if "golden_memories" not in sess or not sess.get("golden_memories"):
                sess["golden_memories"] = _generate_session_golden_memories(
                    sess, theme, story_id=story_id, week=w,
                )
                changed = True
        if changed:
            with open(path, "w") as f:
                json.dump(wd, f, ensure_ascii=False, indent=2)


def _backfill_target_memory_refs(story_dir, *, num_weeks=None):
    qa_path = os.path.join(story_dir, "test_qa.json")
    if not os.path.exists(qa_path):
        return 0
    with open(qa_path) as f:
        data = json.load(f)
    if num_weeks is None:
        num_weeks = max(
            (qa.get("test_week", 0) for qa in data.get("test_questions", [])),
            default=0,
        )
    if num_weeks <= 0:
        return 0
    content_to_refs = _build_content_to_refs(story_dir, num_weeks=num_weeks)
    updated = 0
    for qa in data.get("test_questions", []):
        cur = _normalise_target_memory_refs(qa.get("target_memory_refs"))
        if cur:
            qa["target_memory_refs"] = cur
            continue
        triplets = content_to_refs.get(qa.get("target_memory", ""), [])
        if triplets:
            qa["target_memory_refs"] = [
                {"week": w, "session_id": s, "index": i_}
                for (w, s, i_) in triplets
            ]
            updated += 1
        else:
            qa["target_memory_refs"] = []
    if updated or any(
        "target_memory_refs" not in qa for qa in data.get("test_questions", [])
    ):
        with open(qa_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return updated


# ============================================================================
# ORCHESTRATION
# ============================================================================

def run_single_story(story_id, *, num_weeks, convs_per_week, qa_per_week):
    reset_token_usage()
    progress = load_progress()
    story_key = f"story_{story_id}"
    theme = STORY_THEMES[(story_id - 1) % len(STORY_THEMES)]

    print(f"\n{'='*50}")
    print(f"STORY {story_id}: {theme[:60]}")
    print(f"{'='*50}")

    try:
        generate_story_data(
            story_id, theme,
            num_weeks=num_weeks, convs_per_week=convs_per_week,
            qa_per_week=qa_per_week,
        )
        update_progress(lambda p: p.setdefault("stories", {})
                        .setdefault(story_key, {}).update({
            "data": "done",
            "tokens_data": get_token_usage_snapshot(),
            "num_weeks": num_weeks,
            "convs_per_week": convs_per_week,
            "qa_per_week": qa_per_week,
        }))
        return True
    except Exception as e:
        print(f"  ERROR in data generation: {e}")
        traceback.print_exc()
        update_progress(lambda p: p.setdefault("stories", {})
                        .setdefault(story_key, {}).update({"data": f"error: {str(e)}"}))
        return False


def _run_subprocess(story_id, num_weeks, convs_per_week, qa_per_week):
    rc = subprocess.run(
        [sys.executable, '-u', __file__,
         '--story', str(story_id),
         '--weeks', str(num_weeks),
         '--sessions', str(convs_per_week),
         '--qa-per-week', str(qa_per_week),
         '--no-parallel'],
        env=os.environ.copy(), cwd=os.path.dirname(os.path.abspath(__file__)),
    ).returncode
    return story_id, rc


def _validate_dataset(stories, num_weeks):
    """Schema/ref-integrity validator. No LLM calls. Non-zero exit on issues."""
    issues = 0
    for sid in stories:
        sd = story_data_dir(sid)
        if not os.path.isdir(sd):
            print(f"  [story {sid}] MISSING dir {sd}")
            issues += 1
            continue
        # outline.json is required by the new pipeline (legacy stories may
        # not have it; we report that as a single warning, not an error,
        # so legacy stories validated under --weeks=W still pass for runtime).
        out_path = os.path.join(sd, "outline.json")
        if not os.path.exists(out_path):
            print(f"  [story {sid}] (notice) outline.json missing -- "
                  f"this story predates the outline-first pipeline")
        valid_refs = set()
        for w in range(1, num_weeks + 1):
            wp = os.path.join(sd, f"week{w}.json")
            if not os.path.exists(wp):
                print(f"  [story {sid} week {w}] MISSING {wp}")
                issues += 1
                continue
            with open(wp) as f:
                wd = json.load(f)
            sessions = wd.get("conversations")
            if not isinstance(sessions, list) or not sessions:
                print(f"  [story {sid} week {w}] conversations missing/empty")
                issues += 1
                continue
            for sess in sessions:
                session_id = str(sess.get("session_id") or "")
                gm = sess.get("golden_memories")
                if not isinstance(gm, list):
                    print(
                        f"  [story {sid} week {w} session {session_id or '?'}] "
                        f"golden_memories must be list[str]"
                    )
                    issues += 1
                    continue
                for idx, item in enumerate(gm):
                    if not isinstance(item, str) or not item.strip():
                        print(
                            f"  [story {sid} week {w} session {session_id or '?'}] "
                            f"invalid golden_memories[{idx}]"
                        )
                        issues += 1
                        continue
                    valid_refs.add((w, session_id, idx))
        qp = os.path.join(sd, "test_qa.json")
        if not os.path.exists(qp):
            print(f"  [story {sid}] MISSING test_qa.json")
            issues += 1
            continue
        with open(qp) as f:
            qd = json.load(f)
        qas = qd.get("test_questions")
        if not isinstance(qas, list) or not qas:
            print(f"  [story {sid}] test_questions missing/empty")
            issues += 1
            continue
        skipped_oow = 0
        for qa in qas:
            qid = qa.get("question_id", "?")
            try:
                tw = int(qa.get("test_week"))
            except (TypeError, ValueError):
                tw = None
            if tw is not None and tw > num_weeks:
                skipped_oow += 1
                continue
            refs = _normalise_target_memory_refs(qa.get("target_memory_refs"))
            if not refs:
                print(f"  [story {sid} qa {qid}] missing/empty target_memory_refs")
                issues += 1
                continue
            for ref in refs:
                key = (ref["week"], ref["session_id"], ref["index"])
                if key not in valid_refs:
                    print(f"  [story {sid} qa {qid}] unresolved target_memory_ref {ref}")
                    issues += 1
        if skipped_oow:
            print(f"  [story {sid}] skipped {skipped_oow} QA(s) with test_week > {num_weeks} "
                  f"(out of validation window)")
    print(f"  validate: {issues} issue(s)")
    return issues


def _do_backfill_golden(stories, num_weeks):
    """Legacy mode: patch already-generated stories with missing fields."""
    for sid in stories:
        sd = story_data_dir(sid)
        if not os.path.isdir(sd):
            print(f"[story {sid}] no data dir, skipping")
            continue
        theme = STORY_THEMES[(sid - 1) % len(STORY_THEMES)]
        print(f"[story {sid}] backfilling golden_memories ...")
        _backfill_session_golden_memories(sd, theme, story_id=sid, num_weeks=num_weeks)
        print(f"[story {sid}] backfilling golden_feedback ...")
        _backfill_golden_feedback(sd, theme, story_id=sid)
        print(f"[story {sid}] backfilling target_memory_refs ...")
        n = _backfill_target_memory_refs(sd, num_weeks=num_weeks)
        print(f"[story {sid}] backfilled {n} QA refs")


def main():
    parser = argparse.ArgumentParser(description="AdaMem data preparation (outline-first)")
    parser.add_argument('--story', type=int, help='Run single story only (1-indexed)')
    parser.add_argument('--stories', type=int, default=NUM_STORIES_DEFAULT,
                        help=f'Number of stories (default: {NUM_STORIES_DEFAULT})')
    parser.add_argument('--weeks', type=int, default=NUM_WEEKS_DEFAULT,
                        help=f'Weeks per story (default: {NUM_WEEKS_DEFAULT})')
    parser.add_argument('--sessions', type=int, default=CONVS_PER_WEEK_DEFAULT,
                        help=f'Conversation sessions per week (default: {CONVS_PER_WEEK_DEFAULT})')
    parser.add_argument('--qa-per-week', type=int, default=QA_PER_WEEK_DEFAULT,
                        help=f'QA per week (default: {QA_PER_WEEK_DEFAULT})')
    parser.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL,
                        help=f'Parallel story workers (default: {DEFAULT_PARALLEL})')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Internal: run inline (used by subprocess workers).')
    parser.add_argument('--validate', action='store_true',
                        help='Schema/ref-integrity check only. No LLM calls.')
    parser.add_argument('--backfill-golden', action='store_true',
                        help='Legacy patcher: fill missing golden_memories / '
                             'golden_feedback / target_memory_refs on old data.')
    args = parser.parse_args()

    if args.story:
        story_ids = [args.story]
    else:
        story_ids = list(range(1, args.stories + 1))

    if args.validate:
        rc = _validate_dataset(story_ids, args.weeks)
        sys.exit(1 if rc else 0)

    if args.backfill_golden:
        _do_backfill_golden(story_ids, args.weeks)
        rc = _validate_dataset(story_ids, args.weeks)
        sys.exit(1 if rc else 0)

    if args.story:
        run_single_story(args.story,
                         num_weeks=args.weeks,
                         convs_per_week=args.sessions,
                         qa_per_week=args.qa_per_week)
        return

    print("AdaMem data preparation (outline-first)")
    print(f"  Stories: {args.stories}")
    print(f"  Weeks per story: {args.weeks}")
    print(f"  Sessions per week: {args.sessions}")
    print(f"  QA per week: {args.qa_per_week}")
    print(f"  Output: {DATA_DIR}")
    print(f"  Parallel: {args.parallel}")
    print("=" * 60)

    progress = load_progress()
    pending = []
    for sid in range(1, args.stories + 1):
        st = progress.get("stories", {}).get(f"story_{sid}", {})
        if st.get("data") == "done":
            print(f"[Story {sid}] data ✓")
            continue
        pending.append(sid)
    if not pending:
        print("All stories already prepared.")
        return

    print(f"Pending: {pending}  (parallel={args.parallel})")
    start = time.time()
    if args.parallel <= 1 or args.no_parallel:
        for sid in pending:
            run_single_story(sid,
                             num_weeks=args.weeks,
                             convs_per_week=args.sessions,
                             qa_per_week=args.qa_per_week)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(_run_subprocess, sid, args.weeks, args.sessions, args.qa_per_week): sid
                       for sid in pending}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    sid, rc = fut.result()
                    print(f"[Story {sid}] {'Completed ✓' if rc == 0 else f'Failed ✗ (exit {rc})'}")
                except Exception as e:
                    print(f"[Story {sid}] Worker exception: {e}")
    elapsed = (time.time() - start) / 60
    print(f"\nALL DONE in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
