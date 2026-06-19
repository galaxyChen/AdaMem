#!/usr/bin/env python3
"""A3 -- Policy-following ability.

For each AdaMem daily add-call, the policy in effect was injected as
``custom_instructions``. We ask the judge: of the memories the system actually
extracted, how many are the KIND of information the policy says to keep vs.
noise the policy says to ignore? This measures whether the model *obeys* the
policy (orthogonal to whether the policy itself is correct -- that's A4).

Scope: adamem cells only (they carry custom_instructions). All models/fbmodes.

Outputs:
  out/a3_policy_following.csv         (per cell: overall + helps interpret A6)
  out/a3_policy_following_byweek.csv  (per cell x week)
  out/a3_policy_following.json
"""

import os
import json
import argparse
import collections

import _common as C
import _judge_common as J

PROMPT = """You are auditing whether a memory-extraction system FOLLOWED its
extraction policy on one conversation.

## Extraction policy that was injected into the system
{policy}

## Memories the system actually extracted (numbered)
{memories}

## Task
The policy describes, per person, what KIND of information to keep -- and
implicitly what to ignore as noise. For EACH extracted memory decide:
- follow  : it is the kind of information the policy says to KEEP for the
            relevant person;
- violate : it is information the policy indicates should be ignored / not
            prioritized (irrelevant chit-chat, or a category the policy says
            not to track for that person).

Return ONE line of JSON only:
{{"follow_indices": [int, ...], "violate_indices": [int, ...], "reason": "<<= 30 words>"}}
Every index 0..N-1 must appear in exactly one of the two lists."""


def _collect_jobs(method, model, fbmode):
    """Return list of (week, [memory_texts], policy_text) for dialogue adds."""
    jobs = []
    for sid in C.list_stories(method, model, fbmode):
        path = f"{C.cell_dir(method, model, fbmode)}/story_{sid}/extracted_memories.jsonl"
        for row in C.iter_jsonl(path):
            if row.get("source") != "dialogue":
                continue
            ci = (row.get("custom_instructions") or "").strip()
            mems = [(it.get("memory") or "").strip()
                    for it in (row.get("added") or []) if (it.get("memory") or "").strip()]
            if not ci or not mems:
                continue
            jobs.append((row.get("week") or 0, mems, ci))
    return jobs


def run(model="deepseek-v4-flash", fbmode="verbose"):
    C.ensure_out()
    method = "adamem"
    if not C.list_stories(method, model, fbmode):
        print(f"[A3] no adamem cell for {C.short_model(model)}/{fbmode}; skip")
        return None
    jobs = _collect_jobs(method, model, fbmode)
    prompts = []
    for _, mems, policy in jobs:
        numbered = "\n".join(f"[{i}] {J.short(m, 300)}" for i, m in enumerate(mems))
        prompts.append(PROMPT.format(policy=J.short(policy, 4000), memories=numbered))
    print(f"[A3] {C.short_model(model)}/{fbmode}: {len(jobs)} add-calls -> judging...", flush=True)
    parsed = J.judge_many(prompts) if prompts else []

    by_week = collections.defaultdict(lambda: [0, 0])  # week -> [follow, total]
    n_parsed = 0
    for (week, mems, _), p in zip(jobs, parsed):
        if not p:
            continue
        n_parsed += 1
        follow = p.get("follow_indices") or []
        violate = p.get("violate_indices") or []
        try:
            n_follow = len([i for i in follow if 0 <= int(i) < len(mems)])
            n_total = n_follow + len([i for i in violate if 0 <= int(i) < len(mems)])
        except Exception:
            continue
        if n_total == 0:
            n_total = len(mems)
            n_follow = min(n_follow, n_total)
        by_week[week][0] += n_follow
        by_week[week][1] += n_total

    tot_f = sum(v[0] for v in by_week.values())
    tot_t = sum(v[1] for v in by_week.values())
    overall = C.pct(tot_f, tot_t)
    result = {"model": C.short_model(model), "fbmode": fbmode,
              "add_calls": len(jobs), "judged": n_parsed,
              "follow_rate": overall,
              "by_week": {w: C.pct(by_week[w][0], by_week[w][1]) for w in sorted(by_week)}}
    fr_str = f"{overall:.1f}%" if overall is not None else "n/a"
    print(f"      follow_rate = {fr_str}  (judged {n_parsed}/{len(jobs)})")
    return result, by_week


def run_all():
    C.ensure_out()
    rows, byweek_rows, jobj = [], [], {}
    for model in C.MODELS:
        for fbmode in C.FBMODES:
            r = run(model, fbmode)
            if not r:
                continue
            res, by_week = r
            key = f"{res['model']}|{fbmode}"
            jobj[key] = res
            rows.append([res["model"], fbmode, res["add_calls"], res["judged"],
                         round(res["follow_rate"], 1) if res["follow_rate"] is not None else ""])
            for w in sorted(by_week):
                fr = C.pct(by_week[w][0], by_week[w][1])
                byweek_rows.append([res["model"], fbmode, w,
                                    round(fr, 1) if fr is not None else "",
                                    by_week[w][1]])
    C.write_csv(f"{C.OUT_DIR}/a3_policy_following.csv",
                ["model", "fbmode", "add_calls", "judged", "follow_rate"], rows)
    C.write_csv(f"{C.OUT_DIR}/a3_policy_following_byweek.csv",
                ["model", "fbmode", "week", "follow_rate", "n_memories"], byweek_rows)
    C.write_json(f"{C.OUT_DIR}/a3_policy_following.json", jobj)
    print("\n[A3] summary (policy-following rate)")
    print(f"{'model':9}{'fbmode':10}{'add_calls':>10}{'follow%':>9}")
    for r in rows:
        print(f"{r[0]:9}{r[1]:10}{r[2]:10d}{(r[4] if r[4] != '' else 0):9.1f}")
    print(f"  -> {C.OUT_DIR}/a3_policy_following.csv / _byweek.csv / .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="single model; default all")
    ap.add_argument("--fbmode", default=None, help="single fbmode; default all")
    a = ap.parse_args()
    if a.model and a.fbmode:
        run(a.model, a.fbmode)
    else:
        run_all()
