#!/usr/bin/env python3
"""A4 -- Does each policy update match the TRUE preference? (convergence)

For each AdaMem run we replay ``policy_snapshots.json``: week 0 (init) and the
``new_policy`` after every weekly reflection. For each (story, week) we ask the
judge to score, per character, how well the LEARNED by_character policy line
matches that character's GROUND-TRUTH preference (preference_desc from
character_profiles.json). Score 0/1/2 -> normalized to [0,1].

The per-week mean alignment (averaged over 6 characters and over stories) is a
**policy-convergence curve**; comparing verbose vs with_gold answers A5 (does
explicit preference feedback let the model learn the right policy faster than
implicit correctness-only feedback?).

Scope: adamem cells. Outputs:
  out/a4_policy_alignment_byweek.csv   (model x fbmode x week -> mean alignment)
  out/a4_policy_alignment_bychar.csv   (final-week per-character alignment)
  out/a4_policy_alignment.json
  out/a4_policy_convergence.png
"""

import os
import json
import argparse
import collections

import _common as C
import _judge_common as J

PROMPT = """You are scoring how well a LEARNED memory-extraction preference
matches the TRUE preference for each person.

For each person you are given the TRUE preference (what an ideal system should
focus on for them) and the LEARNED preference the system wrote down. Score the
LEARNED one:
  2 = captures the true preference (same focus; extra wording is fine)
  1 = partially overlaps but misses or dilutes the core focus
  0 = wrong, contradictory, generic, or missing

## Persons
{block}

Return ONE line of JSON only, mapping each person name to an integer score:
{{"<person>": 0|1|2, ...}}"""


def _policy_by_week(snap_path):
    """Return {week: {canonical_char: learned_text}} for week 0..N."""
    if not os.path.exists(snap_path):
        return {}
    d = json.load(open(snap_path, encoding="utf-8"))
    out = {}
    for item in d.get("history", []) or []:
        w = item.get("week")
        if item.get("phase") == "init":
            pol = item.get("policy") or {}
        else:
            pol = item.get("new_policy") or {}
        bc = {}
        for name, txt in (pol.get("by_character") or {}).items():
            bc[J.normalize_char(name)] = txt or ""
        out[w] = bc
    return out


def run(model="deepseek-v4-flash", fbmode="verbose"):
    method = "adamem"
    stories = C.list_stories(method, model, fbmode)
    if not stories:
        print(f"[A4] no adamem cell for {C.short_model(model)}/{fbmode}; skip")
        return None

    # Build judge jobs: one per (story, week).
    jobs = []  # (story, week, [chars])
    prompts = []
    for sid in stories:
        gt = J.load_char_profiles(sid)  # canonical char -> true pref
        pbw = _policy_by_week(f"{C.cell_dir(method, model, fbmode)}/story_{sid}/policy_snapshots.json")
        for w in sorted(pbw):
            learned = pbw[w]
            chars = [c for c in J.GT_CHARACTERS if gt.get(c)]
            block = "\n".join(
                f"- {c}\n    TRUE: {gt.get(c,'')}\n    LEARNED: {J.short(learned.get(c,'') or '(none)', 300)}"
                for c in chars)
            prompts.append(PROMPT.format(block=block))
            jobs.append((sid, w, chars))
    print(f"[A4] {C.short_model(model)}/{fbmode}: {len(jobs)} (story,week) judgements...", flush=True)
    parsed = J.judge_many(prompts) if prompts else []

    # week -> [sum_norm_score, count]
    by_week = collections.defaultdict(lambda: [0.0, 0])
    # (final week) char -> [sum, count]
    final_char = collections.defaultdict(lambda: [0.0, 0])
    max_week = max((w for _, w, _ in jobs), default=0)
    for (sid, w, chars), p in zip(jobs, parsed):
        if not p:
            continue
        for c in chars:
            try:
                sc = int(p.get(c))
            except Exception:
                continue
            norm = max(0.0, min(2, sc)) / 2.0
            by_week[w][0] += norm
            by_week[w][1] += 1
            if w == max_week:
                final_char[c][0] += norm
                final_char[c][1] += 1

    result = {"model": C.short_model(model), "fbmode": fbmode,
              "by_week": {w: (by_week[w][0] / by_week[w][1] if by_week[w][1] else None)
                          for w in sorted(by_week)},
              "final_by_char": {c: (final_char[c][0] / final_char[c][1] if final_char[c][1] else None)
                                for c in J.GT_CHARACTERS if final_char[c][1]}}
    return result


def run_all():
    C.ensure_out()
    results = {}
    for model in C.MODELS:
        for fbmode in C.FBMODES:
            r = run(model, fbmode)
            if r:
                results[f"{r['model']}|{fbmode}"] = r

    # byweek csv
    weeks = sorted({int(w) for r in results.values() for w in r["by_week"]})
    bw_rows = []
    for key, r in results.items():
        for w in sorted(r["by_week"]):
            v = r["by_week"][w]
            bw_rows.append([r["model"], r["fbmode"], w,
                            round(100 * v, 1) if v is not None else ""])
    C.write_csv(f"{C.OUT_DIR}/a4_policy_alignment_byweek.csv",
                ["model", "fbmode", "week", "alignment_pct"], bw_rows)

    # final per-char csv
    fc_rows = []
    for key, r in results.items():
        for c, v in r["final_by_char"].items():
            fc_rows.append([r["model"], r["fbmode"], c,
                            round(100 * v, 1) if v is not None else ""])
    C.write_csv(f"{C.OUT_DIR}/a4_policy_alignment_bychar.csv",
                ["model", "fbmode", "character", "alignment_pct"], fc_rows)
    C.write_json(f"{C.OUT_DIR}/a4_policy_alignment.json", results)

    _plot(results)

    print("\n[A4] policy-vs-truth alignment (final week, mean over chars)")
    print(f"{'model':9}{'fbmode':10}{'final_align%':>13}")
    for key, r in results.items():
        bw = r["by_week"]
        last = bw[max(bw)] if bw else None
        print(f"{r['model']:9}{r['fbmode']:10}{(100*last if last is not None else 0):13.1f}")
    print(f"  -> {C.OUT_DIR}/a4_policy_alignment_byweek.csv / _bychar.csv / .json / .png")


def _plot(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[A4] matplotlib unavailable ({e})")
        return
    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    })
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"verbose": "-", "with_gold": "--"}
    colors = {"deepseek": "#1f77b4", "gemini": "#ff7f0e"}
    fb_label = {"verbose": "Explicit", "with_gold": "Implicit"}
    model_label = {"deepseek": "DeepSeek", "gemini": "Gemini"}
    for key, r in results.items():
        bw = {int(w): v for w, v in r["by_week"].items()}
        ws = sorted(bw)
        ys = [100 * bw[w] if bw[w] is not None else None for w in ws]
        ax.plot(ws, ys, styles.get(r["fbmode"], "-"), marker="o",
                color=colors.get(r["model"]),
                label=f"{model_label.get(r['model'], r['model'])} / {fb_label.get(r['fbmode'], r['fbmode'])}")
    ax.set_xlabel("Week"); ax.set_ylabel("Policy alignment (%)")
    ax.set_title("Policy Convergence to Ground Truth")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{C.OUT_DIR}/a4_policy_convergence.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--fbmode", default=None)
    a = ap.parse_args()
    if a.model and a.fbmode:
        print(run(a.model, a.fbmode))
    else:
        run_all()
