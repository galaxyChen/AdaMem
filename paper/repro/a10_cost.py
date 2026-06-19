#!/usr/bin/env python3
"""A10 -- Cost / benefit accounting from token_usage.

Aggregates qa_records[*].token_usage across stories per cell, broken into
components (mem0_llm = extraction, reflect = AdaMem policy update, answer,
judge, embedding). Reports total tokens, tokens per QA, and -- for M0 vs
AdaMem -- the extra reflection overhead against the accuracy gain.

Outputs:
  out/a10_cost.csv            (per cell x component)
  out/a10_cost_summary.csv    (m0 vs adamem: extra cost vs acc gain)
  out/a10_cost.json
"""

import collections
import _common as C

COMPONENTS = ["embedding", "mem0_llm", "answer", "judge", "reflect"]


def collect():
    # (method, model, fb) -> {component -> total_tokens}, n_qa, acc(correct,total)
    agg = {}
    for method, model, fb in C.iter_cells():
        comp = collections.defaultdict(int)
        n_qa = 0
        cor = tot = 0
        n_story = 0
        for sid in C.list_stories(method, model, fb):
            blob = C.load_qa_meta_blob(method, model, fb, sid)
            if not blob:
                continue
            n_story += 1
            tu = blob.get("token_usage") or {}
            for k in COMPONENTS:
                comp[k] += (tu.get(k) or {}).get("total", 0)
            recs = blob.get("qa_records", [])
            n_qa += len(recs)
            for r in recs:
                tot += 1
                cor += 1 if r.get("correct") else 0
        agg[(method, C.short_model(model), fb)] = {
            "components": dict(comp),
            "total_tokens": sum(comp.values()),
            "n_qa": n_qa,
            "n_story": n_story,
            "accuracy": C.pct(cor, tot),
        }
    return agg


def run():
    C.ensure_out()
    agg = collect()

    rows = []
    for (method, model, fb), d in sorted(agg.items()):
        comp = d["components"]
        per_qa = (d["total_tokens"] / d["n_qa"]) if d["n_qa"] else 0
        rows.append([method, model, fb, d["n_story"], d["n_qa"],
                     round(d["accuracy"], 1) if d["accuracy"] is not None else "",
                     comp.get("embedding", 0), comp.get("mem0_llm", 0),
                     comp.get("answer", 0), comp.get("judge", 0), comp.get("reflect", 0),
                     d["total_tokens"], round(per_qa, 1)])
    C.write_csv(f"{C.OUT_DIR}/a10_cost.csv",
                ["method", "model", "fbmode", "n_story", "n_qa", "acc",
                 "embedding", "mem0_llm", "answer", "judge", "reflect",
                 "total_tokens", "tokens_per_qa"], rows)
    C.write_json(f"{C.OUT_DIR}/a10_cost.json", {f"{m}|{mm}|{fb}": d
                                                for (m, mm, fb), d in agg.items()})

    # m0 vs adamem: extra reflection cost vs accuracy gain.
    # Compare on extraction+reflection tokens (the memory-construction budget),
    # excluding the shared answer/judge/embedding which are method-agnostic.
    srows = []
    models = sorted({mm for (_, mm, _) in agg})
    fbmodes = sorted({fb for (_, _, fb) in agg})
    for model in models:
        for fb in fbmodes:
            m0 = agg.get(("m0", model, fb))
            ad = agg.get(("adamem", model, fb))
            if not m0 or not ad:
                continue
            m0_build = m0["components"].get("mem0_llm", 0)
            ad_build = ad["components"].get("mem0_llm", 0) + ad["components"].get("reflect", 0)
            extra = ad_build - m0_build
            extra_pct = C.pct(extra, m0_build)
            acc_gain = (ad["accuracy"] - m0["accuracy"]) if (
                ad["accuracy"] is not None and m0["accuracy"] is not None) else None
            srows.append([model, fb,
                          m0_build, ad_build, extra,
                          round(extra_pct, 1) if extra_pct is not None else "",
                          ad["components"].get("reflect", 0),
                          round(m0["accuracy"], 1), round(ad["accuracy"], 1),
                          round(acc_gain, 1) if acc_gain is not None else ""])
    C.write_csv(f"{C.OUT_DIR}/a10_cost_summary.csv",
                ["model", "fbmode", "m0_build_tok", "adamem_build_tok", "extra_tok",
                 "extra_pct", "reflect_tok", "m0_acc", "adamem_acc", "acc_gain"], srows)

    print("[A10] memory-construction cost vs accuracy gain (m0 vs adamem)")
    print(f"{'model':9}{'fbmode':10}{'m0_build':>11}{'ada_build':>11}{'extra%':>8}"
          f"{'reflect':>10}{'acc_gain':>9}")
    for r in srows:
        print(f"{r[0]:9}{r[1]:10}{r[2]:11d}{r[3]:11d}{(r[5] or 0):8.1f}{r[6]:10d}"
              f"{(r[9] if r[9] != '' else 0):+9.1f}")
    print(f"  -> {C.OUT_DIR}/a10_cost.csv / a10_cost_summary.csv / .json")


if __name__ == "__main__":
    run()
