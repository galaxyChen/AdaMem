#!/usr/bin/env python3
"""A7 -- Accuracy broken down by qa_type.

qa_type distinguishes ``within_pref`` questions from cross-character
distractor types. AdaMem is designed to resist dilution from irrelevant
characters, so its gain is expected to concentrate on distractor questions.

Computed for every available cell (method x model x fbmode), micro-averaged
over stories. Also emits an M0-vs-AdaMem delta view per (model, fbmode).

Outputs:
  out/a7_by_qa_type.csv         (long: one row per cell x qa_type)
  out/a7_delta_by_qa_type.csv   (m0 vs adamem delta per model/fb/qa_type)
  out/a7_by_qa_type.json
"""

import collections
import _common as C


def collect():
    # (method, model, fb, qa_type) -> [correct, total]
    agg = collections.defaultdict(lambda: [0, 0])
    qa_types = set()
    for method, model, fb in C.iter_cells():
        for sid in C.list_stories(method, model, fb):
            recs = C.load_qa_records(method, model, fb, sid) or []
            for r in recs:
                qt = r.get("qa_type") or "unknown"
                qa_types.add(qt)
                cell = (method, C.short_model(model), fb, qt)
                agg[cell][1] += 1
                agg[cell][0] += 1 if r.get("correct") else 0
    return agg, sorted(qa_types)


def run():
    C.ensure_out()
    agg, qa_types = collect()

    rows = []
    json_obj = collections.defaultdict(dict)
    for (method, model, fb, qt), (cor, tot) in sorted(agg.items()):
        a = C.pct(cor, tot)
        rows.append([method, model, fb, qt, round(a, 1) if a is not None else "", tot])
        json_obj[f"{method}|{model}|{fb}"][qt] = {"acc": a, "n": tot}
    C.write_csv(f"{C.OUT_DIR}/a7_by_qa_type.csv",
                ["method", "model", "fbmode", "qa_type", "acc", "n"], rows)
    C.write_json(f"{C.OUT_DIR}/a7_by_qa_type.json", json_obj)

    # delta view: adamem - m0 per (model, fb, qa_type)
    drows = []
    models = sorted({m for (_, m, _, _) in agg})
    fbmodes = sorted({fb for (_, _, fb, _) in agg})
    for model in models:
        for fb in fbmodes:
            for qt in qa_types:
                m0 = agg.get(("m0", model, fb, qt))
                ad = agg.get(("adamem", model, fb, qt))
                if not m0 or not ad:
                    continue
                a_m0 = C.pct(m0[0], m0[1])
                a_ad = C.pct(ad[0], ad[1])
                drows.append([model, fb, qt, round(a_m0, 1), round(a_ad, 1),
                              round(a_ad - a_m0, 1), m0[1]])
    C.write_csv(f"{C.OUT_DIR}/a7_delta_by_qa_type.csv",
                ["model", "fbmode", "qa_type", "m0_acc", "adamem_acc", "delta", "n"], drows)

    print("[A7] accuracy by qa_type (adamem - m0 delta)")
    print(f"{'model':9}{'fbmode':10}{'qa_type':22}{'m0':>7}{'adamem':>8}{'delta':>7}{'n':>6}")
    for r in drows:
        print(f"{r[0]:9}{r[1]:10}{r[2]:22}{r[3]:7.1f}{r[4]:8.1f}{r[5]:+7.1f}{r[6]:6d}")
    print(f"  -> {C.OUT_DIR}/a7_by_qa_type.csv / a7_delta_by_qa_type.csv / .json")


if __name__ == "__main__":
    run()
