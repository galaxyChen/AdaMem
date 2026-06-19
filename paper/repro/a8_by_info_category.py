#!/usr/bin/env python3
"""A8 -- Accuracy broken down by info_category.

info_category is one of Decision/Conclusion, Fact/Number, Agreement/Promise,
Emotion/Attitude, Schedule/Time -- each maps to a preference archetype. This
shows which preference types AdaMem helps most. info_category lives in
test_qa.json, joined to qa_records by question_id.

Outputs:
  out/a8_by_info_category.csv
  out/a8_delta_by_info_category.csv
  out/a8_by_info_category.json
"""

import collections
import _common as C


def collect():
    agg = collections.defaultdict(lambda: [0, 0])  # (method, model, fb, cat) -> [cor, tot]
    cats = set()
    for method, model, fb in C.iter_cells():
        for sid in C.list_stories(method, model, fb):
            meta = C.load_test_qa_meta(sid)
            recs = C.load_qa_records(method, model, fb, sid) or []
            for r in recs:
                qid = r.get("question_id")
                cat = (meta.get(qid) or {}).get("info_category") or "unknown"
                cats.add(cat)
                cell = (method, C.short_model(model), fb, cat)
                agg[cell][1] += 1
                agg[cell][0] += 1 if r.get("correct") else 0
    return agg, sorted(cats)


def run():
    C.ensure_out()
    agg, cats = collect()

    rows = []
    json_obj = collections.defaultdict(dict)
    for (method, model, fb, cat), (cor, tot) in sorted(agg.items()):
        a = C.pct(cor, tot)
        rows.append([method, model, fb, cat, round(a, 1) if a is not None else "", tot])
        json_obj[f"{method}|{model}|{fb}"][cat] = {"acc": a, "n": tot}
    C.write_csv(f"{C.OUT_DIR}/a8_by_info_category.csv",
                ["method", "model", "fbmode", "info_category", "acc", "n"], rows)
    C.write_json(f"{C.OUT_DIR}/a8_by_info_category.json", json_obj)

    drows = []
    models = sorted({m for (_, m, _, _) in agg})
    fbmodes = sorted({fb for (_, _, fb, _) in agg})
    for model in models:
        for fb in fbmodes:
            for cat in cats:
                m0 = agg.get(("m0", model, fb, cat))
                ad = agg.get(("adamem", model, fb, cat))
                if not m0 or not ad:
                    continue
                a_m0 = C.pct(m0[0], m0[1])
                a_ad = C.pct(ad[0], ad[1])
                drows.append([model, fb, cat, round(a_m0, 1), round(a_ad, 1),
                              round(a_ad - a_m0, 1), m0[1]])
    C.write_csv(f"{C.OUT_DIR}/a8_delta_by_info_category.csv",
                ["model", "fbmode", "info_category", "m0_acc", "adamem_acc", "delta", "n"], drows)

    print("[A8] accuracy by info_category (adamem - m0 delta)")
    print(f"{'model':9}{'fbmode':10}{'info_category':22}{'m0':>7}{'adamem':>8}{'delta':>7}{'n':>6}")
    for r in drows:
        print(f"{r[0]:9}{r[1]:10}{r[2]:22}{r[3]:7.1f}{r[4]:8.1f}{r[5]:+7.1f}{r[6]:6d}")
    print(f"  -> {C.OUT_DIR}/a8_by_info_category.csv / a8_delta_by_info_category.csv / .json")


if __name__ == "__main__":
    run()
