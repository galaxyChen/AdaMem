#!/usr/bin/env python3
"""Main paper table: Accuracy + Extraction-F1 + Memory-Efficiency-Ratio (MER).

All three columns are in [0,1] and "higher is better":

  Accuracy       mean of qa_records[*].correct
  Extraction F1  harmonic mean of
                   P = extraction precision (is_in_golden / total extracted)
                   R = extraction recall   (union of golden captured / total golden)
                 -- "preference precision/coverage", NOT factual correctness.
  MER            (Acc / MemVol) / (Acc_ID / MemVol_ID), ID-normalized so the
                 Ideal-Memory baseline = 1.0. Measures how efficiently stored
                 memory translates into accuracy, relative to a perfect lean
                 memory store. MemVol = total added memories per story
                 (dialogue + QA writeback).

FC has no memory store -> F1/MER = None. ID is the MER reference (=1.0).

Outputs:
  out/main_table.csv      (per cell)
  out/main_table.json     (per cell + marginals)
  out/main_table_marginals.csv
"""

import os
import glob
import json
import collections

import _common as C

DS = "deepseek-v4-flash"


def _golden_total(story):
    n = 0
    for wf in glob.glob(os.path.join(C.DATA_ROOT, f"story_{story}", "week*.json")):
        d = json.load(open(wf, encoding="utf-8"))
        for c in d.get("conversations", []):
            n += len(c.get("golden_memories", []) or [])
    return n


def cell_metrics(method, model, fbmode):
    base = C.cell_dir(method, model, fbmode)
    if not os.path.isdir(base):
        return None
    tot = cor = 0
    ej_in = ej_tot = 0
    vol = 0
    rec_union = []
    stories = C.list_stories(method, model, fbmode)
    for sid in stories:
        recs = C.load_qa_records(method, model, fbmode, sid) or []
        for r in recs:
            tot += 1
            cor += 1 if r.get("correct") else 0
        matched = set()
        for r in C.iter_jsonl(os.path.join(base, f"story_{sid}", "extraction_judged.jsonl")):
            ej_tot += 1
            if r.get("is_in_golden"):
                ej_in += 1
                if r.get("matched_golden_global_index") is not None:
                    matched.add(r["matched_golden_global_index"])
        for row in C.iter_jsonl(os.path.join(base, f"story_{sid}", "extracted_memories.jsonl")):
            vol += len(row.get("added") or [])
        gt = _golden_total(sid)
        if ej_tot and gt:
            rec_union.append(len(matched) / gt)
    acc = cor / tot if tot else None
    P = ej_in / ej_tot if ej_tot else None
    R = (sum(rec_union) / len(rec_union)) if rec_union else None
    F1 = (2 * P * R / (P + R)) if (P and R) else None
    vol_st = vol / len(stories) if stories else 0
    return {"accuracy": acc, "precision": P, "recall": R, "f1": F1,
            "mem_vol": vol_st, "n_story": len(stories), "n_qa": tot}


def run():
    C.ensure_out()
    # ID reference per fbmode (model-independent; answer model fixed deepseek).
    id_ref = {}
    for fb in C.FBMODES:
        m = cell_metrics("id", DS, fb)
        if m and m["mem_vol"]:
            id_ref[fb] = (m["accuracy"], m["mem_vol"])

    cells = {}
    for method, model, fb in C.iter_cells():
        m = cell_metrics(method, model, fb)
        if not m:
            continue
        mer = None
        if m["accuracy"] is not None and m["mem_vol"] and fb in id_ref:
            aid, vid = id_ref[fb]
            mer = (m["accuracy"] / m["mem_vol"]) / (aid / vid)
        m["mer"] = mer
        cells[(method, C.short_model(model), fb)] = m

    # ---- per-cell CSV ----
    rows = []
    for (method, model, fb), m in sorted(cells.items()):
        rows.append([method, model, fb,
                     _r(m["accuracy"]), _r(m["f1"]), _r3(m["mer"]),
                     _r(m["precision"]), _r(m["recall"]), round(m["mem_vol"]),
                     m["n_story"]])
    C.write_csv(f"{C.OUT_DIR}/main_table.csv",
                ["method", "model", "fbmode", "accuracy", "extraction_f1", "mer",
                 "precision", "recall", "mem_vol", "n_story"], rows)

    # ---- marginals (mean over cells) ----
    # method axis: over all methods. model/fbmode axes: restrict to the
    # comparison methods {m0, adamem} so the average is apples-to-apples
    # (id/fc exist only for deepseek and would skew a by-model marginal).
    def marg(group_idx, allowed_methods=None):
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        keymap = {"method": 0, "model": 1, "fbmode": 2}
        gi = keymap[group_idx]
        for key, m in cells.items():
            if allowed_methods and key[0] not in allowed_methods:
                continue
            for metric in ("accuracy", "f1", "mer"):
                if m.get(metric) is not None:
                    agg[key[gi]][metric].append(m[metric])
        return {k: {mt: sum(v) / len(v) for mt, v in d.items()} for k, d in agg.items()}

    marg_rows = []
    marginals = {}
    axis_methods = {"method": None, "model": {"m0", "adamem"}, "fbmode": {"m0", "adamem"}}
    for axis in ("method", "model", "fbmode"):
        mm = marg(axis, axis_methods[axis])
        marginals[axis] = mm
        for k, d in mm.items():
            marg_rows.append([axis, k, _r(d.get("accuracy")), _r(d.get("f1")), _r3(d.get("mer"))])
    C.write_csv(f"{C.OUT_DIR}/main_table_marginals.csv",
                ["axis", "value", "accuracy", "extraction_f1", "mer"], marg_rows)

    C.write_json(f"{C.OUT_DIR}/main_table.json",
                 {"id_ref": id_ref,
                  "cells": {f"{a}|{b}|{c}": v for (a, b, c), v in cells.items()},
                  "marginals": marginals})

    # ---- stdout ----
    print("[main_table] Accuracy / Extraction-F1 / MER (all in [0,1], higher better)")
    print(f"{'method':7}{'model':9}{'fb':10}{'Acc':>7}{'F1':>7}{'MER':>7}{'(P':>7}{'R':>6}{'Vol)':>7}")
    for r in rows:
        print(f"{r[0]:7}{r[1]:9}{r[2]:10}"
              f"{_s(r[3])}{_s(r[4])}{_s3(r[5])}{_s(r[6])}{_s(r[7])}{r[8]:7}")
    print("\nMarginals (mean):")
    for r in marg_rows:
        print(f"  {r[0]:7} {r[1]:30} Acc={_s(r[2])} F1={_s(r[3])} MER={_s3(r[4])}")
    print(f"\n  -> {C.OUT_DIR}/main_table.csv / _marginals.csv / .json")


def _r(x):
    return round(100 * x, 1) if x is not None else None


def _r3(x):
    return round(x, 3) if x is not None else None


def _s(x):
    return f"{x:6.1f}" if x is not None else "    — "


def _s3(x):
    return f"{x:6.3f}" if x is not None else "    — "


if __name__ == "__main__":
    run()
