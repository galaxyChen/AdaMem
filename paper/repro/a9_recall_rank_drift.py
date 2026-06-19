#!/usr/bin/env python3
"""A9 -- Retrieval quality drift over weeks.

As the memory store grows, does the target memory sink in the ranked recall
list? Per-week Recall@k / MRR / mean-rank-of-hits for ID / M0 / AdaMem.
Requires recall_judged.jsonl (only cells already judged -- default
deepseek/verbose). FC is excluded (no real retrieval).

Outputs:
  out/a9_recall_rank_drift_<model>_<fb>.csv
  out/a9_recall_rank_drift_<model>_<fb>.json
  out/a9_recall_rank_drift_<model>_<fb>.png
"""

import argparse
import collections
import _common as C


def per_week_recall(method, model, fbmode, k=5):
    # week -> dict accumulators
    w_hit_k = collections.defaultdict(int)
    w_tot = collections.defaultdict(int)
    w_mrr = collections.defaultdict(float)
    w_rank_sum = collections.defaultdict(float)
    w_rank_n = collections.defaultdict(int)
    found = False
    for sid in C.list_stories(method, model, fbmode):
        path = f"{C.cell_dir(method, model, fbmode)}/story_{sid}/recall_judged.jsonl"
        for row in C.iter_jsonl(path):
            found = True
            if row.get("skipped"):
                continue
            w = row.get("week")
            w_tot[w] += 1
            if row.get("target_in_recall"):
                rank = row.get("target_rank")
                if rank:
                    w_mrr[w] += 1.0 / rank
                    if rank <= k:
                        w_hit_k[w] += 1
                    w_rank_sum[w] += rank
                    w_rank_n[w] += 1
    if not found:
        return None
    out = {}
    for w in sorted(w_tot):
        out[w] = {
            "recall_at_k": C.pct(w_hit_k[w], w_tot[w]),
            "mrr": (100.0 * w_mrr[w] / w_tot[w]) if w_tot[w] else None,
            "mean_hit_rank": (w_rank_sum[w] / w_rank_n[w]) if w_rank_n[w] else None,
            "n": w_tot[w],
        }
    return out


def run(model="deepseek-v4-flash", fbmode="verbose", k=5):
    C.ensure_out()
    short = C.short_model(model)
    methods = []
    data = {}
    for m in ("id", "m0", "adamem"):
        d = per_week_recall(m, model, fbmode, k)
        if d:
            methods.append(m)
            data[m] = d
    if not methods:
        print(f"[A9] no recall_judged for {short}/{fbmode}; skip")
        return
    weeks = sorted({w for m in data for w in data[m]})

    header = ["week"]
    for m in methods:
        header += [f"{m}_recall@{k}", f"{m}_mrr", f"{m}_mean_hit_rank", f"{m}_n"]
    rows = []
    json_rows = []
    for w in weeks:
        row = [w]
        rec = {"week": w}
        for m in methods:
            d = data[m].get(w, {})
            r5 = d.get("recall_at_k"); mr = d.get("mrr"); hr = d.get("mean_hit_rank")
            row += [round(r5, 1) if r5 is not None else "",
                    round(mr, 1) if mr is not None else "",
                    round(hr, 2) if hr is not None else "",
                    d.get("n", 0)]
            rec[m] = d
        rows.append(row)
        json_rows.append(rec)
    base = f"{C.OUT_DIR}/a9_recall_rank_drift_{short}_{fbmode}"
    C.write_csv(base + ".csv", header, rows)
    C.write_json(base + ".json", {"model": model, "fbmode": fbmode, "k": k, "rows": json_rows})
    _maybe_plot(base + ".png", weeks, data, methods, short, fbmode, k)

    print(f"[A9] retrieval drift  model={short} fbmode={fbmode}  k={k}")
    print("week | " + " | ".join(f"{m}_R@{k}/MRR/rank" for m in methods))
    for w in weeks:
        cells = []
        for m in methods:
            d = data[m].get(w, {})
            r5 = d.get("recall_at_k"); mr = d.get("mrr"); hr = d.get("mean_hit_rank")
            cells.append(f"{(r5 or 0):4.0f}/{(mr or 0):4.0f}/{(hr or 0):4.2f}")
        print(f"  {w:2d} | " + " | ".join(cells))
    print(f"  -> {base}.csv / .json / .png")


def _maybe_plot(png, weeks, data, methods, short, fbmode, k):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[A9] matplotlib unavailable, skip figure ({e})")
        return
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "lines.linewidth": 2.2, "lines.markersize": 7,
    })
    label = {"id": "Ideal", "m0": "Mem0", "adamem": "AdaMem"}
    colors = {"id": "#2ca02c", "m0": "#d62728", "adamem": "#1f77b4"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for m in methods:
        ax1.plot(weeks, [(data[m].get(w, {}).get("recall_at_k")) for w in weeks],
                 marker="o", label=label.get(m, m.upper()), color=colors.get(m))
        ax2.plot(weeks, [(data[m].get(w, {}).get("mrr")) for w in weeks],
                 marker="s", label=label.get(m, m.upper()), color=colors.get(m))
    ax1.set_title(f"Recall@{k} over Weeks")
    ax1.set_xlabel("Week"); ax1.set_ylabel(f"Recall@{k} (%)")
    ax1.set_ylim(0, 100); ax1.grid(alpha=0.3); ax1.legend()
    ax2.set_title("MRR over Weeks")
    ax2.set_xlabel("Week"); ax2.set_ylabel("MRR (%)")
    ax2.set_ylim(0, 100); ax2.grid(alpha=0.3); ax2.legend()
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--fbmode", default="verbose")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    run(a.model, a.fbmode, a.k)
