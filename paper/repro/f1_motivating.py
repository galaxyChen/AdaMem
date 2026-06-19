#!/usr/bin/env python3
"""F1 -- Motivating experiment.

Per-week QA accuracy for ID / M0 / AdaMem plus cumulative extracted-memory
volume for M0 / AdaMem, micro-averaged over all available stories for a given
(model, fbmode). Default cell: deepseek / verbose.

Outputs:
  out/f1_motivating_<model>_<fb>.csv
  out/f1_motivating_<model>_<fb>.json
  out/f1_motivating_<model>_<fb>.png
"""

import argparse
import collections

import _common as C


def per_week_accuracy(method, model, fbmode):
    by_week = collections.defaultdict(lambda: [0, 0])  # week -> [correct, total]
    for sid in C.list_stories(method, model, fbmode):
        recs = C.load_qa_records(method, model, fbmode, sid) or []
        for r in recs:
            w = r.get("week")
            by_week[w][1] += 1
            by_week[w][0] += 1 if r.get("correct") else 0
    return {w: (cor, tot) for w, (cor, tot) in by_week.items()}


def per_week_memory_volume(method, model, fbmode):
    """Daily-dialogue add volume per week (excludes qa_writeback / golden)."""
    by_week = collections.defaultdict(int)
    for sid in C.list_stories(method, model, fbmode):
        path = f"{C.cell_dir(method, model, fbmode)}/story_{sid}/extracted_memories.jsonl"
        for row in C.iter_jsonl(path):
            if row.get("source") == "dialogue":
                by_week[row.get("week")] += len(row.get("added") or [])
    return dict(by_week)


def run(model="deepseek-v4-flash", fbmode="verbose", smooth_window=3):
    C.ensure_out()
    short = C.short_model(model)
    methods = [m for m in ("id", "m0", "adamem") if C.list_stories(m, model, fbmode)]
    acc = {m: per_week_accuracy(m, model, fbmode) for m in methods}
    vol = {m: per_week_memory_volume(m, model, fbmode) for m in ("m0", "adamem")
           if C.list_stories(m, model, fbmode)}

    weeks = sorted({w for m in acc for w in acc[m]})

    # ---- CSV ----
    header = ["week"]
    for m in methods:
        header += [f"{m}_acc", f"{m}_n"]
    for m in vol:
        header += [f"{m}_mem_week", f"{m}_mem_cum"]
    rows = []
    cum = {m: 0 for m in vol}
    json_rows = []
    for w in weeks:
        row = [w]
        rec = {"week": w}
        for m in methods:
            cor, tot = acc[m].get(w, (0, 0))
            a = C.pct(cor, tot)
            row += [round(a, 1) if a is not None else "", tot]
            rec[f"{m}_acc"] = a
            rec[f"{m}_n"] = tot
        for m in vol:
            cum[m] += vol[m].get(w, 0)
            row += [vol[m].get(w, 0), cum[m]]
            rec[f"{m}_mem_week"] = vol[m].get(w, 0)
            rec[f"{m}_mem_cum"] = cum[m]
        rows.append(row)
        json_rows.append(rec)

    base = f"{C.OUT_DIR}/f1_motivating_{short}_{fbmode}"
    C.write_csv(base + ".csv", header, rows)
    C.write_json(base + ".json", {"model": model, "fbmode": fbmode,
                                  "stories": C.list_stories("m0", model, fbmode),
                                  "rows": json_rows})

    _maybe_plot(base + ".png", weeks, acc, vol, methods, short, fbmode, smooth_window)

    # ---- stdout ----
    print(f"[F1] motivating  model={short} fbmode={fbmode}  "
          f"stories={C.list_stories('m0', model, fbmode)}")
    print("week | " + " | ".join(f"{m:>6}" for m in methods) + " || "
          + " | ".join(f"{m}_cum" for m in vol))
    cum = {m: 0 for m in vol}
    for w in weeks:
        accs = []
        for m in methods:
            cor, tot = acc[m].get(w, (0, 0))
            a = C.pct(cor, tot)
            accs.append(f"{a:6.1f}" if a is not None else "   -- ")
        for m in vol:
            cum[m] += vol[m].get(w, 0)
        print(f"  {w:2d} | " + " | ".join(accs) + " || "
              + " | ".join(f"{cum[m]:5d}" for m in vol))
    print(f"  -> {base}.csv / .json / .png")


def _smooth(ys, window=3):
    """Centered moving average; shrinks at edges; ignores None."""
    n = len(ys)
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vals = [v for v in ys[lo:hi] if v is not None]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def _maybe_plot(png, weeks, acc, vol, methods, short, fbmode, smooth_window=3):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[F1] matplotlib unavailable, skip figure ({e})")
        return
    fig, ax1 = plt.subplots(figsize=(6, 4))
    colors = {"id": "#2ca02c", "m0": "#d62728", "adamem": "#1f77b4"}
    labels = {"id": "Ideal Memory", "m0": "Mem0", "adamem": "AdaMem"}
    for m in methods:
        raw = [C.pct(*acc[m].get(w, (0, 0))) for w in weeks]
        sm = _smooth(raw, smooth_window)
        # raw as faint markers (honesty), smoothed as the bold trend line
        ax1.plot(weeks, raw, marker="o", ms=3, linestyle="none",
                 color=colors.get(m), alpha=0.30)
        ax1.plot(weeks, sm, linewidth=2.2, label=labels.get(m, m), color=colors.get(m))
    ax1.set_xlabel("week"); ax1.set_ylabel("QA accuracy (%)")
    ax1.set_title(f"QA Accuracy on AdaMem Benchmark ({short}/explicit)")
    ax1.set_ylim(50, 100); ax1.grid(alpha=0.3); ax1.legend()
    fig.tight_layout(); fig.savefig(png, dpi=140)
    
    # Also save/copy to paper/draft if it is the verbose cell to update the main paper's figure
    if "verbose" in png:
        import shutil
        draft_path = png.replace("paper/repro/out", "paper/draft")
        try:
            shutil.copy(png, draft_path)
            print(f"Copied {png} to {draft_path}")
        except Exception as e:
            print(f"Failed to copy to {draft_path}: {e}")
            
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--fbmode", default="verbose")
    ap.add_argument("--smooth", type=int, default=3, help="centered moving-average window")
    a = ap.parse_args()
    run(a.model, a.fbmode, a.smooth)
