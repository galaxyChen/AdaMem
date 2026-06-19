#!/usr/bin/env python3
"""Paper-experiment matrix runner.

Cartesian-expand 4 methods × 4 memory models × 2 fbmodes × N stories and
dispatch each cell to the corresponding ``run_<method>.py`` as an isolated
subprocess. ID and FC are run **independently in every (memory_model,
fbmode) cell** -- there is no result reuse, even though their internal
behaviour is invariant to those dimensions.

Cells whose three lossless artifacts (``qa_records.json``,
``extracted_memories.jsonl``, ``recall.jsonl``) already exist are skipped
(resumable). The runner does **not** mutate dataset files; it only reads
``data/scaling/`` and writes under ``exp/paper/``.

Sanity-check at startup:
  - probes ``OPENAI_BASE_URL`` (chat endpoint) by sending a 1-token chat call;
  - probes the embedding endpoint by sending a 1-string embedding request.

Both run via :func:`AdaMem.common.sanity_check_connectivity` and hard-fail
if either is misconfigured.

Usage:
    python run_paper_matrix.py                                # full 4×4×2 matrix
    python run_paper_matrix.py --methods adamem,m0 \\
        --memory-models deepseek-v4-flash                            # subset
    python run_paper_matrix.py --stories 1 --weeks 2 \\
        --parallel-experiment 2                                      # smoke
"""

import argparse
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    log,
    NUM_STORIES_DEFAULT, NUM_WEEKS_DEFAULT,
    MEMORY_MODEL_CHOICES, set_memory_model,
    paper_story_dir, is_paper_cell_complete,
    print_effective_config, sanity_check_connectivity,
    PAPER_EXP_ROOT,
)


METHODS_ALL = ("id", "fc", "m0", "adamem")
FBMODES_ALL = ("with_gold", "verbose")
DEFAULT_MEMORY_MODELS = tuple(MEMORY_MODEL_CHOICES)  # vendor aliases

ADAMEM_DIR = os.path.dirname(os.path.abspath(__file__))
METHOD_TO_SCRIPT = {
    "id":     os.path.join(ADAMEM_DIR, "run_id.py"),
    "fc":     os.path.join(ADAMEM_DIR, "run_fc.py"),
    "m0":     os.path.join(ADAMEM_DIR, "run_m0.py"),
    "adamem": os.path.join(ADAMEM_DIR, "run_adamem.py"),
}


def _parse_csv(s, allowed=None, name="value"):
    items = [x.strip() for x in (s or "").split(",") if x.strip()]
    if allowed is not None:
        bad = [x for x in items if x not in allowed]
        if bad:
            raise SystemExit(
                f"--{name}: unknown items {bad}; expected subset of {sorted(allowed)}"
            )
    return items


def _cell_dir(method, memory_model, fbmode, story_id, out_root):
    return paper_story_dir(
        method, fbmode, story_id,
        memory_model_alias=memory_model, out_root=out_root,
    )


def _run_one_cell(*, method, memory_model, fbmode, story_id, num_weeks,
                  out_root, exp_tag, log_dir, parallel_story=1):
    """Spawn one ``run_<method>.py`` subprocess for one (model, fbmode, story).

    Returns a dict with keys ``ok``, ``rc``, ``elapsed_s``, ``log_path``.
    """
    cell_dir = _cell_dir(method, memory_model, fbmode, story_id, out_root)
    if is_paper_cell_complete(cell_dir):
        return {"ok": True, "rc": 0, "elapsed_s": 0.0,
                "skipped": True, "log_path": None}

    script = METHOD_TO_SCRIPT[method]
    cmd = [
        sys.executable, "-u", script,
        "--story", str(story_id),
        "--weeks", str(num_weeks),
        "--fbmode", fbmode,
        "--memory-model", memory_model,
        "--exp-tag", exp_tag,
        "--no-parallel",
        # Each cell is its own subprocess; the per-method --parallel only
        # matters when --story=all, which we don't use here.
        "--parallel", "1",
        # The matrix runner does its own up-front health check; child
        # processes can skip it to save 2 LLM calls per cell.
        "--skip-sanity-check",
    ]
    if out_root:
        cmd.extend(["--out-dir", out_root])
    log_path = os.path.join(
        log_dir,
        f"{method}__{memory_model}__{fbmode}__story_{story_id}.log",
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = os.environ.copy()
    env.setdefault("ADAMEM_MEMORY_MODEL", memory_model)
    if exp_tag:
        env["ADAMEM_EXP_TAG"] = exp_tag

    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            rc = subprocess.run(
                cmd, env=env, cwd=ADAMEM_DIR,
                stdout=f, stderr=subprocess.STDOUT,
            ).returncode
    except Exception as e:
        elapsed = time.time() - t0
        return {"ok": False, "rc": -1, "elapsed_s": elapsed,
                "skipped": False, "log_path": log_path,
                "error": f"{type(e).__name__}: {e}"}
    elapsed = time.time() - t0
    return {"ok": rc == 0, "rc": rc, "elapsed_s": elapsed,
            "skipped": False, "log_path": log_path}


def _validate_dataset_schema(*, stories, weeks):
    for sid in stories:
        cmd = [
            sys.executable, "-u", os.path.join(ADAMEM_DIR, "prepare_data.py"),
            "--validate",
            "--story", str(sid),
            "--weeks", str(weeks),
        ]
        rc = subprocess.run(cmd, cwd=ADAMEM_DIR).returncode
        if rc != 0:
            raise RuntimeError(
                f"dataset schema validation failed for story_{sid}; run `python prepare_data.py "
                "--backfill-golden --validate` or regenerate the dataset before matrix run"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Paper-experiment 4×4×2 matrix runner (4 methods × 4 memory models × 2 fbmodes)",
    )
    parser.add_argument("--methods", default=",".join(METHODS_ALL),
                        help=f"Comma-separated subset of {sorted(METHODS_ALL)}")
    parser.add_argument("--memory-models", default=",".join(DEFAULT_MEMORY_MODELS),
                        help="Comma-separated vendor aliases. Defaults to all 4 candidates.")
    parser.add_argument("--fbmodes", default=",".join(FBMODES_ALL),
                        help=f"Comma-separated subset of {sorted(FBMODES_ALL)}")
    parser.add_argument("--stories", type=int, default=NUM_STORIES_DEFAULT)
    parser.add_argument("--story", default=None,
                        help="If set, only run this single story id (overrides --stories).")
    parser.add_argument("--weeks", type=int, default=NUM_WEEKS_DEFAULT)
    parser.add_argument("--exp-tag", default=os.environ.get("ADAMEM_EXP_TAG", "paper"))
    parser.add_argument("--out-dir", default=PAPER_EXP_ROOT,
                        help="Root output dir (default: AdaMem/exp/paper)")
    parser.add_argument("--parallel-experiment", type=int, default=1,
                        help="How many (method, memory_model, fbmode) cells to run in parallel. "
                             "Each cell internally runs stories sequentially.")
    parser.add_argument("--parallel-story", type=int, default=1,
                        help="How many stories within a single cell to run in parallel. "
                             "Currently honoured by spawning N concurrent subprocesses for the "
                             "same (method, memory_model, fbmode) tuple but distinct story ids.")
    parser.add_argument("--skip-sanity-check", action="store_true",
                        help="Skip the proxy/embedding probe at startup.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print the matrix plan, do not spawn subprocesses.")
    args = parser.parse_args()

    methods = _parse_csv(args.methods, allowed=set(METHODS_ALL), name="methods")
    memory_models = _parse_csv(args.memory_models, allowed=set(MEMORY_MODEL_CHOICES),
                               name="memory-models")
    fbmodes = _parse_csv(args.fbmodes, allowed=set(FBMODES_ALL), name="fbmodes")
    if args.story is not None:
        stories = [int(args.story)]
    else:
        stories = list(range(1, args.stories + 1))

    print_effective_config({
        "methods": ",".join(methods),
        "memory_models": ",".join(memory_models),
        "fbmodes": ",".join(fbmodes),
        "stories": ",".join(str(s) for s in stories),
        "weeks": args.weeks,
        "out_root": args.out_dir,
        "parallel_experiment": args.parallel_experiment,
        "parallel_story": args.parallel_story,
    })

    if not args.skip_sanity_check and not args.dry_run:
        try:
            sanity_check_connectivity()
        except Exception as e:
            print(f"sanity_check FAILED: {e}", file=sys.stderr)
            sys.exit(5)
        try:
            _validate_dataset_schema(stories=stories, weeks=args.weeks)
        except Exception as e:
            print(f"dataset validation FAILED: {e}", file=sys.stderr)
            sys.exit(6)

    # --- Plan ---
    plan = []   # list of (method, memory_model, fbmode, story_id)
    for method in methods:
        for mm in memory_models:
            for fb in fbmodes:
                for sid in stories:
                    plan.append((method, mm, fb, sid))

    total_cells = len(plan)
    skipped = 0
    todo = []
    for cell in plan:
        method, mm, fb, sid = cell
        out_dir = _cell_dir(method, mm, fb, sid, args.out_dir)
        if is_paper_cell_complete(out_dir):
            skipped += 1
        else:
            todo.append(cell)
    print(f"matrix: total={total_cells} cached={skipped} todo={len(todo)}")

    if args.dry_run:
        for cell in plan:
            method, mm, fb, sid = cell
            out_dir = _cell_dir(method, mm, fb, sid, args.out_dir)
            mark = "SKIP" if is_paper_cell_complete(out_dir) else "RUN "
            print(f"  [{mark}] {method:<6} {mm:<48} {fb:<10} story_{sid}")
        return

    if not todo:
        print("Nothing to do.")
        return

    # --- Schedule ---
    log_dir = os.path.join(args.out_dir, "_run_logs")
    os.makedirs(log_dir, exist_ok=True)
    # Group cells by (method, mm, fb) -- "experiment" -- so that
    # ``--parallel-story`` controls intra-experiment concurrency and
    # ``--parallel-experiment`` controls how many experiments run at once.
    by_exp = {}
    for method, mm, fb, sid in todo:
        by_exp.setdefault((method, mm, fb), []).append(sid)
    exp_keys = list(by_exp.keys())
    print(f"experiments={len(exp_keys)} cells={len(todo)} "
          f"parallel_experiment={args.parallel_experiment} parallel_story={args.parallel_story}")

    overall_t0 = time.time()
    results = []  # list of (cell_tuple, result_dict)
    failures = []

    def _run_experiment(exp_key):
        method, mm, fb = exp_key
        sids = by_exp[exp_key]
        local_results = []
        if args.parallel_story <= 1:
            for sid in sids:
                r = _run_one_cell(
                    method=method, memory_model=mm, fbmode=fb, story_id=sid,
                    num_weeks=args.weeks, out_root=args.out_dir,
                    exp_tag=args.exp_tag, log_dir=log_dir,
                )
                local_results.append(((method, mm, fb, sid), r))
                _print_progress(method, mm, fb, sid, r)
        else:
            with ThreadPoolExecutor(max_workers=args.parallel_story) as ex:
                fut_map = {
                    ex.submit(_run_one_cell, method=method, memory_model=mm,
                              fbmode=fb, story_id=sid, num_weeks=args.weeks,
                              out_root=args.out_dir, exp_tag=args.exp_tag,
                              log_dir=log_dir): sid
                    for sid in sids
                }
                for fut in as_completed(fut_map):
                    sid = fut_map[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        r = {"ok": False, "rc": -1, "elapsed_s": 0.0,
                             "skipped": False, "log_path": None,
                             "error": f"{type(e).__name__}: {e}"}
                    local_results.append(((method, mm, fb, sid), r))
                    _print_progress(method, mm, fb, sid, r)
        return local_results

    if args.parallel_experiment <= 1:
        for k in exp_keys:
            results.extend(_run_experiment(k))
    else:
        with ThreadPoolExecutor(max_workers=args.parallel_experiment) as ex:
            futs = {ex.submit(_run_experiment, k): k for k in exp_keys}
            for fut in as_completed(futs):
                try:
                    results.extend(fut.result())
                except Exception as e:
                    k = futs[fut]
                    print(f"experiment {k} crashed: {e}", file=sys.stderr)
                    traceback.print_exc()

    failures = [c for c, r in results if not r.get("ok")]
    elapsed_min = (time.time() - overall_t0) / 60.0
    ok_count = sum(1 for _, r in results if r.get("ok") and not r.get("skipped"))
    skip_count = sum(1 for _, r in results if r.get("skipped"))
    print(f"\n=== matrix done in {elapsed_min:.1f} min ===")
    print(f"  cells run OK: {ok_count}")
    print(f"  cells skipped (resume): {skip_count}")
    print(f"  cells failed: {len(failures)}")
    if failures:
        print("  ----- failures -----")
        for cell in failures:
            method, mm, fb, sid = cell
            print(f"    [FAIL] {method} {mm} {fb} story_{sid}")
        sys.exit(2)


def _print_progress(method, mm, fb, sid, r):
    if r.get("skipped"):
        return
    status = "OK" if r.get("ok") else f"FAIL rc={r.get('rc')}"
    elapsed = r.get("elapsed_s", 0.0)
    log_path = r.get("log_path") or "-"
    print(f"  [{status:<8}] {method:<6} {mm:<48} {fb:<10} story_{sid}  "
          f"({elapsed:.1f}s, log={log_path})", flush=True)


if __name__ == "__main__":
    main()
