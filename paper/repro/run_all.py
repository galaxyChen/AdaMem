#!/usr/bin/env python3
"""Run all zero-cost paper analyses and dump artifacts to paper/repro/out/.

Zero-cost = reads only saved artifacts under exp/paper/ and data/scaling/;
no LLM calls, no re-judging.

  T1  main table: Accuracy + Extraction-F1 + MER       (all cells)
  F1  motivating per-week accuracy + memory growth     (deepseek/verbose)
  A7  accuracy by qa_type                            (all cells)
  A8  accuracy by info_category                      (all cells)
  A9  retrieval rank drift over weeks                (judged cells)
  A10 token cost / benefit                           (all cells)
"""

import main_table
import f1_motivating
import a7_by_qa_type
import a8_by_info_category
import a9_recall_rank_drift
import a10_cost


def main():
    print("=" * 70)
    main_table.run()
    print("=" * 70)
    f1_motivating.run("deepseek-v4-flash", "verbose")
    print("=" * 70)
    a7_by_qa_type.run()
    print("=" * 70)
    a8_by_info_category.run()
    print("=" * 70)
    a9_recall_rank_drift.run("deepseek-v4-flash", "verbose")
    print("=" * 70)
    a10_cost.run()
    print("=" * 70)
    print("done -> paper/repro/out/")


if __name__ == "__main__":
    main()
