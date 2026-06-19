# AdaMem: Learning What to Remember for Personalized Long-Horizon LLM Agents

This repository contains the official code, benchmark, and reproduction
artifacts for the paper:

> **AdaMem: Learning What to Remember for Personalized Long-Horizon LLM Agents**
> Xingyu Chen, Rui Wang, Zhaopeng Tu, Liefeng Bo.

AdaMem reframes long-term memory for LLM agents from *remembering everything* to
**learning what to remember**. Instead of extracting memories uniformly, AdaMem
maintains a structured, role-specific **Memory Policy** and refines it from weekly
QA feedback through a lightweight, patch-style self-reflection step with failure
rollback. To study this setting we release **AdaMem-Bench**, a benchmark that
simulates weeks of personalized interaction with week-by-week QA.

With this repository you can:

1. **Inspect the benchmark** — the generated AdaMem-Bench stories used in the paper.
2. **Read and run the method** — the full evaluation pipeline (AdaMem and baselines).
3. **Reproduce every figure and table** — from the saved experiment artifacts, with no API key required.

No provider URLs, API keys, or internal model-routing identifiers are bundled.
Chat/embedding endpoints are read from environment variables, and model
identifiers use the names reported in the paper.

---

## 1. Installation

```bash
git clone https://github.com/galaxyChen/AdaMem.git
cd AdaMem
pip install -r requirements.txt
```

`requirements.txt` lists the Python dependencies. Reproducing the paper figures
only needs `matplotlib`; running the full pipeline additionally needs the memory
and LLM-client dependencies listed there.

---

## 2. Repository layout

```
.
├── README.md                 # this file
├── LICENSE                   # MIT license
├── requirements.txt          # Python dependencies
│
├── data/scaling/             # (1) BENCHMARK (AdaMem-Bench)
│   └── story_{2,3,4,5,10}/   #     character_profiles, outline, week{1..10}, test_qa
│                             #     (the 5 stories used in the reported experiments)
│
├── common.py                 # (2) EVALUATION CODE — shared infrastructure
├── call_llm.py               #     OpenAI-compatible client used by LLM judges
├── prepare_data.py           #     data-synthesis pipeline (produced data/scaling/)
├── DATA_GENERATION.md        #     data-synthesis documentation
├── run_id.py / run_fc.py     #     method runners: Ideal Memory / Full Context
├── run_m0.py / run_adamem.py #     method runners: Mem0 baseline / AdaMem
├── run_paper_matrix.py       #     matrix driver (method × model × feedback × story)
├── analysis/
│   └── run_paper_judge.py    #     LLM-as-judge: extraction / recall judging
│
├── exp/paper/                # saved experiment artifacts (inputs for reproduction)
│   └── <method>/<model>/<feedback>/story_<id>/
│
└── paper/
    ├── repro/                # (3) FIGURE/TABLE REPRODUCTION CODE
    │   ├── run_all.py        #     regenerate all zero-cost analyses
    │   ├── main_table.py     #     Table 2 (Acc / F1 / MER / Vol + marginals)
    │   ├── f1_motivating.py  #     Figure 1 (per-week accuracy + memory growth)
    │   ├── a8_by_info_category.py  # Table 4 (accuracy by information category)
    │   ├── a9_recall_rank_drift.py # Figure 2a (Recall@5 / MRR over weeks)
    │   ├── a4_policy_alignment.py  # Figure 2b (policy convergence; needs judge)
    │   ├── a3_policy_following.py  # RQ4 policy-following rate (needs judge)
    │   ├── a7_by_qa_type.py / a10_cost.py
    │   ├── judge_memory.py / _judge_common.py  # (re)judging utilities (need endpoint)
    │   ├── _common.py        #     shared paths/loaders
    │   └── out/              #     PRECOMPUTED outputs (CSV / JSON / PNG)
    └── draft/                # final figures as used in the paper
```

---

## 3. Model and setting names

| In the paper                | In this code / artifacts |
|-----------------------------|--------------------------|
| DeepSeek-V4-Flash (extractor / QA / judge) | `deepseek-v4-flash` |
| Gemini-3.5-Flash (extractor) | `gemini-3.5-flash` |
| Explicit feedback           | `verbose`   |
| Implicit feedback           | `with_gold` |
| Full Context / Ideal Memory / Mem0 / AdaMem | `fc` / `id` / `m0` / `adamem` |

QA answers and all judging use `deepseek-v4-flash`; the two extraction models
(`deepseek-v4-flash`, `gemini-3.5-flash`) are the only experiment variable.

---

## 4. Reproduce the paper figures and tables (no API key required)

The analyses under `paper/repro/` are **zero-cost**: they read only the saved
artifacts in `exp/paper/` and the benchmark in `data/scaling/`, with no LLM
calls. Only `matplotlib` is required.

```bash
pip install matplotlib
cd paper/repro
python3 run_all.py          # → writes CSV / JSON / PNG into paper/repro/out/
```

This regenerates:

- **Table 2** (main results) — `main_table.py` → `out/main_table*.csv|json`
- **Figure 1** (motivating) — `f1_motivating.py` → `out/f1_motivating_deepseek_verbose.png`
- **Table 4** (by information category) — `a8_by_info_category.py`
- **Figure 2a** (Recall@5 / MRR drift) — `a9_recall_rank_drift.py`
- Cost/benefit (RQ4) — `a10_cost.py`

Individual scripts accept `--model {deepseek-v4-flash,gemini-3.5-flash}` and
`--fbmode {verbose,with_gold}`.

### Analyses that recompute LLM judgments (optional)

`a3_policy_following.py`, `a4_policy_alignment.py` (Figure 2b), and
`judge_memory.py` call the `deepseek-v4-flash` judge and therefore require a
configured chat endpoint (see §5). Their precomputed outputs are already in
`paper/repro/out/` (e.g. `a4_policy_convergence.png`), so re-running them is not
needed to reproduce the figures.

---

## 5. Run the full evaluation pipeline (requires your own endpoints)

The method runners and data-synthesis pipeline talk to an OpenAI-compatible
chat endpoint and an OpenAI-compatible embedding endpoint. No endpoint or
credential is bundled; configure them via environment variables:

```bash
export OPENAI_BASE_URL="https://<your-openai-compatible-host>/v1"
export OPENAI_API_KEY="<your-key>"
export EMBEDDING_BASE_URL="https://<your-embedding-host>/v1"
export EMBEDDING_API_KEY="<your-key>"
# optional: EMBEDDING_MODEL (default text-embedding-v4), EMBEDDING_DIMS (default 1024)
```

Then, for example:

```bash
# one cell
python3 run_adamem.py --story 1 --weeks 10 --fbmode verbose --memory-model deepseek-v4-flash
# full matrix (methods × models × feedback × stories), resumable
python3 run_paper_matrix.py
# (re)produce extraction/recall judgments used by the analyses
python3 analysis/run_paper_judge.py --methods adamem --memory-models deepseek-v4-flash
```

Artifacts are written under `exp/paper/<method>/<model>/<feedback>/story_<id>/`.

See `DATA_GENERATION.md` for how the AdaMem-Bench stories under `data/scaling/`
were synthesized with `prepare_data.py`.

---

## 6. Notes on the bundled artifacts

- `exp/paper/` contains only the files consumed by the reproduction scripts:
  `qa_records.json`, `extracted_memories.jsonl`, `extraction_judged.jsonl`,
  `recall_judged.jsonl`, `policy_snapshots.json`, and the `*_done.json` markers.
  The large raw retrieval inputs (`recall.jsonl`) and the per-question full
  prompt transcripts (`qa_records[*].test_messages`, `qa_turns`) were dropped to
  keep the bundle compact; none of them are used to compute the paper numbers.
- The bundle covers exactly the **5 stories (2, 3, 4, 5, 10)** used in the
  reported experiments — both `data/scaling/` and `exp/paper/` are restricted to
  these, matching the paper's evaluation scale. Numbers are micro-averaged over
  stories.

---

## 7. Citation

If you find AdaMem or AdaMem-Bench useful, please cite:

```bibtex
@article{chen2026adamem,
  title   = {AdaMem: Learning What to Remember for Personalized Long-Horizon LLM Agents},
  author  = {Chen, Xingyu and Wang, Rui and Tu, Zhaopeng and Bo, Liefeng},
  year    = {2026}
}
```

---

## 8. License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
