# Numerical Intelligence Transfer

Can numerical reasoning shown by a small LLM (Qwen2.5-1.5B/0.5B-Instruct) be transferred into a lightweight
sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`) without destroying its semantics? And are the
gains really *in the embedding*, or do they come from a numerical parser?

## Quick start

```bash
pip install -r requirements.txt
python main.py --smoke           # ~1 min, offline: tiny random encoder + MOCK teacher (pipeline check only)
python main.py --download-only   # fetch MiniLM, Qwen teachers, STS-B (run where internet is available)
python main.py                   # full study -> results/, models/, reports/
sbatch run_hpc.slurm             # same, on a SLURM GPU node (offline after --download-only)
```

The hardware is detected automatically. A GPU with ≥ 8 GB uses the Qwen2.5-1.5B teacher (bf16/fp16) with a budget of
4,000 pairs. A smaller GPU or a CPU falls back to Qwen2.5-0.5B, and on CPU the budget is 600 pairs.
Interrupted runs resume: finished checkpoints and teacher labels are reused (`--no-resume` to recompute).

## What runs

| Stage | Output |
|---|---|
| Dataset: ~5,000 statements, 7 categories, Python ground truth, quantity-grouped 70/15/15 split, held-out templates, number-words stress slice | `data/*.csv` |
| Teacher: Yes/No and A/B/Equal via next-token probabilities, ratio via generation, all verified against ground truth | `data/teacher_*.csv`, `results/tables/teacher_accuracy.json` |
| Exp1 original · Exp2 parser + original · parser-only reference | |
| Exp3 math-supervised fine-tuning · Exp3b same, on the teacher's budget (no-teacher ablation) · Exp4a raw teacher · Exp4b verified teacher · Exp5 Exp3 + parser · Exp6 learned numerical projection (no parser) · Exp3-abs | `models/numerical_transfer/*` |
| Ablation A–E (semantic → +equivalence → +ordering → +distance → +context) | `results/tables/ablation_seed42.csv` |
| Metrics (mean ± std over seeds 42, 43, 44; bootstrap CIs) | `results/metrics.json`, `results/tables/` |
| Figures 1–9 | `results/figures/` |
| Reports (auto-generated from the saved numbers; verdicts use a pre-declared rule) | `reports/experimental_report.md`, `reports/methodology.md` |

## Layout
```
config/config.yaml          all hyperparameters, seeds, budgets
main.py                     single reproducible entry point
src/numerical_parser.py     units, affine temperature, ground-truth relations, parser features
src/dataset_generator.py    statements, splits, pairs, triplets, stress slice, leakage audit
src/teacher_supervision.py  teacher prompts, soft labels, verification
src/numerical_distillation.py  losses (L_eq, L_ord, L_dist, L_ctx, L_sem), training, checkpoints
src/baseline_evaluation.py  Exp1/Exp2 + descriptive baseline cosine analysis
src/evaluation.py           all metrics (cosine-based and probe-based kept separate)
src/visualisation.py        figures
src/report_writer.py        report generation from results
run_hpc.slurm               SLURM template
```

## Key design decisions
* **No LLM output is ground truth.** Every determinable relation is computed in Python. Temperature uses the kelvin
  scale (affine conversions). Ratios of Celsius or Fahrenheit values are never formed.
* **Leakage.** All statements of the same physical quantity, in any unit, precision or context, share a split.
* **Equivalence ≠ context.** "37 °C (patient)" and "37 °C (machine)" are numerically equal but are *not* equivalence
  positives. A context loss keeps them apart.
* **Teacher vs direct supervision.** Exp3b uses exactly the teacher's pairs with ground-truth labels, so
  Exp4b − Exp3b isolates what distillation adds.
* **Parser vs learning.** The number-words slice cannot be read by the parser. Exp6 learns a numerical subspace without
  any parser.

## Extension experiments (other encoders, larger reasoners)

Each extension writes only to `runs/<name>/`; the main study in `results/`, `models/` and `paper/` is never touched.

| Name | What changes | Trained models (seeds 42-44) |
|---|---|---|
| `reasoner_3b` | LLM reasoner Qwen2.5-3B-Instruct | NumBridge-Matched, -Raw, -Verified (MiniLM) |
| `reasoner_7b` | LLM reasoner Qwen2.5-7B-Instruct | NumBridge-Matched, -Raw, -Verified (MiniLM) |
| `encoder_mpnet` | encoder all-mpnet-base-v2 (reuses the 1.5B labels) | Exact, Matched, Raw, Verified, Dual |
| `encoder_labse` | encoder LaBSE (reuses the 1.5B labels) | Exact, Matched, Raw, Verified, Dual |

```bash
python main.py --download-only --download-models \
  sentence-transformers/all-mpnet-base-v2,sentence-transformers/LaBSE,Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct
for e in reasoner_3b reasoner_7b encoder_mpnet encoder_labse; do
  sbatch --dependency=singleton run_extensions.slurm $e
done
```
`--download-only` fetches files without loading them, so it is safe on a login node.
