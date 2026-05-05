# Anchored Relational Residual Adaptation for CLIP

This repository contains the reproducibility package for a CS510 project on lightweight relational residual adaptation for frozen CLIP-style dual encoders.

The paper-facing method is Anchored Relational Residual Bottleneck (RRB): a bounded residual adapter applied to frozen OpenCLIP embeddings, selected for relation gains while explicitly measuring retrieval and zero-shot guardrail damage.

## Included Contents

- `experiments/`: run registry, CLI launcher, OpenCLIP backend, audit helpers, and RRB modules.
- `scripts/`: dataset staging, remote launch scripts, guardrail evaluation, diagnostics, and result summarization scripts.
- `tests/`: scaffold tests that check the experiment registry, dry-run manifests, helper logic, and diagnostics paths.
- `refine-logs/`: canonical proposal, experiment plan/tracker, consolidated result report, and diagnostics summaries.
- `outputs/`: compact selected result summaries only, including the R041-R045 top-venue follow-up analysis and small manifest examples.
- `figures/`: script and generated TikZ/table assets derived from the consolidated result files.
- `paper/`: LaTeX paper source, bibliography, generated paper figures/tables, writing logs, and final `main.pdf`.
- `research-wiki/`: structured notes used to preserve claims, gaps, and related-work context.

## Excluded Contents

The git package intentionally excludes raw literature PDFs, downloaded benchmarks, checkpoints, full remote output directories, Python caches, LaTeX intermediates, local TeX package caches, and private review-loop state. These files are either large, generated, not redistributable, or unnecessary for reproducing the reported tables and paper source.

Remote filesystem paths in logs and scripts are provenance records from the original runs. They are not required unless reproducing the full GPU training workflow on the same server layout.

## Quick Verification

Install the experiment dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-experiment.txt
```

Run the scaffold tests:

```bash
python -m unittest tests.test_experiment_scaffold
```

Regenerate paper assets from the canonical result summaries:

```bash
python figures/generate_paper_assets.py
```

Build the paper from source with a standard LaTeX toolchain from `paper/`:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The checked-in final paper is `paper/main.pdf`.
