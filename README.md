# Anchored Relational Residual Adaptation for CLIP

This repository contains the reproducibility package for a CS510 project on lightweight relational residual adaptation for frozen CLIP-style dual encoders.

The paper-facing method is Anchored Relational Residual Bottleneck (RRB): a bounded residual adapter applied to frozen OpenCLIP embeddings, selected for relation gains while explicitly measuring retrieval and zero-shot guardrail damage.

## Included Contents

- `experiments/`: run registry, CLI launcher, OpenCLIP backend, audit helpers, and RRB modules.
- `scripts/`: dataset staging, remote launch scripts, guardrail evaluation, diagnostics, and result summarization scripts.

## Excluded Contents

The git package intentionally excludes raw literature PDFs, downloaded benchmarks, checkpoints, full remote output directories, Python caches, LaTeX intermediates, local TeX package caches, and private review-loop state. These files are either large, generated, not redistributable, or unnecessary for reproducing the reported tables and paper source.

## Submitted Project Report
The checked-in final paper is `Report.pdf`. The complete history of idea refinement, experiment design, and result anaylsis can be found in `NARRATIVE_REPORT.md`.
