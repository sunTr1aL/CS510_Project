# Repository Manifest

This manifest records the publish selection made for the GitHub repository.

## Selected For Upload

| Path | Reason |
| --- | --- |
| `README.md`, `REPOSITORY_MANIFEST.md`, `.gitignore` | Repository orientation and reproducibility boundary. |
| `requirements-experiment.txt` | Python dependencies for the experiment scaffold. |
| `experiments/` | Core implementation: run registry, CLI, audit helpers, CLIP backend, and RRB modules. |
| `scripts/` | Reproduction, remote launch, dataset staging, guardrail evaluation, and summary scripts. |
| `tests/` | Local scaffold verification. |
| `refine-logs/` | Canonical proposal, tracker, consolidated result report, and diagnostics. |
| `outputs/remote/rrb_topvenue_20260430_163958/analysis/` | Compact R041-R045 follow-up result summary used by the final paper assets. |
| `outputs/runs/R001/`, `outputs/dry/R011/` | Small example manifests and audit summary for scaffold provenance. |
| `figures/` | Asset-generation code and generated paper tables/figures. |
| `paper/` source files and `paper/main.pdf` | Paper source, bibliography, final PDF, and writing logs. |
| `research-wiki/` | Claim, gap, and related-work context used during project refinement. |
| `AUTO_REVIEW.md`, `PAPER_PLAN.md`, `NARRATIVE_REPORT.md` | Higher-level review, paper planning, and final narrative report artifacts. |

## Excluded From Upload

| Path or Pattern | Reason |
| --- | --- |
| `literature/` | Raw downloaded PDFs are large and may not be redistributable. |
| `data/`, `datasets/`, `benchmarks/`, `checkpoints/` | External datasets/checkpoints should be staged separately. |
| `outputs/` outside the selected compact summaries | Full training/evaluation outputs can be large and are not required for the paper tables. |
| `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors` | Model checkpoints are large generated artifacts. |
| `__pycache__/`, `.pytest_cache/`, `.DS_Store` | Local/generated cache files. |
| `paper/texmf/` | Local TeX package cache; use a normal LaTeX installation instead. |
| `paper/main_round*.pdf`, `paper/*.aux`, `paper/*.log`, `paper/*.bbl`, `paper/*.blg`, `paper/*.out` | Intermediate paper build and review-loop artifacts. |
| `REVIEW_STATE.json`, `paper/PAPER_IMPROVEMENT_STATE.json` | Local process state, not reproducibility material. |
