# Paper Writing Pipeline Report

Input: `NARRATIVE_REPORT.md`  
Venue: ACM Proceedings  
Date: 2026-05-05

## Pipeline Summary

| Phase | Status | Output |
|---|---|---|
| Paper plan | Complete | `../PAPER_PLAN.md` |
| Figures/tables | Complete | `../figures/` and `figures/` |
| LaTeX writing | Complete | `main.tex`, `sections/*.tex`, `references.bib` |
| Compilation | Complete | `main.pdf` |
| Improvement loop | Complete | 4 Gemini rounds, `main_round1.pdf`--`main_round4.pdf` |

## Final Build

- Final PDF: `main.pdf`
- Total pages: 10
- References begin: page 9
- Main body before references/appendix: 8 pages
- Undefined citations: 0
- Undefined references: 0
- Overfull boxes above 10pt: 0

## Improvement Scores

| Round | Score | Summary |
|---|---:|---|
| Round 1 | 8.5/10 | Strengthened guardrail disclaimers, terminology, Figure 1 marker, ROP/Flickr future-work framing. |
| Round 2 | 9.2/10 | Clarified relation metric aggregation, R022 context, and Figure 1 marker meaning. |
| Round 3 | 9.5/10 | No edits required; submission ready. |
| Round 4 | 10/10 | No edits required; final submission-ready verdict. |

## Deliverables

- `main.pdf` -- final compiled paper
- `main_round0_expanded.pdf` -- expanded pre-improvement baseline
- `main_round1.pdf` -- after round 1
- `main_round2.pdf` -- after round 2
- `main_round3.pdf` -- after round 3
- `main_round4.pdf` -- final round artifact
- `PAPER_IMPROVEMENT_LOG.md` -- review/fix log

## Remaining Caveats

- The manuscript intentionally frames RRB as a bounded relation-vs-preservation analysis, not a broad compositional reasoning solution.
- Standard PEFT baselines such as LoRA/text-only/vision-only tuning are listed as missing future-work comparisons.
- The local RACA-CLIP entry keeps conservative bibliographic metadata because the project copy does not include full public metadata.
