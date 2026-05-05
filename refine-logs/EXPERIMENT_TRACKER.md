# Experiment Tracker

**Last updated**: 2026-04-30  
**Canonical results**: `refine-logs/CONSOLIDATED_METHOD_RESULTS_2026-04-30.md`  
**Current status**: R041-R045 top-venue follow-ups were run on 2026-04-30. R044 was skipped because the R043 ROP pilot failed its relation gate.

## Completed Archive

| Runs | Milestone | Status | Takeaway |
|---|---|---|---|
| R001-R025 | M0-M4 | DONE | Archived diagnostic history: baselines, hard-token variants, first RRB sweep, and controls. See canonical consolidated results. |
| R026 | M5a | DONE | Consolidated completed evidence into the paper-facing results archive. |
| R027-R029 | M5b-M5c | DONE | R028 is the preferred simple checkpoint; R027/R029 expose seed/control tradeoffs. |
| R030 | M6 | DONE | Exported final selected-checkpoint diagnostics and failure taxonomy inputs. |
| R031-R032 | M7 | DONE | R028-style relation gains replicate, but guardrail preservation is seed-fragile. |
| R033 | M7 | DONE | Matched no-slot residual is relation-strong but guardrail-collapsing. |
| R034 | M7 | DONE | Winoground is flat-to-negative for R028; use as external-validity limitation. |
| R035-R039 | M8 | DONE | Anchor-20 lowers residual drift but still gives only 3/5 COCO pass rate. |
| R040 | M8 | DONE | Constrained no-slot residual remains unsafe versus slot RRB. |
| R041-R045 | M9-M11 | DONE / SKIPPED | Slot probes are diffuse, ROP pilot improves guardrails but loses too much relation accuracy, and Flickr30k confirms an external retrieval tax. |

## Forward Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R041 | M9 | Slot responsibility probe | R028, R035, R038 checkpoints | Existing structured counterfactual audit records and held-out relation examples | slot entropy, top-k slot concentration, edited-span overlap, attribution delta | REQUIRED_FOR_TOP_VENUE | DONE | All probed checkpoints were diffuse: entropy ~1.0000, top-1 concentration ~0.126, verdict `diffuse_or_unproven`. Avoid mechanistic slot-specialization claims. |
| R042 | M10a | ROP basis and preflight audit | PCA/protected-subspace audit over frozen OpenCLIP embeddings | COCO/anchor corpus | variance explained, residual projection energy, shape/dry-run checks | REQUIRED_FOR_TOP_VENUE | DONE | Built rank-64 protected basis from 4096 embeddings; cumulative variance explained `0.6969`. |
| R043 | M10b | ROP guardrail pilot | No-targeted RRB plus orthogonal residual penalty/projection, anchor 20, seed 1 | Updated public suite | relation, COCO R@1, ImageNet top-1, residual magnitude, protected-subspace energy | REQUIRED_FOR_TOP_VENUE | DONE | Gate failed: relation `0.7441` < `0.76` despite COCO `0.7360` and ImageNet `0.6566`. Treat ROP as failed preservation ablation, not a new direction. |
| R044 | M10c | ROP replication seeds | Best R043 configuration, seeds 2-3 | Updated public suite | relation, COCO/ImageNet pass rate, residual and projection diagnostics | REQUIRED_FOR_TOP_VENUE | SKIPPED | Correctly skipped because R043 failed the pilot decision rule. |
| R045 | M11 | External retrieval validation | OpenCLIP, R005, R028, R038, R043 | Flickr30k retrieval | I2T/T2I R@1/R@5/R@10, drop versus OpenCLIP | REQUIRED_FOR_TOP_VENUE | DONE | R028 Flickr30k I2T R@1 `0.8260` vs OpenCLIP `0.8620`; R038 `0.8440`; R043 `0.8330`. External retrieval tax remains. |

## Launch Rules

1. For the bounded workshop/class-project/thesis writeup, write from the completed evidence without creating a new experiment run.
2. For top-venue robustness, run R041 first, then R042/R043; run R044 only if R043 improves preservation without erasing relation gains.
3. Run R045 only after dataset staging is reproducible and the target requires a non-COCO retrieval claim.
4. Do not rebrand ROP as the main novelty; the novelty check rated it weak, so it is a cited preservation ablation.
5. Before launching R041-R045 through the automated runner, add the matching configs/scripts and update scaffold tests.
