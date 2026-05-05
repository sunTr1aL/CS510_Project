# Experiment Plan

**Problem**: Improve relation, role, and binding sensitivity in CLIP-style dual encoders without hiding the retrieval and zero-shot alignment cost.

**Method Thesis**: Relational Residual Bottlenecking (RRB) keeps frozen OpenCLIP embeddings as the dominant path and learns a bounded slot residual from structured counterfactuals plus guardrail anchoring. The current evidence supports a bounded analysis paper; it does not yet support a robust top-venue preservation claim.

**Date**: 2026-04-30  
**Canonical results**: `refine-logs/CONSOLIDATED_METHOD_RESULTS_2026-04-30.md`  
**Review decision**: Round 2 review is `5.5/10`: stop and write for the workshop/class-project/thesis target; only run the new directions below if the target changes to a top-venue main-track claim.  
**Execution status**: Completed public-suite campaign through R040 plus R034 Winoground. No new training is required for the bounded writeup.

## Claim Map

| Claim | Status | Required Evidence | Boundary |
|---|---|---|---|
| C1: Bounded RRB improves the relation-vs-guardrail tradeoff over hard bottlenecking, no-slot residuals, and plain fine-tuning | Supported with caveats | R019, R028, R029, R031-R040 versus R003/R005 and hard-token R006/R011/R012/R014 | Must report COCO/ImageNet tax and seed fragility; do not claim lossless preservation |
| C2: Structured relation counterfactuals, not generic hard negatives or `L_targeted`, are the useful training signal | Supported | R023/R028 no-targeted runs and R024 generic-hard-negative control | `L_targeted` is optional diagnostic history, not a necessary mechanism |
| A1: Broad binding generalization and interpretable slot specialization are not yet established | Negative / missing | R034 Winoground is flat-to-negative; no slot-responsibility probe exists | Avoid "slot-specialized" or "Winoground-improving" language unless optional blocks pass |
| T1: Orthogonal/null-space residual preservation is a possible guardrail ablation, not headline novelty | Optional top-venue direction | Novelty check rated ROP-style idea `3.5/10`; needs direct ablation against R028/R035-R039 | Frame as prior-art-inspired preservation control, not the main contribution |

## Paper Storyline

1. OpenCLIP has useful broad alignment but is weak on controlled relation edits.
2. Plain/hard bottleneck adaptation can increase relation scores while damaging retrieval and zero-shot guardrails.
3. RRB constrains the edit to a small residual on top of frozen embeddings, using structured counterfactuals and anchor mimicry to make the tradeoff less destructive.
4. The final evidence is a bounded tradeoff result: R028 is the cleanest single checkpoint, R035-R039 expose stricter-anchor seed fragility, R033/R040 show no-slot residuals are unsafe, and R034 blocks a broad Winoground claim.
5. The honest conclusion is "relation gains with explicit alignment tax," not "robustly fixed CLIP."

## Experiment Blocks

### B1: Main Relation/Guardrail Evidence

- **Purpose**: Establish the core relation-vs-preservation tradeoff.
- **Hypothesis**: A bounded residual RRB checkpoint improves relation accuracy while retaining more COCO/ImageNet performance than hard bottleneck or no-slot residual controls.
- **Systems**: OpenCLIP, plain adapter/fine-tuning baselines, hard bottleneck variants, RRB, no-slot residual controls.
- **Datasets**: Staged ARO/SugarCrepe relation suites, COCO retrieval, ImageNet zero-shot.
- **Metrics**: Relation accuracy, COCO I2T/T2I R@1, ImageNet top-1, residual relative magnitude, z0 cosine.
- **Acceptance**: Main table must include both relation gain and guardrail degradation; R028 is the selected checkpoint.
- **Runs**: R003-R017, R019-R025, R027-R040.
- **Status**: DONE. Use `refine-logs/CONSOLIDATED_METHOD_RESULTS_2026-04-30.md`.

### B2: Limitation and Diagnostic Evidence

- **Purpose**: Make the negative and mechanism-boundary evidence explicit.
- **Hypothesis**: The current method is relation-positive but not broadly robust, and slot structure is safer than unconstrained residual capacity.
- **Systems**: R028, R033, R034, R035-R040, R030 diagnostics.
- **Datasets**: Updated public suite plus Winoground.
- **Metrics**: Winoground text/image/group scores, failure taxonomy, residual drift, no-slot guardrail collapse.
- **Acceptance**: Report R034 as a limitation and keep no-slot controls as safety-relevant evidence, not as a solved mechanism proof.
- **Runs**: R030, R033, R034, R040.
- **Status**: DONE. Required for any writeup.

### B3: Slot Responsibility Probe

- **Purpose**: Decide whether the paper can say anything mechanistic about relation slots.
- **Hypothesis**: If slots matter, relation-edited spans should concentrate responsibility in a small subset of slots compared with random or unedited spans.
- **Systems**: R028 selected checkpoint, optional R035/R038 stricter-anchor checkpoints, OpenCLIP baseline where applicable.
- **Datasets**: Existing structured counterfactual audit records and held-out relation examples.
- **Metrics**: Slot responsibility entropy, top-k slot concentration, edited-span overlap, before/after attribution delta.
- **Acceptance**: Use slot language only if responsibility is concentrated and aligned with edited relation spans; otherwise call the bottleneck architectural, not interpretable.
- **Runs**: R041.
- **Status**: TODO only for top-venue claims; optional for bounded writeup.

### B4: ROP / Orthogonal Guardrail Ablation

- **Purpose**: Test the reviewer-suggested residual-direction constraint as a preservation ablation.
- **Hypothesis**: Penalizing or projecting residual edits away from high-variance OpenCLIP embedding directions can reduce COCO/ImageNet damage without erasing relation gains.
- **Systems**: No-targeted RRB with anchor 10 or 20 plus ROP-style orthogonal residual penalty/projection; compare to R028 and R035-R039.
- **Datasets**: Same updated public suite as final RRB runs.
- **Metrics**: Relation accuracy, COCO R@1, ImageNet top-1, residual magnitude, projection energy in protected principal subspace.
- **Acceptance**: Continue only if a seed-1 pilot improves guardrail metrics at similar relation accuracy; otherwise report as failed preservation ablation.
- **Runs**: R042-R044.
- **Status**: TODO only if targeting top-venue robustness. Novelty check says this is not a headline method.

### B5: External Retrieval Validation

- **Purpose**: Check whether the selected method and any ROP variant generalize beyond COCO retrieval.
- **Hypothesis**: A preservation claim needs at least one non-COCO retrieval dataset; otherwise the paper should explicitly remain COCO-bounded.
- **Systems**: OpenCLIP, R003/R005 baselines, R028, best stricter-anchor checkpoint, and best ROP checkpoint only if B4 passes.
- **Datasets**: Flickr30k retrieval or a similarly staged public retrieval suite.
- **Metrics**: I2T/T2I R@1/R@5/R@10 and relative drop from OpenCLIP.
- **Acceptance**: Add to main paper only if dataset staging is reproducible and the result sharpens the claim boundary.
- **Runs**: R045.
- **Status**: TODO only for top-venue robustness.

## Run Order and Milestones

| Milestone | Runs | Decision Rule | Budget | Notes |
|---|---|---|---|---|
| M9 Slot responsibility | R041 | If slot responsibility is diffuse or unaligned with edited spans, remove mechanistic slot language from title/abstract | Eval/analysis only; no new training | Can be run from existing checkpoints and audit records |
| M10 ROP guardrail ablation | R042-R044 | Run seed 1 first; launch replication seeds only if relation stays near R028 while COCO/ImageNet improve | Same cost class as one RRB train+eval per seed | Treat as preservation ablation because novelty is weak |
| M11 External retrieval | R045 | Run after B4 only if the target requires a broader preservation claim; otherwise defer | Eval-only after dataset staging | Flickr30k is the preferred simple check |
| M12 Paper analysis pack | None | Required for bounded writeup: figures and tables cover relation gain, guardrail tax, seed fragility, Winoground negative, and no-slot controls | Local/remote analysis only | This is a writing/analysis task, not an experiment run |

## Compute and Data Budget

- **Bounded writeup path**: zero new training. Only table/figure generation and text revision remain.
- **Top-venue path**: R041 and R045 are analysis/eval tasks; R042-R044 add up to three RRB-sized train/eval runs.
- **Data dependencies**: Existing public-suite staging is sufficient for the bounded writeup. Flickr30k staging is required only if R045 is launched.
- **Implementation dependency**: R041-R045 are planning IDs. Add scripts/config registry entries before launching them through the automated runner.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| More experiments blur the bounded story | Do not run B3-B5 unless the target is explicitly top-venue robustness |
| ROP looks like a new contribution but is novelty-low | Cite it as an orthogonal/null-space preservation ablation and keep RRB as the method |
| Slot wording overclaims mechanism | Run R041 before using mechanistic slot language; otherwise describe slots as an architectural bottleneck only |
| Retrieval preservation remains seed-fragile | Report pass rates and raw drops, not only best checkpoints |
| Winoground remains negative | Use it as external-validity limitation evidence |

## Final Checklist

- [x] Main relation-vs-guardrail table consolidated from all completed public-suite runs
- [x] R028 selected as the paper-facing checkpoint
- [x] Full RRB and no-targeted stricter-anchor seed fragility reported
- [x] No-slot residual controls reported
- [x] Winoground external-validity result reported as a limitation
- [x] No claim that `L_targeted` is necessary
- [x] No claim that broad retrieval is preserved without qualification
- [x] No claim that Winoground generalization improved
- [ ] Paper analysis pack generated for the bounded writeup
- [ ] R041 slot responsibility completed before any mechanistic slot claim
- [ ] R042-R045 completed only if the target changes to top-venue robustness
