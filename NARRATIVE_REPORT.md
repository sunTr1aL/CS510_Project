# Narrative Report: Guarded Relational Residual Adaptation for CLIP

## Core Story

CLIP-style dual encoders are strong at broad image-text alignment, but they often match captions by object co-occurrence while underweighting relation, role, and binding changes. This project began from that failure mode: if a caption changes "left of" to "right of", swaps attributes, or changes a relation while preserving object vocabulary, the embedding should move in a way that affects image-text similarity. The difficult part is that relation supervision can be too strong. The early hard-bottleneck variants showed that relation signal is reachable, but replacing the pretrained CLIP path destroyed COCO retrieval and ImageNet zero-shot behavior.

The final method is Relational Residual Bottlenecking (RRB). RRB keeps the frozen OpenCLIP ViT-B/16 embedding `z0` as the dominant path and learns only a small parser-free residual correction from slot-bottlenecked token features:

```text
z = normalize(z0 + alpha * delta_rel)
```

The residual is trained with structured relation counterfactuals, anchor feature mimicry to the frozen OpenCLIP embedding, and residual norm regularization. The final paper-facing version removes `L_targeted` from the central claim: the evidence says targeted slot supervision is not necessary for the observed relation gain. The "slot" component should be described as an architectural bottleneck or safety prior, not as an interpretable semantic slot mechanism.

The main result is a bounded tradeoff, not a solved preservation claim. The selected checkpoint, R028, improves relation accuracy from the plain-adapter baseline R005 at `0.6956` to `0.7734`, while keeping COCO image-to-text R@1 at `0.7340` and ImageNet top-1 at `0.6342`. This is far safer than hard token bottlenecking or no-slot residual controls, but it still pays an alignment tax versus OpenCLIP and remains seed-fragile on COCO. Winoground and Flickr30k follow-ups make the boundary sharper: the current method improves staged relation suites under explicit guardrails, but it does not yet establish broad binding generalization or lossless retrieval preservation.

## Claims

1. **RRB improves the relation-vs-guardrail tradeoff over plain fine-tuning and hard bottlenecking.**  
   R028 reaches relation `0.7734`, a `+0.0778` absolute gain over R005, with COCO I2T R@1 `0.7340` and ImageNet top-1 `0.6342`. Hard token variants can raise relation accuracy, but their guardrails collapse: R011 relation `0.7488` comes with COCO `0.3920` and ImageNet `0.0660`.

2. **Structured relational counterfactuals are the useful training signal; generic hard negatives are not enough.**  
   R024 preserves broad CLIP behavior with COCO `0.7990` and ImageNet `0.6676`, but relation falls to `0.6736`, below the plain-adapter baseline. This supports the claim that relation-specific counterfactuals are doing real work.

3. **`L_targeted` is not a necessary mechanism claim.**  
   R023 removes targeted loss and reaches relation `0.7922`, though it misses the COCO gate. R028 also uses `L_targeted=0` and passes the pilot guardrails. The supported method is structured residual adaptation with anchoring, not targeted slot supervision.

4. **The slot-bottlenecked residual interface is safety-relevant, but semantic slot specialization is unproven.**  
   No-slot residual controls are relation-strong but unsafe: R033 reaches relation `0.7825` but COCO `0.3740`; R040 still has relation `0.7773` but COCO `0.4820` and ImageNet `0.4798`. However, R041 slot-responsibility probes on R028/R035/R038 are diffuse, with entropy near `1.0000` and top-1 slot concentration near `0.126`, so the report should not claim interpretable relation-specialized slots.

5. **The final result is a bounded analysis result, not a top-venue robustness result.**  
   Full RRB and stronger-anchor variants reproduce relation gains, but COCO pass rates are imperfect. Winoground is flat-to-negative for R028, and Flickr30k shows an external image-to-text retrieval tax. The honest conclusion is "relation gains with explicit alignment cost."

## Experiments

### Setup

- **Models**: OpenCLIP ViT-B/16 with `laion2b_s34b_b88k`; frozen image and text encoders; trainable RRB modules on both branches. The selected R028 configuration uses `K=8` slots, fixed residual gate `alpha=0.05`, `lambda_anchor=10`, `lambda_delta=0.01`, `anchor_ratio=0.75`, and no targeted slot loss.
- **Data**: Structured relation/counterfactual suites from staged ARO and SugarCrepe sources, COCO retrieval with 1,000 images and 5,002 captions, ImageNet zero-shot with 5,000 examples, Winoground with 400 examples, and Flickr30k retrieval with 1,000 images for the final external check.
- **Hardware**: Remote `timan1.cs.illinois.edu` server, CUDA runs on NVIDIA RTX A6000 GPUs. The server currently exposes four RTX A6000 GPUs with 49,140 MiB memory each. Runs used the `relclip` conda environment and no wandb logging.
- **Training / evaluation defaults**: Remote launch scripts use `TRAIN_EXAMPLES=2048`, `EVAL_LIMIT_PER_KEY=512`, `EPOCHS=5`, `BATCH_SIZE=64`, and `LR=1e-4` for the RRB campaign unless a run-specific control overrides those settings.
- **Baselines and controls**: R003 pretrained OpenCLIP, R005 plain adapter, hard token bottlenecks R006/R011/R012/R014, full RRB R019/R025/R027/R029, no-targeted RRB R023/R028/R031/R032/R035-R039, generic-hard-negative R024, no-slot residual controls R033/R040, ROP ablation R043, and external evaluations R034/R045.
- **Source audit**: The local linked `FINAL_PROPOSAL.md` and `EXPERIMENT_PLAN.md` are newer than the server copies, so this report uses the local versions for the current method framing and plan. `EXPERIMENT_RESULTS.md`, `EXPERIMENT_TRACKER.md`, and `CONSOLIDATED_METHOD_RESULTS_2026-04-30.md` match the server checksums. The R041-R045 analysis copy also matches the remote output root under `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_topvenue_20260430_163958/analysis/`.

### Experiment 1: Main Relation/Guardrail Tradeoff

This experiment asks whether relation sensitivity can improve without the hard-bottleneck collapse observed in early variants. The pilot gate is relation accuracy above R005, COCO I2T R@1 at least `0.70`, and ImageNet top-1 at least `0.55`.

| Method | Relation Acc. | Delta vs R005 | COCO I2T R@1 | ImageNet Top-1 | Residual Mag. | z0 Cosine | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| R003 OpenCLIP | 0.6564 | -0.0392 | 0.7990 | 0.6670 | - | - | baseline |
| R005 plain adapter | 0.6956 | +0.0000 | 0.7790 | 0.6438 | - | - | safe baseline |
| R011 hard token targeted | 0.7488 | +0.0532 | 0.3920 | 0.0660 | - | - | collapse |
| R014 hard token no targeted | 0.7332 | +0.0376 | 0.4110 | 0.0602 | - | - | collapse |
| R019 full RRB seed 1 | 0.7697 | +0.0741 | 0.7190 | 0.6152 | 0.1328 | 0.9900 | pass |
| **R028 selected no-targeted RRB** | **0.7734** | **+0.0778** | **0.7340** | **0.6342** | **0.0804** | **0.9963** | **selected** |
| R033 no-slot residual | 0.7825 | +0.0869 | 0.3740 | 0.3308 | 0.2062 | 0.9754 | collapse |
| R040 constrained no-slot residual | 0.7773 | +0.0817 | 0.4820 | 0.4798 | 0.0969 | 0.9942 | guardrail-fragile |

**Interpretation**: RRB is the only path that produces a clear relation gain while keeping the broad alignment guardrails in a usable range. Hard bottlenecking and no-slot residuals show that relation accuracy alone is an unsafe selection criterion. R028 is weaker than OpenCLIP on COCO and ImageNet, but the damage is bounded rather than catastrophic.

### Experiment 2: Structured Counterfactuals and Targeted-Loss Ablations

This experiment separates the useful training signal from the mechanisms that should not be overclaimed.

| Method | Structured Counterfactuals | `L_targeted` | Relation Acc. | COCO I2T R@1 | ImageNet Top-1 | Interpretation |
|---|---|---:|---:|---:|---:|---|
| R023 no-targeted RRB | yes | 0 | 0.7922 | 0.6820 | 0.5982 | relation-strong, COCO fail |
| R024 generic hard negatives | no | 1 | 0.6736 | 0.7990 | 0.6676 | guardrails preserved, relation fail |
| **R028 no-targeted stronger-anchor RRB** | **yes** | **0** | **0.7734** | **0.7340** | **0.6342** | **best paper-facing tradeoff** |
| R029 matched full stronger-anchor RRB | yes | 1 | 0.7731 | 0.7280 | 0.6462 | passes, but less simple |

**Interpretation**: The useful ingredient is structured relational counterfactual training under residual anchoring. Generic hard negatives do not teach the same relation behavior, and targeted slot loss is not needed for the supported effect.

### Experiment 3: Seed Stability and Stronger Anchoring

This experiment checks whether relation gains and guardrail behavior replicate across seeds and anchor strengths.

| Setting | Metric | Mean | Sample Std | Pass Rate / Note |
|---|---|---:|---:|---|
| Full RRB R019/R025/R027 | Relation | 0.7768 | 0.0062 | 3/3 relation-positive |
| Full RRB R019/R025/R027 | COCO I2T R@1 | 0.6927 | 0.0294 | 1/3 pass |
| Full RRB R019/R025/R027 | ImageNet Top-1 | 0.5898 | 0.0337 | all above 0.55, one low |
| No-targeted anchor 10 R028/R031/R032 | Relation | 0.7782 | 0.0042 | 3/3 relation-positive |
| No-targeted anchor 10 R028/R031/R032 | COCO I2T R@1 | 0.7110 | 0.0332 | 2/3 pass |
| No-targeted anchor 20 R035-R039 | Relation | 0.7789 | 0.0032 | 5/5 relation-positive |
| No-targeted anchor 20 R035-R039 | COCO I2T R@1 | 0.7092 | 0.0232 | 3/5 pass |
| No-targeted anchor 20 R035-R039 | Residual Mag. | 0.0666 | 0.0031 | lower drift than R028 |

**Interpretation**: The relation improvement is stable across seeds, but the COCO guardrail is not. Stronger anchoring reduces residual magnitude and improves some ImageNet behavior, but it does not eliminate retrieval fragility. This is why the report should present a Pareto tradeoff and pass rates, not only the selected checkpoint.

### Experiment 4: External Validity and Top-Venue Follow-Ups

This experiment checks whether the bounded relation result extends to broader binding, interpretable slots, orthogonal preservation, and non-COCO retrieval.

| Check | Key Result | Verdict |
|---|---|---|
| R034 Winoground | R028 text/image/group `0.2400/0.1100/0.0700` vs OpenCLIP `0.2775/0.1075/0.0825` | flat-to-negative; no broad binding claim |
| R041 slot responsibility | entropy near `1.0000`, top-1 concentration near `0.126`, edited overlap near `0.140` for R028/R035/R038 | diffuse; no interpretable slot claim |
| R043 ROP ablation | relation `0.7441`, COCO `0.7360`, ImageNet `0.6566` | guardrails marginally improve, relation gate fails |
| R045 Flickr30k R028 | I2T R@1 `0.8260` vs OpenCLIP `0.8620`; T2I R@1 `0.6962` vs `0.6980` | external I2T retrieval tax |
| R045 Flickr30k R038 | I2T R@1 `0.8440` vs OpenCLIP `0.8620`; T2I R@1 `0.6964` vs `0.6980` | better than R028 but still not lossless |

**Interpretation**: The follow-ups strengthen the limitation boundary rather than changing the headline. RRB remains a useful relation-vs-guardrail tradeoff method on staged relation suites, but it is not yet a broad compositional reasoning fix for CLIP.

## Figures

1. **Figure 1: Relation vs. guardrail Pareto plot.** Scatter all key runs from R003-R040 with relation accuracy on the x-axis and COCO I2T R@1 or ImageNet top-1 on the y-axis. Mark R028, hard-token variants, no-slot controls, and anchor-20 seeds. Data source: `refine-logs/CONSOLIDATED_METHOD_RESULTS_2026-04-30.md`.
2. **Table 1: Main method comparison.** Use the Experiment 1 table above as the primary quantitative table. It should include relation accuracy, COCO I2T R@1, ImageNet top-1, residual magnitude, and gate status.
3. **Figure 2: Seed and anchoring summary.** Bar or point-range plot for full RRB, anchor-10 no-targeted RRB, and anchor-20 no-targeted RRB. Show mean and sample standard deviation for relation and COCO.
4. **Table 2: Ablation table for supervision and targeted loss.** Use R023/R024/R028/R029 to show that structured counterfactuals matter and `L_targeted` is optional.
5. **Figure 3: R030 failure taxonomy.** Compact horizontal bar plot of high-count remaining failures: ARO left/right spatial predicates, SugarCrepe object/attribute swaps, and relation replacements. Data source: `refine-logs/R030_DIAGNOSTICS.md` and `refine-logs/R030_diagnostics_summary.json`.
6. **Table 3: External validity limits.** Combine R034 Winoground, R041 slot responsibility, R043 ROP, and R045 Flickr30k results to make the claim boundary explicit.

## Known Weaknesses

- **COCO guardrail fragility remains the central weakness.** R028 passes the pilot COCO gate, but seed groups show imperfect pass rates: full RRB passes `1/3`, anchor-10 no-targeted passes `2/3`, and anchor-20 no-targeted passes `3/5`.
- **The method pays an alignment tax.** R028 drops COCO I2T R@1 from OpenCLIP `0.7990` to `0.7340` and from R005 `0.7790` to `0.7340`. Flickr30k confirms a non-COCO I2T retrieval drop.
- **Winoground does not improve.** R028 improves image score by only `+0.0025` over OpenCLIP while losing text and group score.
- **Semantic slot specialization is not demonstrated.** R041 finds diffuse responsibility, so the paper should avoid mechanistic slot claims unless a new probe or ablation changes that evidence.
- **ROP is not a productive headline pivot.** The reviewer-suggested orthogonal projection idea has low novelty relative to null-space/principal-subspace PEFT, and the R043 pilot fails the relation gate.
- **Evaluation is still bounded.** COCO and Flickr30k retrieval are sampled at 1,000 images, ImageNet at 5,000 examples, and the relation suites are staged public counterfactual benchmarks rather than a full deployment distribution.
- **Residual magnitude is diagnostic, not sufficient.** Low drift helps but does not guarantee safe retrieval behavior; residual direction and objective still matter.

## Related Work

- **Relation-aware and compositional CLIP alignment**: RACA-CLIP, SugarCrepe-style compositional evaluations, ARO relation/order tests, and related compositional contrastive work directly motivate the project. RRB differs by keeping the frozen CLIP embedding path dominant and treating relation supervision as a bounded residual correction instead of a replacement representation.
- **Scene-graph and structured image-text matching**: Cross-modal Scene Graph Matching, TSGR2, SEMScene, StructXLIP, and related graph/structure-aware VLM methods use explicit structure, graph matching, or structural cues. RRB intentionally avoids parsers, detectors, and test-time graph extraction; its supervision can be structured, but inference remains a normal dual encoder.
- **Spatial and binding benchmarks**: Winoground, ARO, SugarCrepe, SpatialVLM-style evaluations, and recent causal/order VLM probes expose the failure mode this project targets. The current evidence is strongest on ARO/SugarCrepe-style staged relation suites and negative on Winoground.
- **Preservation-oriented adapters and PEFT**: Residual adapters, feature mimicry, GNSP, KeepLoRA, PEGP/PSOFT-style principal or null-space methods, and visual prompt/null-space approaches are close to the preservation side. This project should not claim novelty from residuals or orthogonal projection alone. Its defensible contribution is the specific structured-counterfactual residual tradeoff plus the hard-bottleneck diagnostic evidence.
- **LLM/VLM-assisted data generation**: Modern text or VLM tools can generate and filter relation counterfactual captions. RRB uses that only offline for supervision and filtering; no LLM, cross-encoder, scene graph parser, or reranker is used at inference.

## Proposed Title

Guarded Relational Residual Adaptation for CLIP

## Target Venue

Workshop, class-project, or thesis-style ML venue in 2026. The current evidence is not ready for a top ML main-track robustness claim without stronger external retrieval, slot-mechanism evidence, and better COCO seed stability.
