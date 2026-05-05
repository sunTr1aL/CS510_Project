# Paper Plan

**Title**: Anchored Relational Residual Adaptation for CLIP  
**Venue**: ACM Proceedings (SIGCONF-style)  
**Type**: Empirical method and analysis paper  
**Date**: 2026-05-02  
**Page budget**: No active page limit for this draft; prioritize completeness and claim-evidence alignment over compression.  
**Section count**: 6 main sections plus appendix

## Assumptions

- "ACM Proceedings" is interpreted as an ACM SIGCONF-style proceedings paper using anonymous review formatting.
- The user explicitly asked to ignore page limits for now.
- The paper should be written as a bounded workshop/class-project/thesis-quality analysis result, not as a top-venue robustness claim.
- `NARRATIVE_REPORT.md`, `refine-logs/CONSOLIDATED_METHOD_RESULTS_2026-04-30.md`, `refine-logs/R030_DIAGNOSTICS.md`, and `outputs/remote/rrb_topvenue_20260430_163958/analysis/R041_R045_results.md` are the current local sources of truth.

## Claims-Evidence Matrix

| Claim | Evidence | Status | Section |
|---|---|---|---|
| RRB improves relation sensitivity while avoiding the catastrophic guardrail collapse of hard token bottlenecks. | R028 relation `0.7734` vs R005 `0.6956`; COCO I2T R@1 `0.7340`; ImageNet top-1 `0.6342`; hard-token R011 relation `0.7488` but COCO `0.3920` and ImageNet `0.0660`. | Supported with bounded preservation caveat | Section 4 |
| Structured relational counterfactuals are the useful supervision signal; generic hard negatives are not enough. | R024 generic hard negatives preserve COCO `0.7990` and ImageNet `0.6676`, but relation falls to `0.6736`, below R005. | Supported | Sections 3, 4 |
| `L_targeted` is not a necessary mechanism claim. | R023 reaches relation `0.7922` with `L_targeted=0` but fails COCO; R028 uses `L_targeted=0` and passes pilot guardrails; R029 with targeted loss is similar or weaker on COCO. | Supported as an anti-claim | Sections 3, 4 |
| A multi-channel bottlenecked residual interface is preservation-relevant, but semantic slot specialization is unproven. | No-slot R033 relation `0.7825` but COCO `0.3740`, ImageNet `0.3308`; constrained no-slot R040 still COCO `0.4820`; R041 responsibility probes show entropy near `1.0000` and top-1 concentration near `0.126`. | Preservation prior supported; interpretability unsupported | Sections 4, 5 |
| The final result is a relation-vs-guardrail tradeoff, not broad binding generalization or lossless retrieval preservation. | Stronger-anchor seed groups keep relation stable but COCO pass rate is imperfect; Winoground R034 is flat-to-negative; Flickr30k I2T R@1 drops from OpenCLIP `0.8620` to R028 `0.8260`. | Supported limitation | Sections 4, 5 |

## Structure

### Abstract

- **Problem**: CLIP-style dual encoders can match objects while underweighting relation, role, and binding changes.
- **Approach**: Relational Residual Bottlenecking (RRB), a parser-free residual module that keeps frozen OpenCLIP embeddings as the dominant path and adds a bounded relation-trained correction.
- **Key result**: R028 raises relation accuracy from R005 `0.6956` to `0.7734` while keeping COCO I2T R@1 `0.7340` and ImageNet top-1 `0.6342`; hard bottlenecks collapse the same guardrails.
- **Implication**: Structured counterfactual residual adaptation can improve relation sensitivity, and the failed targeted-loss hypothesis is itself a useful negative result; broad retrieval preservation and semantic channel interpretability remain unproven.
- **Estimated length**: 160-200 words.

### Section 1: Introduction

- **Opening hook**: Modern image-text encoders are useful because a single embedding space supports retrieval and zero-shot recognition, but that same coarse alignment can blur relational edits.
- **Gap**: Prior relation-aware methods often use explicit structure, stronger reranking, or representation replacement; preservation-oriented adapters often protect broad alignment without improving relation sensitivity. This draft defines "guardrails" narrowly as retention of retrieval and zero-shot behavior, not red-team or content-safety behavior.
- **Research question**: Can relation sensitivity be added as a small residual correction while keeping the pretrained embedding path dominant?
- **Contributions**:
  1. Define RRB, a parser-free multi-channel residual adapter for frozen OpenCLIP image and text encoders.
  2. Show that RRB improves staged relation accuracy with bounded guardrail damage, unlike hard bottlenecks and no-slot residual controls.
  3. Identify the real training signal: structured relational counterfactuals, not generic hard negatives or the earlier targeted slot loss.
  4. Report limitations directly: COCO seed fragility, negative Winoground transfer, Flickr30k I2T tax, and diffuse slot responsibility.
- **Hero figure**: Figure 1, relation-vs-COCO Pareto plot across key runs, highlighting OpenCLIP, plain adapter, hard token bottlenecks, selected RRB, anchor-20 seeds, and no-slot controls.
- **Estimated length**: 1.0-1.2 ACM pages.
- **Key citations**: CLIP, OpenCLIP, ARO, SugarCrepe, Winoground, RACA-CLIP.

### Section 2: Related Work

- **Relation-aware and compositional CLIP alignment**: Position against RACA-CLIP, ARO/SugarCrepe-style evaluations, and compositional contrastive approaches.
- **Structured image-text matching**: Position against scene-graph and structure-heavy methods such as cross-modal scene graph matching, TSGR2, SEMScene, StructXLIP.
- **Binding and spatial benchmarks**: Explain why ARO/SugarCrepe and Winoground test related but different generalization claims.
- **Preservation-oriented PEFT and adapters**: Connect to residual adapters, feature mimicry, null-space/principal-subspace methods, KeepLoRA, GNSP, PSOFT, and explain why ROP is an ablation, not the contribution.
- **Estimated length**: 1 ACM page.

### Section 3: Relational Residual Bottlenecking

- **Notation**: Frozen OpenCLIP image/text embeddings `z0_v`, `z0_t`; token sequences `P_v`, `P_t`; channel states `S_v`, `S_t`; residuals `delta_v`, `delta_t`; residual gate `alpha`.
- **Method**: `z = normalize(z0 + alpha * delta_rel)` for each branch.
- **Architecture**: Frozen OpenCLIP ViT-B/16, parser-free latent residual channels over CLIP tokens, one channel self-attention block, branch-specific residual projection.
- **Objective**: `L = lambda_intervene L_intervene + lambda_anchor L_anchor + lambda_delta L_delta`; `L_intervene` is a structured margin loss on edited relation pairs, `L_anchor` mimics frozen OpenCLIP features on anchor pairs, and `L_delta` penalizes residual feature norm/drift. Targeted slot loss is excluded from the final method and treated as a diagnostic control.
- **Why residual and bottleneck**: Residual keeps the pretrained path; the channel bottleneck constrains the correction interface without claiming interpretable semantic slots.
- **Estimated length**: 1.3-1.5 ACM pages.

### Section 4: Experiments

- **Setup**: OpenCLIP ViT-B/16 LAION2B, staged ARO/SugarCrepe relation suites, COCO 1k retrieval, ImageNet 5k zero-shot, Winoground 400, Flickr30k 1k retrieval, RTX A6000 server, no wandb.
- **Main result**: Table 1 plus Figure 1. Compare R003, R005, hard token variants, selected R028, no-slot controls, and constrained no-slot controls. Do not claim dominance over unrun LoRA or other standard PEFT baselines; list those as missing baselines.
- **Ablations**: Table 2 for structured counterfactuals and targeted loss; Figure 2 for seed and anchoring summaries.
- **External validity**: Table 3 for Winoground, slot responsibility, ROP, Flickr30k.
- **Estimated length**: 2.8-3.2 ACM pages.

### Section 5: Analysis and Limitations

- **Residual drift**: Lower residual magnitude helps but does not guarantee anchor preservation; R023/R028/R029 and R035-R039 illustrate direction/objective sensitivity.
- **Generalization gap**: Explain why ARO/SugarCrepe-style structured counterfactual suites improve while Winoground and Flickr30k do not. The likely explanation is partial adaptation to benchmark-style relational edits rather than broad compositional semantics.
- **Failure taxonomy**: Figure 3 shows remaining high-count failures: left/right spatial predicates, SugarCrepe swaps, attribution, and relation replacement.
- **Claim boundary**: No broad Winoground binding claim, no robust retrieval preservation claim, no interpretable slot claim, no originality claim for ROP.
- **Estimated length**: 1.0-1.2 ACM pages.

### Section 6: Conclusion

- **Restatement**: RRB demonstrates that relation sensitivity can be improved through bounded residual adaptation while preserving more of CLIP than replacement bottlenecks.
- **Limitations**: Alignment tax, seed-fragile COCO gate, external validity gap, and unproven slot semantics.
- **Future work**: Better residual-direction constraints, broader retrieval validation, and direct slot-ablation/mechanism probes.
- **Estimated length**: 0.35-0.5 ACM pages.

### Appendix

- Extended run table for R003-R045.
- Per-source relation results.
- Additional details on diagnostics, hyperparameters, and remote roots.
- BibTeX/source notes for citations that could not be fully verified from local files.

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|---|---|---|---|---|
| Figure 1 | Scatter/Pareto | Relation accuracy versus COCO I2T R@1 for key systems. Highlight selected R028, OpenCLIP/R005 baselines, hard-token collapse, no-channel collapse, and anchor-20 seeds. Caption must state that standard LoRA/text-only/vision-only baselines were not run. | `refine-logs/CONSOLIDATED_METHOD_RESULTS_2026-04-30.md` | HIGH |
| Figure 2 | Grouped bar/point summary | Mean relation and COCO for full RRB, no-targeted anchor-10, and no-targeted anchor-20 seed groups with sample std. | `NARRATIVE_REPORT.md`; consolidated results | HIGH |
| Figure 3 | Horizontal bar | R030 failure taxonomy by high-count failure groups. | `refine-logs/R030_diagnostics_summary.json` | MEDIUM |
| Table 1 | LaTeX table | Main method comparison: relation, COCO I2T R@1, ImageNet top-1, residual magnitude, z0 cosine, verdict. | Consolidated results | HIGH |
| Table 2 | LaTeX table | Supervision/targeted-loss ablation: R023, R024, R028, R029. | Consolidated results | HIGH |
| Table 3 | LaTeX table | External validity and mechanism limits: Winoground, slot responsibility, ROP, Flickr30k. | R041-R045 summary | HIGH |

## Citation Plan

- **Introduction**: CLIP/OpenCLIP; ARO; SugarCrepe; Winoground; compositionality and relation-aware CLIP work.
- **Related Work**:
  - Relation-aware CLIP/compositional alignment: RACA-CLIP, SugarCrePe, ARO.
  - Structured image-text matching: cross-modal scene graph matching, TSGR2, SEMScene, StructXLIP.
  - Benchmarks: Winoground, SpatialVLM-style spatial reasoning probes.
  - Preservation/adapters: residual adapters, visual prompt tuning/null-space methods, GNSP, KeepLoRA, principal-subspace or orthogonal PEFT.
- **Method**: CLIP/OpenCLIP; residual/adapter and preservation citations where verified.
- **Citation verification rule**: Use existing local literature filenames and DBLP/CrossRef/arXiv lookups where possible; if exact venue metadata is unavailable, keep entries conservative and mark them in comments rather than inventing details.

## Reviewer Feedback

Gemini outline review scored the initial plan `6.5/10`. Minimum fixes applied:

- Recalibrated "slot" language to "multi-channel" or "bottlenecked residual channels" except when discussing the failed slot-responsibility probe.
- Defined guardrails as retrieval and zero-shot anchor preservation, not content safety.
- Added an explicit generalization-gap analysis for ARO/SugarCrepe versus Winoground/Flickr30k.
- Required the method section to define `L_delta`, `L_anchor`, and the fixed residual gate.
- Added a limitation that LoRA/text-only/vision-only PEFT baselines were not run, instead of claiming an unsupported Pareto dominance over them.
- Elevated the targeted-loss anti-claim as a key insight.

## Next Steps

- [ ] Generate data figures and LaTeX tables in `figures/`.
- [ ] Draft ACM LaTeX in `paper/`.
- [ ] Compile `paper/main.pdf`.
- [ ] Run four Gemini review/fix/recompile rounds.
