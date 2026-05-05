# Research Proposal: Guarded Relational Residual Bottlenecking for CLIP

## Problem Anchor

- Bottom-line problem: Improve CLIP-style dual-encoder embeddings so relation, role, and binding changes affect image-text similarity rather than being washed out by object co-occurrence.
- Must-solve bottleneck: Hard relational bottlenecks can expose relation signal but destroy the pretrained CLIP manifold; plain fine-tuning preserves the manifold but gives weaker relation gains. The method must improve relation sensitivity under explicit COCO retrieval and ImageNet zero-shot guardrails.
- Non-goals: This is not a multimodal LLM, chain-of-thought reasoner, scene-graph parser, detector-heavy pipeline, VQA model, or task-specific reranker. It is also not a claim that `L_targeted` is necessary unless a redesigned targeted loss is separately validated.
- Constraints: Use the current OpenCLIP ViT-B/16 scaffold and staged public evaluation suites: ARO/SugarCrepe relation tests, COCO retrieval with 1,000 images, ImageNet zero-shot with 5,000 examples, and the completed Winoground check. Keep inference as a normal dual encoder. Keep the backbone frozen for the current paper path. Use the existing remote experiment workflow without wandb. Treat Flickr30k-style retrieval as the next external guardrail if the target venue requires broader retrieval evidence.
- Success condition: A final method beats the R005 plain-adapter relation baseline while avoiding the hard-bottleneck guardrail collapse; structured relational counterfactual training must beat generic hard negatives; COCO/ImageNet degradation must be bounded and reported as an alignment tax rather than hidden.

## Technical Gap

CLIP-style dual encoders often match images and captions through object co-occurrence while underweighting relation, role, and binding changes. The local experiments show that relation signal is reachable but easy to extract in the wrong way. Hard token bottlenecks improve relation accuracy but catastrophically damage COCO retrieval and ImageNet zero-shot behavior. Plain adapter fine-tuning and generic hard negatives preserve broad CLIP behavior but do not produce the same relation gains.

The missing mechanism is a constrained interface between relation supervision and the pretrained embedding manifold. The model should not replace CLIP's embedding with a new bottleneck representation. It should keep the frozen embedding `z0` as the dominant path and learn only a small relation-specific correction when structured counterfactual evidence requires it.

## Method Thesis

- One-sentence thesis: Relational Residual Bottlenecking learns a parser-free slot-bottlenecked residual correction on top of frozen CLIP embeddings, improving relation-sensitive matching while bounding damage to broad retrieval and zero-shot behavior.
- Smallest adequate intervention: one slot self-attention block per branch, one residual projection, a bounded gate, structured counterfactual supervision, and anchor regularization against `z0`.
- Current claim boundary: the paper should claim a better relation-vs-guardrail tradeoff, not lossless CLIP preservation, top-venue-ready robustness, targeted-slot-loss necessity, Winoground improvement, or semantic slot interpretability. The no-slot controls support the slot-bottlenecked residual interface as a safety-relevant architectural prior, but they do not prove relation-specialized slots.

### 2026-04-30 Review Decision

A fresh external review scored the current package `5/10` for a top ML venue and judged it `NOT READY`. The review did not identify a necessary near-term method pivot; it recommended reframing and analysis. The strongest blockers are Winoground regression, COCO seed fragility, and lack of direct slot-responsibility evidence.

The reviewer suggested a Relational Orthogonal Projection direction: constrain `delta_rel` to be orthogonal to principal components of the frozen CLIP embedding space. A follow-up novelty check rated this direction `3.5/10` because recent null-space and principal-subspace PEFT work for VLMs already covers the core mechanism. Therefore ROP should not replace RRB as the headline method. It can be a future guardrail ablation inspired by prior work, not a new method claim.

Round 2 re-review after this reframing scored the package `5.5/10`: ready for a bounded workshop/thesis-style analysis paper, still not ready for a main-track top venue. The recommended stopping point is to write the paper now as a characterization of relational adaptation difficulty. Unless a slot-responsibility analysis is added, avoid using "slot" in the title or abstract; title the work around bottlenecked residual adaptation instead.

## Contribution Focus

- Dominant contribution: A guardrail-constrained residual adaptation mechanism for relation-sensitive CLIP embeddings.
- Supporting contribution: A diagnostic comparison showing why hard bottleneck replacement is unsafe and why structured counterfactuals are more useful than generic hard negatives.
- Explicit non-contributions: No parser, detector, graph extractor, multimodal reasoning decoder, test-time LLM, broad SOTA benchmark claim, or necessary `L_targeted` claim.

## Proposed Method

### Complexity Budget

- Frozen / reused backbone: OpenCLIP ViT-B/16 image and text encoders, tokenizer, preprocessing, and current pairwise relation evaluation harness.
- New trainable components: an image-side and text-side RRB module, each with learned slot queries, slot-to-token attention, one slot self-attention block, a readout, and a residual projection.
- Excluded additions: explicit scene graphs, object detectors, GNN edge passing, cross-encoder reranking, symmetric multi-stage reasoning, extra relation heads, test-time LLM calls, and broad backbone unfreezing.

### System Overview

```text
Image I -> frozen OpenCLIP image tokens Pv and base embedding z0_v
        -> visual slot states Sv
        -> residual direction delta_v
        -> z_v = normalize(z0_v + alpha_v * delta_v)

Text T  -> frozen OpenCLIP text tokens Pt and base embedding z0_t
        -> text slot states St
        -> residual direction delta_t
        -> z_t = normalize(z0_t + alpha_t * delta_t)

Similarity(I, T) = z_v dot z_t
```

The slot module is parser-free. Learned slot queries cross-attend to frozen CLIP tokens; a single slot self-attention layer lets slots exchange context; the pooled slot state is projected into CLIP embedding space as a residual direction. The final representation remains a normalized CLIP-like embedding. Until a direct responsibility or ablation analysis is run, these are architectural slots, not proven semantic relation slots.

### Core Mechanism

- Input / output: frozen token features and frozen base embedding `z0`; output `z = normalize(z0 + alpha * delta_rel)`.
- Default architecture: `K=8` slots, one slot self-attention block, mean slot pooling, linear residual projection, fixed small gate such as `alpha=0.05` or a capped learned gate.
- Core training signal: structured relation/role counterfactual ranking plus preservation regularization.
- Preservation signal: feature mimicry to frozen OpenCLIP embeddings and residual norm regularization.
- Model selection: choose checkpoints on a relation-vs-guardrail Pareto frontier with predeclared COCO/ImageNet gates.

The core objective is:

```text
L = w_task * L_intervene
  + lambda_anchor * L_anchor
  + lambda_delta * L_delta
```

where `L_intervene` ranks the true caption above a structured relation/role counterfactual, `L_anchor` keeps final image/text embeddings close to frozen OpenCLIP embeddings on clean anchor examples, and `L_delta` discourages large residual corrections.

`L_targeted` is not part of the central method claim. The current `R023` result falsifies any simple claim that it is necessary for relation accuracy. It can remain as an appendix diagnostic or stabilization ablation, but the paper should headline bounded residual relation adaptation rather than targeted-slot supervision.

### Modern Primitive Usage

The method uses foundation-model-era tools only where they directly serve the bottleneck:

- A text LLM can generate structured counterfactual captions that alter relation, role, order, or binding while preserving object vocabulary.
- A frozen VLM/CLIP scorer can filter ambiguous edits and provide margin diagnostics.
- Neither component is used during inference.

This keeps the deployed model a compact dual encoder while using modern models to create the supervision that manual scene-graph annotation would otherwise provide.

### Training Plan

1. Build structured counterfactual triplets with minimal lexical drift.
2. Filter ambiguous counterfactuals with a frozen scorer and maintain a small manual audit set.
3. Train only the RRB residual modules on structured triplets plus clean anchor examples.
4. Select checkpoints under predeclared guardrails, currently COCO I2T R@1 `>= 0.70` and ImageNet top-1 `>= 0.55` for the pilot suite.
5. Report the Pareto frontier, not only the best relation score.
6. Compare against pretrained OpenCLIP, plain adapter fine-tuning, hard token bottlenecking, no-targeted RRB, and generic-hard-negative RRB.

### Failure Modes and Diagnostics

- COCO retrieval fragility:
  - Detect with COCO I2T R@1 across seeds and checkpoints.
  - Mitigate by reporting bounded alignment tax, tuning anchor strength conservatively, and avoiding post-hoc gate changes.
- Top-venue readiness gap:
  - Detect with R028 seed replications, no-slot residual adapter controls, Winoground evaluation, and direct slot-responsibility analysis.
  - Mitigate by keeping the current story at workshop/class-project evidence level unless COCO fragility, external validity, and slot-mechanism evidence improve.
- Residual overcorrection:
  - Detect with residual relative magnitude and `cos(z0, z)`.
  - Mitigate with fixed small gates, stronger anchoring, and Pareto selection.
- Generic hard negatives fail to teach relations:
  - Detect with matched generic-negative controls.
  - Current evidence supports this: `R024` preserves COCO/ImageNet but relation accuracy falls below the plain adapter.
- Slot responsibility is not interpretable:
  - Detect with edited-span overlap and entropy diagnostics.
  - Mitigate by keeping slot-localization evidence in the appendix, not as a main claim.
- `L_targeted` remains inconclusive:
  - Detect with matched no-targeted controls.
  - Current evidence says it should not be claimed as necessary.

## Novelty and Elegance Argument

RRB is closest to relation-aware CLIP alignment, compositional CLIP adaptation, residual adapters, and scene-graph image-text matching. The focused difference is that RRB avoids explicit parsing and avoids replacing the CLIP embedding. It instead learns a parser-free relational residual whose magnitude and selection are constrained by the original CLIP manifold.

The hard-bottleneck negative result is part of the novelty argument: it shows that relation supervision can be too strong when routed through a replacement bottleneck. The residual interface is therefore not a cosmetic adapter choice; it is the mechanism that turns a diagnostic relation signal into a usable dual-encoder tradeoff.

This novelty argument is currently evidence-limited. R033/R040 are useful no-slot controls because they show that a simple residual path can be relation-strong but unsafe. They still do not prove semantic slot specialization. The paper should describe the slots as a lightweight architectural prior unless a slot-responsibility or slot-ablation analysis is added.

ROP-style orthogonal projection is not a viable headline novelty. The closest overlaps are null-space and principal-subspace PEFT methods such as PEGP, GNSP, KeepLoRA, and PSOFT. If used later, it should be cited as a borrowed guardrail regularizer and evaluated only as a preservation ablation.

## Claim-Driven Validation Sketch

### Claim 1: RRB improves the relation-vs-guardrail tradeoff.

- Minimal experiment: final Pareto table over `R003`, `R005`, hard token variants, `R019`, `R023`, `R024`, `R025`, `R027-R029`, `R031-R040`, and `R034`.
- Baselines / ablations: pretrained OpenCLIP, plain adapter, hard token bottleneck, full RRB seeds, no-targeted RRB, generic-hard-negative RRB, no-slot residual controls, anchor-20 preservation sweep, and Winoground external-validity check.
- Metric: relation accuracy, COCO I2T R@1, ImageNet top-1, residual relative magnitude, and `cos(z0, z)`.
- Expected evidence: RRB beats the plain adapter on relation accuracy while avoiding hard-bottleneck collapse. COCO/ImageNet drops are reported as bounded alignment tax.

### Claim 2: Structured relational counterfactuals matter.

- Minimal experiment: matched RRB training with structured relation counterfactuals versus generic hard negatives.
- Baselines / ablations: `R019`/`R025` style structured RRB versus `R024` generic-hard-negative RRB.
- Metric: relation-suite accuracy plus COCO/ImageNet guardrails.
- Expected evidence: generic hard negatives preserve broad behavior but do not reproduce relation gains.

### Claim 3: `L_targeted` is not a necessary mechanism claim.

- Minimal experiment: compare full RRB and no-targeted RRB under matched settings.
- Baselines / ablations: `R019`/`R025` full RRB versus `R023` no-targeted RRB.
- Metric: relation accuracy, COCO I2T R@1, ImageNet top-1, and optional slot diagnostics.
- Expected evidence: no-targeted RRB can achieve strong relation accuracy; the main paper should not depend on targeted-slot supervision.

## Experiment Handoff Inputs

- Must-prove claims: bounded residual adaptation improves the relation-vs-guardrail tradeoff; structured counterfactuals beat generic hard negatives; targeted-slot loss is optional under current evidence.
- Must-report limitations: COCO seed fragility, R028 alignment tax, negative Winoground text/group scores, and unproven semantic slot specialization.
- Must-run ablations already complete for the bounded writeup: consolidation table, structured versus generic hard negatives, no-targeted versus full RRB, hard bottleneck diagnostic, plain adapter baseline, no-slot residual controls, and anchor-20 preservation sweep.
- Critical datasets / metrics: ARO/SugarCrepe relation suite, COCO 1k retrieval, ImageNet 5k zero-shot, Winoground, residual drift diagnostics, and seed distributions.
- Highest-risk assumptions: COCO retrieval robustness, cross-benchmark relation generalization, edit quality, and whether the slot-bottlenecked residual interface has a measurable responsibility pattern.

## Compute & Timeline Estimate

- Estimated GPU-hours: no more training is required for a bounded workshop/class-project/thesis writeup. R027-R040 and R034 already cover the immediate seed, no-slot, stricter-anchor, and Winoground follow-ups.
- Data / annotation cost: a 200-example manual edit audit remains useful only if the data pipeline is foregrounded. Slot-responsibility analysis can start from existing checkpoints before launching new training.
- Timeline: one day for final tables/figures and reviewer-facing writeup cleanup. Top-venue work would require additional external retrieval, slot analysis, and possibly a cited ROP-style guardrail ablation.
