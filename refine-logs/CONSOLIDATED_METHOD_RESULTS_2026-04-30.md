# Consolidated Method and Results: Relational Residual Bottlenecking for CLIP

Date: 2026-04-30

This is the canonical project-facing method and results report. It complements `refine-logs/FINAL_PROPOSAL.md` and replaces earlier refinement, review, novelty, narrative, and dated result-analysis files.

## Executive Summary

The project started from the problem that CLIP-style dual encoders often match images and captions by object co-occurrence while missing relation, role, and binding structure. The original RelBottleneck-CLIP idea inserted a parser-free token bottleneck into OpenCLIP and trained it with structured counterfactual captions. The hard bottleneck proved that relation signal is reachable, but it catastrophically damaged broad CLIP behavior.

The current method is **Relational Residual Bottlenecking (RRB)**. RRB keeps the frozen OpenCLIP embedding `z0` present and adds a bounded parser-free relational correction:

```text
z = normalize(z0 + alpha * delta_rel)
```

The residual `delta_rel` is produced by parser-free slot states over CLIP tokens. The core training recipe uses structured counterfactual captions, anchor-batch feature mimicry, and residual regularization. The previous targeted slot loss is now diagnostic only; current results do not support it as a necessary mechanism. The phrase "relational slots" should be treated as an architectural shorthand, not as a proven interpretability claim.

Final claim boundary after R034-R040:

- Supported: RRB gives a much better relation-vs-guardrail tradeoff than hard bottlenecking and plain adapter fine-tuning.
- Supported: structured relational counterfactuals matter; generic hard negatives do not reproduce the relation gains.
- Supported with caveat: full RRB relation gains replicate directionally over three seeds, but COCO retrieval is seed-fragile under the original R019 setting.
- Supported with caveat: stronger anchoring recovers guardrails for selected no-targeted RRB checkpoints and lowers residual drift, but it does not eliminate COCO seed fragility.
- Supported with caveat: no-slot residual controls are relation-strong but guardrail-fragile, even after tighter residual constraints; this supports the slot-bottlenecked residual interface as a safety-relevant prior, not as proven semantic slot specialization.
- Not supported: the current `L_targeted` loss is necessary for relation accuracy.
- Not supported: broad retrieval is robustly preserved without qualification; COCO retrieval remains the limiting guardrail.
- Not supported: R028 improves Winoground binding/role generality over OpenCLIP; R034 is flat-to-negative except for a small image-score gain.

External review update on 2026-04-30: a new hard review scored the package `5/10` for a top ML venue and returned `NOT READY`. The actionable refinement is to reframe and analyze, not to pivot the headline method. The reviewer-suggested Relational Orthogonal Projection idea was novelty-checked at `3.5/10`; it is too close to recent null-space/principal-subspace PEFT for VLMs to be the main method. After reframing, Round 2 scored `5.5/10`: ready for a bounded workshop/thesis analysis writeup, still not ready for a top-venue main track.

## Method

### Method History

The initial proposal was **RelBottleneck-CLIP**: a hard parser-free latent bottleneck inserted between CLIP tokens and the final embedding, trained with structured counterfactual captions. The first concrete design used latent slots, pairwise relational interactions, `L_intervene`, `L_targeted`, and a light sparsity term.

The reviewed/refined design made three simplifying changes before implementation:

- replace explicit GNN-style edge passing with one slot self-attention block,
- define `L_targeted` through soft slot-token responsibility over edited spans,
- keep LLM/VLM counterfactual generation and filtering offline so inference remains a normal dual encoder.

The hard-replacement direction was then superseded by **RRB** after experiments showed that relation accuracy could improve while broad CLIP retrieval and zero-shot behavior collapsed. RRB is the current method because it preserves the frozen CLIP path and learns only a bounded relational correction.

### Problem

Improve a CLIP-style vision-language encoder so its embeddings encode compositional relational structure, not just coarse object similarity.

### Non-Goals

This project does not build a multimodal LLM, chain-of-thought reasoner, scene-graph parser, detector-heavy pipeline, or task-specific VQA architecture. The method should remain a lightweight dual-encoder update.

### Architecture

RRB attaches one residual relational module to each CLIP branch:

```text
Image I -> frozen OpenCLIP image embedding z0_v
        -> visual tokens Pv -> K visual slots Sv -> relational residual delta_v

Text  T -> frozen OpenCLIP text embedding z0_t
        -> text tokens Pt -> K text slots St -> relational residual delta_t

Final embeddings:
z_v = normalize(z0_v + alpha_v * delta_v)
z_t = normalize(z0_t + alpha_t * delta_t)
```

Default design:

- OpenCLIP ViT-B/16 backbone.
- One slot self-attention block.
- Parser-free latent slot induction.
- Bounded fixed or vector residual gate.
- `z0` remains the dominant pretrained path.

### Training Objective

The core objective is:

```text
L = lambda_intervene * L_intervene
  + lambda_anchor * L_anchor
  + lambda_delta * L_delta
```

Terms:

- `L_intervene`: margin ranking against structured counterfactual captions.
- `L_anchor`: feature mimicry against frozen OpenCLIP embeddings on anchor pairs.
- `L_delta`: residual norm/gate regularization.

`L_global` can be used as a secondary clean-pair alignment term when a run explicitly tests it. `L_targeted` should be treated as optional or diagnostic. It should not be a headline mechanism claim unless redesigned and revalidated.

### Novelty Positioning

The novelty is not residual adaptation or feature mimicry alone. Those are crowded. The defensible contribution is the combination of:

1. a parser-free slot-bottlenecked residual interface,
2. bounded residual correction rather than exclusive replacement of CLIP embeddings,
3. structured counterfactual supervision on relation/role changes,
4. diagnostic evidence showing why hard bottlenecking is unsafe.

Closest overlaps:

- RACA-CLIP: relation-aware CLIP alignment with explicit/structured relation supervision.
- FSC-CLIP and preservation-oriented compositional fine-tuning: close on preservation losses.
- C2LIP / concept-centric CLIP: close on pooling and compositionality diagnosis.
- Residual CLIP adapters: close on residual mechanics.
- Null-space / principal-subspace PEFT methods such as PEGP, GNSP, KeepLoRA, and PSOFT: close on orthogonal projection and preservation mechanisms.

Positioning rule: lead with structured-counterfactual residual adaptation and the hard-bottleneck diagnostic story, not generic residual-adapter or newly invented orthogonal-projection language. If ROP-style projection is ever tested, cite it as a borrowed guardrail ablation.

## Experiment Roots

| Stage | Remote Root | Purpose |
|---|---|---|
| Updated hard token backend | `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/updated_end2end_20260428_115746` | Show token relation signal and guardrail collapse |
| RRB M3a sweep | `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_20260428_131556` | Test residual guardrail rescue R019-R022 |
| RRB M3b controls | `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m3b_20260428_201251` | Test no-targeted, generic-negative, and seed-2 replicate controls |
| Active final checks | `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_active_20260429_155721` | Run R027-R030: full seed 3, stronger-anchor no-targeted RRB, matched full stronger-anchor control, and final diagnostics |
| M7 review follow-up GPU 0 | `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m7_gpu0_20260429_164404` | Run R031 and R033: selected-checkpoint seed replication and no-slot residual control |
| M7 review follow-up GPU 1 | `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m7_gpu1_20260429_164404` | Run R032: selected-checkpoint seed replication |
| M8 stricter preservation GPU 0 | `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m8_gpu0_20260429_211743` | Run R035, R037, R039: no-targeted RRB with `lambda_anchor=20` |
| M8 stricter preservation GPU 1 | `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m8_gpu1_20260429_211743` | Run R036, R038, R040: anchor-20 seeds plus constrained no-slot control |
| Winoground R034 | `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/winoground_r034_20260429_215139` | External binding/role validity check for R003, R005, and R028 |

The staged public evaluation uses ARO/SugarCrepe relation suites, COCO retrieval with 1,000 images, and ImageNet zero-shot with 5,000 examples. These are not substitutes for Winoground or Flickr30k.

## Superseded Adapter-Backend Evidence

Before the token/residual implementation, a frozen-OpenCLIP embedding-adapter backend was used as a diagnostic pilot. It should not be treated as paper-grade RelBottleneck evidence, but it explains the later pivot.

Expanded adapter root:

`/shared/nas/data/m1/zixuans8/relbottleneck/outputs/paper_grade_20260424_195455`

| Run | System | Relation | COCO I2T R@1 | ImageNet top-1 | Interpretation |
|---|---|---:|---:|---:|---|
| R003 | pretrained OpenCLIP | 0.6564 | 0.7990 | 0.6670 | baseline |
| R005 | plain fine-tune | 0.6956 | 0.7790 | 0.6438 | safe adapter baseline |
| R006 | bottleneck-only | 0.6963 | 0.7860 | 0.6438 | tied plain fine-tune |
| R007 | intervention-only | 0.7807 | 0.2600 | 0.4156 | relation gain, guardrail failure |
| R008 | full K=4 | 0.7890 | 0.6050 | 0.6200 | relation gain, retrieval tax |
| R009 | full K=8 | 0.7872 | 0.4230 | 0.5250 | relation gain, large retrieval tax |
| R010 | full K=16 | 0.7807 | 0.2600 | 0.4156 | relation gain, guardrail failure |
| R011-R013 | full seeds | 0.7911 mean | 0.4980 mean | 0.5473 mean | reproducible relation gain, weak guardrails |
| R014 | no targeted loss | 0.8050 | 0.6700 | 0.6318 | strongest relation result; falsifies targeted-loss necessity in this backend |
| R015 | generic hard negatives | 0.6901 | 0.7830 | 0.6680 | guardrails preserved, relation gain absent |
| R016 | unfiltered edits | 0.8016 | 0.5670 | 0.5728 | relation gain, retrieval tax |
| R017 | overbuilt variant | 0.7601 | 0.1120 | 0.0542 | rejected; guardrail collapse |

Takeaway: structured-margin adapter training had real relation signal, but it did not isolate the intended `L_targeted` mechanism and it degraded broad alignment. This directly motivated bounded residual adaptation and the current RRB claim boundary.

## Main Results

Pilot gate: relation accuracy greater than R005, COCO I2T R@1 at least `0.70`, and ImageNet top-1 at least `0.55`.

| Run | System | Relation | d vs R005 | COCO I2T R@1 | ImageNet top-1 | Residual mag. | z0 cosine mean | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R003 | OpenCLIP | 0.6564 | -0.0392 | 0.7990 | 0.6670 | - | - | baseline |
| R005 | plain adapter | 0.6956 | +0.0000 | 0.7790 | 0.6438 | - | - | safe baseline |
| R006 | hard token bottleneck only | 0.6674 | -0.0282 | 0.4210 | 0.0644 | - | - | collapse |
| R014 | hard token no targeted | 0.7332 | +0.0376 | 0.4110 | 0.0602 | - | - | collapse |
| R011 | hard token targeted | 0.7488 | +0.0532 | 0.3920 | 0.0660 | - | - | collapse |
| R012 | hard token targeted + distill 0.5 | 0.7483 | +0.0527 | 0.3740 | 0.0912 | - | - | collapse |
| R019 | RRB full seed 1 | 0.7697 | +0.0741 | 0.7190 | 0.6152 | 0.1328 | 0.9900 | pass |
| R020 | RRB alpha 0.10 | 0.7700 | +0.0744 | 0.4850 | 0.3922 | 0.2139 | 0.9730 | fail |
| R021 | RRB alpha 0.10 anchor 10 | 0.7734 | +0.0778 | 0.5300 | 0.4822 | 0.1803 | 0.9811 | fail |
| R022 | RRB learned gate | 0.7820 | +0.0864 | 0.6790 | 0.6086 | 0.1255 | 0.9912 | COCO fail |
| R023 | RRB no targeted | 0.7922 | +0.0966 | 0.6820 | 0.5982 | 0.0886 | 0.9953 | COCO fail |
| R024 | RRB generic hard negatives | 0.6736 | -0.0220 | 0.7990 | 0.6676 | 0.0484 | 0.9988 | relation fail |
| R025 | RRB full seed 2 | 0.7809 | +0.0853 | 0.6980 | 0.6026 | 0.1020 | 0.9932 | COCO near miss |
| R027 | RRB full seed 3 | 0.7799 | +0.0843 | 0.6610 | 0.5516 | 0.1247 | 0.9911 | COCO fail |
| R028 | RRB no targeted, stronger anchor | 0.7734 | +0.0778 | 0.7340 | 0.6342 | 0.0804 | 0.9963 | pass; selected |
| R029 | RRB full, stronger anchor | 0.7731 | +0.0775 | 0.7280 | 0.6462 | 0.1073 | 0.9931 | pass; matched control |
| R031 | RRB no targeted, stronger anchor seed 2 | 0.7809 | +0.0853 | 0.7260 | 0.5942 | 0.0747 | 0.9967 | pass |
| R032 | RRB no targeted, stronger anchor seed 3 | 0.7804 | +0.0848 | 0.6730 | 0.5814 | 0.0846 | 0.9961 | COCO fail |
| R033 | no-slot residual adapter control | 0.7825 | +0.0869 | 0.3740 | 0.3308 | 0.2062 | 0.9754 | collapse |
| R035 | RRB no targeted, anchor 20 seed 1 | 0.7802 | +0.0846 | 0.7130 | 0.6128 | 0.0623 | 0.9978 | pass |
| R036 | RRB no targeted, anchor 20 seed 2 | 0.7799 | +0.0843 | 0.6820 | 0.5764 | 0.0677 | 0.9974 | COCO fail |
| R037 | RRB no targeted, anchor 20 seed 3 | 0.7815 | +0.0859 | 0.7140 | 0.6030 | 0.0651 | 0.9976 | pass |
| R038 | RRB no targeted, anchor 20 seed 4 | 0.7734 | +0.0778 | 0.7430 | 0.6694 | 0.0674 | 0.9974 | pass |
| R039 | RRB no targeted, anchor 20 seed 5 | 0.7794 | +0.0838 | 0.6940 | 0.6196 | 0.0707 | 0.9972 | COCO near miss |
| R040 | constrained no-slot residual control | 0.7773 | +0.0817 | 0.4820 | 0.4798 | 0.0969 | 0.9942 | guardrail-fragile |

## Per-Source Results for Final RRB Controls

| Source | R023 no targeted | R024 generic hard negatives | R025 full seed 2 | R027 full seed 3 | R028 no targeted stronger anchor | R029 full stronger anchor |
|---|---:|---:|---:|---:|---:|---:|
| aro_coco_order | 0.8652 | 0.3789 | 0.8457 | 0.8691 | 0.8262 | 0.8477 |
| aro_visual_attribution | 0.7441 | 0.6543 | 0.7598 | 0.7148 | 0.7246 | 0.7031 |
| aro_visual_relation | 0.5469 | 0.4980 | 0.5449 | 0.5449 | 0.5352 | 0.5469 |
| sugarcrepe_replace_att | 0.8906 | 0.8613 | 0.8613 | 0.8809 | 0.8809 | 0.8789 |
| sugarcrepe_replace_obj | 0.9531 | 0.9551 | 0.9609 | 0.9434 | 0.9492 | 0.9570 |
| sugarcrepe_replace_rel | 0.8281 | 0.6992 | 0.7891 | 0.7852 | 0.8086 | 0.7754 |
| sugarcrepe_swap_att | 0.7539 | 0.6758 | 0.7324 | 0.7363 | 0.7227 | 0.7402 |
| sugarcrepe_swap_obj | 0.7154 | 0.6585 | 0.7236 | 0.7480 | 0.7033 | 0.6951 |

## Per-Source Results for Post-Review Runs

| Source | R028 | R031 | R032 | R033 | R035 | R036 | R037 | R038 | R039 | R040 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aro_coco_order | 0.8262 | 0.8320 | 0.8340 | 0.9258 | 0.8555 | 0.8301 | 0.8281 | 0.8047 | 0.8320 | 0.9023 |
| aro_visual_attribution | 0.7246 | 0.7617 | 0.7285 | 0.7500 | 0.7188 | 0.7461 | 0.7324 | 0.7422 | 0.7598 | 0.7422 |
| aro_visual_relation | 0.5352 | 0.5625 | 0.5508 | 0.5566 | 0.5273 | 0.5391 | 0.5547 | 0.5352 | 0.5352 | 0.5352 |
| sugarcrepe_replace_att | 0.8809 | 0.8789 | 0.8730 | 0.8398 | 0.8867 | 0.8809 | 0.8770 | 0.8770 | 0.8613 | 0.8535 |
| sugarcrepe_replace_obj | 0.9492 | 0.9590 | 0.9590 | 0.9258 | 0.9609 | 0.9648 | 0.9590 | 0.9512 | 0.9570 | 0.9375 |
| sugarcrepe_replace_rel | 0.8086 | 0.7910 | 0.7930 | 0.7988 | 0.8145 | 0.7930 | 0.8066 | 0.8105 | 0.8047 | 0.7891 |
| sugarcrepe_swap_att | 0.7227 | 0.7109 | 0.7246 | 0.7285 | 0.7422 | 0.7168 | 0.7227 | 0.7227 | 0.7266 | 0.7285 |
| sugarcrepe_swap_obj | 0.7033 | 0.7195 | 0.7805 | 0.6829 | 0.6870 | 0.7561 | 0.7602 | 0.7114 | 0.7358 | 0.6789 |

## Three-Seed Full RRB Check

This is a three-seed check over the original full-RRB setting: R019 seed 1, R025 seed 2, and R027 seed 3. It supports directionally stable relation gains, but not robust COCO gate passing.

| Metric | Mean | Sample std | Values |
|---|---:|---:|---|
| Relation | 0.7768 | 0.0062 | 0.7697, 0.7809, 0.7799 |
| COCO I2T R@1 | 0.6927 | 0.0294 | 0.7190, 0.6980, 0.6610 |
| ImageNet top-1 | 0.5898 | 0.0337 | 0.6152, 0.6026, 0.5516 |

## Stronger-Anchor Simplification Check

R028 tests the reviewer-facing simplification: remove `L_targeted`, keep structured counterfactual training, and strengthen the anchor. R029 is the matched full-RRB control under the same stronger-anchor setting.

| Run | Variant | lambda anchor | Anchor ratio | L_targeted | Relation | COCO I2T R@1 | ImageNet top-1 | Residual mag. | z0 cosine mean | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R023 | no targeted | 5 | 0.50 | 0 | 0.7922 | 0.6820 | 0.5982 | 0.0886 | 0.9953 | relation-strong, COCO fail |
| R028 | no targeted, stronger anchor | 10 | 0.75 | 0 | 0.7734 | 0.7340 | 0.6342 | 0.0804 | 0.9963 | selected; best simplicity/guardrail tradeoff |
| R029 | full, stronger anchor | 10 | 0.75 | 1 | 0.7731 | 0.7280 | 0.6462 | 0.1073 | 0.9931 | matched control; passes but less simple |

### M7 Review-Driven Follow-Up

After the hard review loop, R031-R033 were run to test top-venue blockers: selected-checkpoint seed stability and a no-slot residual control.

| Metric | Mean | Sample std | Values |
|---|---:|---:|---|
| Stronger-anchor no-targeted relation | 0.7782 | 0.0042 | R028 0.7734, R031 0.7809, R032 0.7804 |
| Stronger-anchor no-targeted COCO I2T R@1 | 0.7110 | 0.0332 | R028 0.7340, R031 0.7260, R032 0.6730 |
| Stronger-anchor no-targeted ImageNet top-1 | 0.6033 | 0.0275 | R028 0.6342, R031 0.5942, R032 0.5814 |

Interpretation: relation gains reproduce for the simplified checkpoint, but COCO is still seed-fragile. R033 shows that a no-slot residual control can reach relation `0.7825` but collapses COCO to `0.3740` and ImageNet to `0.3308`, so a simple residual path is not a safe substitute for the slot-bottlenecked residual interface.

### M8 Stricter Preservation and Constrained No-Slot Control

R035-R039 test whether increasing the anchor to `lambda_anchor=20` stabilizes the selected no-targeted RRB setting. R040 constrains the no-slot residual control with the same stronger anchor and higher delta penalty.

| Metric | Mean | Sample std | Values |
|---|---:|---:|---|
| Anchor-20 no-targeted relation | 0.7789 | 0.0032 | R035 0.7802, R036 0.7799, R037 0.7815, R038 0.7734, R039 0.7794 |
| Anchor-20 no-targeted COCO I2T R@1 | 0.7092 | 0.0232 | R035 0.7130, R036 0.6820, R037 0.7140, R038 0.7430, R039 0.6940 |
| Anchor-20 no-targeted ImageNet top-1 | 0.6162 | 0.0340 | R035 0.6128, R036 0.5764, R037 0.6030, R038 0.6694, R039 0.6196 |
| Anchor-20 residual magnitude | 0.0666 | 0.0031 | R035 0.0623, R036 0.0677, R037 0.0651, R038 0.0674, R039 0.0707 |
| Anchor-20 z0 cosine | 0.9975 | 0.0002 | R035 0.9978, R036 0.9974, R037 0.9976, R038 0.9974, R039 0.9972 |

Interpretation: anchor 20 reduces residual drift relative to R028/R031/R032 and improves mean ImageNet, but it does not improve mean COCO or solve seed fragility. Three of five seeds clear COCO I2T R@1 `>= 0.70`; R036 fails and R039 is a near miss.

R040 reduces no-slot residual magnitude from R033 `0.2062` to `0.0969`, but it still damages guardrails: COCO I2T R@1 `0.4820` and ImageNet top-1 `0.4798`. This strengthens the safety-prior argument for the slot-bottlenecked residual interface, while still not proving semantic slot specialization.

### R034 Winoground External Validity

Winoground access was granted and R034 evaluated the 400-example test split.

| System | Text score | Image score | Group score | Delta vs OpenCLIP |
|---|---:|---:|---:|---|
| R003 OpenCLIP | 0.2775 | 0.1075 | 0.0825 | baseline |
| R005 plain adapter | 0.2700 | 0.0850 | 0.0650 | text -0.0075, image -0.0225, group -0.0175 |
| R028 selected RRB | 0.2400 | 0.1100 | 0.0700 | text -0.0375, image +0.0025, group -0.0125 |

Interpretation: R028 does not provide positive Winoground external-validity evidence over OpenCLIP. It slightly improves image score but loses text and group score, so broad binding/role generalization remains unsupported.

## R030 Diagnostics for Selected Checkpoint

R030 exports diagnostics for R028. The diagnostic files are:

- local report: `refine-logs/R030_DIAGNOSTICS.md`
- local JSON summary: `refine-logs/R030_diagnostics_summary.json`
- remote root: `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_active_20260429_155721/R030`

| Diagnostic | Value |
|---|---:|
| Relation accuracy | 0.7734 |
| COCO I2T R@1 | 0.7340 |
| ImageNet top-1 | 0.6342 |
| Residual relative magnitude mean | 0.0804 |
| z0 cosine mean | 0.9963 |

High-count failure concentrations:

| Source | Edit type | Failures | Total | Failure rate | Mean margin |
|---|---|---:|---:|---:|---:|
| aro_visual_relation | to the right of | 89 | 164 | 0.5427 | -0.0006 |
| aro_visual_relation | to the left of | 87 | 163 | 0.5337 | -0.0009 |
| sugarcrepe_swap_obj | swap_obj | 73 | 246 | 0.2967 | 0.0139 |
| sugarcrepe_swap_att | swap_att | 142 | 512 | 0.2773 | 0.0144 |
| aro_visual_attribution | aro_visual_attribution | 141 | 512 | 0.2754 | 0.0138 |
| sugarcrepe_replace_rel | replace_rel | 98 | 512 | 0.1914 | 0.0237 |
| aro_coco_order | word_order | 89 | 512 | 0.1738 | 0.0134 |
| sugarcrepe_replace_att | replace_att | 61 | 512 | 0.1191 | 0.0374 |

## Findings

1. **Hard token bottlenecking is diagnostic, not paper-facing.**  
   Observation: hard token runs reach relation accuracy near `0.7488`, but COCO I2T and ImageNet collapse.  
   Interpretation: replacing the frozen CLIP path forces relation sensitivity at the cost of broad alignment.  
   Implication: the paper should use hard bottlenecking as a motivation/diagnostic baseline, not as the proposed method.  
   Next step: no additional hard-bottleneck runs unless the appendix needs a focused failure visualization.

2. **R019 is no longer the final selected checkpoint, but it remains the clean original full-RRB point.**  
   Observation: R019 improves relation from R005 `0.6956` to `0.7697` while passing COCO and ImageNet gates.  
   Interpretation: bounded residual adaptation can recover relation sensitivity without immediate guardrail collapse.  
   Implication: R019 is the best single result for the original full-RRB recipe.  
   Next step: compare against R028 as the simplified final checkpoint rather than rerunning R019.

3. **Full RRB relation gains reproduce, but COCO is seed-fragile.**  
   Observation: R019/R025/R027 average relation `0.7768 +/- 0.0062`, but COCO averages `0.6927 +/- 0.0294`, with R025 near-missing the gate and R027 failing it at `0.6610`.  
   Interpretation: the original full-RRB objective is consistently relation-positive, but the guardrail constraint is not stable enough for a preservation claim.  
   Implication: claim directionally stable relation gains, not robust broad retrieval preservation.  
   Next step: if reviewers require more seed evidence, replicate the stronger-anchor setting rather than the original full-RRB setting.

4. **`L_targeted` is not necessary and should be removed from the core method claim.**  
   Observation: R023 removes `L_targeted` and reaches relation `0.7922`, and R028 keeps `L_targeted=0` while passing all gates with relation `0.7734`, COCO `0.7340`, and ImageNet `0.6342`.  
   Interpretation: structured counterfactual supervision plus residual anchoring is sufficient for the supported effect.  
   Implication: the paper can simplify the method and present `L_targeted` as an exploratory diagnostic that was not retained.  
   Next step: only revisit targeted losses if the paper later makes a slot-localization claim.

5. **Generic hard negatives are insufficient.**  
   Observation: R024 preserves COCO `0.7990` and ImageNet `0.6676`, but relation is `0.6736`, below R005.  
   Interpretation: guardrail-safe generic contrastive pressure does not reproduce the relational gains.  
   Implication: structured relation counterfactuals remain the central supervision contribution.  
   Next step: no more generic-hard-negative controls are needed for the main claim.

6. **Stronger anchoring turns the no-targeted variant into the best paper-facing checkpoint.**  
   Observation: R028 passes all gates with relation `0.7734`, COCO `0.7340`, ImageNet `0.6342`, residual magnitude `0.0804`, and z0 cosine `0.9963`. R029 also passes but is less simple and slightly worse on COCO.  
   Interpretation: stronger anchoring and higher anchor ratio reduce harmful residual drift while preserving most relation gain.  
   Implication: R028 should be the final selected checkpoint unless a later writing constraint demands a full-RRB control as the headline.  
   Next step: use R028 in tables and figures; keep R029 as a matched control.

7. **Residual magnitude alone is not enough for model selection.**  
   Observation: R023 has low residual magnitude and high z0 cosine but fails COCO; R028 improves COCO with a similar small residual; R029 has higher residual magnitude and slightly lower COCO than R028.  
   Interpretation: both residual size and residual direction/objective matter.  
   Implication: report residual diagnostics as explanatory context, not as a standalone safety predictor.  
   Next step: use R030 diagnostics for failure transparency, not for a mechanistic proof.

8. **R030 localizes remaining failures.**  
   Observation: the largest high-count failure groups are ARO spatial predicates (`to the right of`, `to the left of`) and SugarCrepe swap/attribute cases.  
   Interpretation: RRB improves relation sensitivity but still struggles with fine spatial symmetry and attribute/object swaps.  
   Implication: the paper should be explicit that RRB is not a complete spatial reasoning solution.  
   Next step: include a compact failure table or qualitative panel if the paper needs an error-analysis figure.

9. **Anchor 20 reduces drift but does not solve COCO fragility.**  
   Observation: R035-R039 average relation `0.7789 +/- 0.0032` and residual magnitude `0.0666 +/- 0.0031`, but COCO averages only `0.7092 +/- 0.0232` with two of five seeds below `0.70`.  
   Interpretation: stronger anchoring compresses residual scale without fully controlling the retrieval-sensitive residual direction.  
   Implication: the paper can report a bounded alignment tax, not robust preservation.  
   Next step: only run another preservation sweep if the target venue requires a tighter COCO guarantee.

10. **Constrained no-slot residual control remains unsafe.**  
    Observation: R040 reduces no-slot residual magnitude from R033 `0.2062` to `0.0969`, but COCO remains `0.4820` and ImageNet top-1 remains `0.4798`.  
    Interpretation: matching the drift budget helps, but the no-slot path still damages broad CLIP behavior far more than slot RRB.  
    Implication: the slot-bottlenecked residual interface is safety-relevant; do not overstate this as semantic slot specialization.  
    Next step: a slot ablation or responsibility analysis is needed before making an interpretability claim.

11. **Winoground does not support broad binding generalization.**  
    Observation: R028 scores text/image/group `0.2400/0.1100/0.0700` versus OpenCLIP `0.2775/0.1075/0.0825`.  
    Interpretation: the ARO/SugarCrepe gains do not transfer cleanly to this external binding benchmark.  
    Implication: keep the contribution scoped to staged relation/counterfactual suites with guardrail diagnostics.  
    Next step: use Winoground as a limitation, not as supporting evidence; Flickr30k-style retrieval would be the next external check.

12. **Relational Orthogonal Projection is not a headline pivot.**  
    Observation: the 2026-04-30 reviewer suggested constraining `delta_rel` to be orthogonal to principal CLIP components. A novelty check found the core mechanism low-novelty because recent VLM/PEFT work already uses null-space or principal-subspace projection to protect pre-trained knowledge.  
    Interpretation: the idea may be useful as a stricter preservation regularizer, but it does not solve the paper's novelty problem by itself.  
    Implication: do not rename the project around ROP or present it as a new method contribution.  
    Next step: only implement ROP as a cited optional guardrail ablation if pursuing top-venue robustness.

## Current Claim Boundary

Allowed claims:

- RRB improves the relation-vs-guardrail tradeoff relative to hard bottlenecking and plain adapter fine-tuning.
- Structured relational counterfactuals matter; generic hard negatives do not reproduce the relation gains.
- Full RRB relation gains are directionally reproducible over three seeds, but guardrails are seed-fragile under the original setting.
- Stronger anchoring makes the no-targeted RRB variant the current best simplicity/guardrail tradeoff, but still not robustly COCO-safe across seeds.
- Anchor-20 no-targeted RRB lowers residual drift and keeps relation stable, but does not eliminate the COCO alignment tax.
- No-slot residual controls are relation-strong but guardrail-fragile, supporting the slot-bottlenecked residual interface as a safety-relevant architectural prior.
- COCO retrieval is the limiting guardrail and should be reported as an explicit alignment tax.
- Winoground R034 is a negative/limitation result for broad binding and role generalization.
- ROP-style orthogonal projection may be explored as a prior-art-inspired preservation ablation, not as a novel method direction.

Disallowed claims:

- Do not claim `L_targeted` is necessary for relation accuracy.
- Do not claim robust gate passing across seeds.
- Do not claim broad retrieval is preserved without qualification.
- Do not claim residual magnitude alone predicts safety.
- Do not claim interpretable slot specialization without a direct slot analysis.
- Do not claim Winoground improvement over OpenCLIP.
- Do not claim Relational Orthogonal Projection as original without positioning it against PEGP, GNSP, KeepLoRA, PSOFT, and related null-space PEFT.

## Suggested Next Experiments

No additional core experiment is required before a bounded workshop/class-project/thesis writeup. The next work should be driven by venue ambition:

1. **Writeup path:** use R028 as the selected checkpoint, report R035-R039 as stronger-anchor robustness evidence, and present R034/R040 as main-table limitations.
2. **Top-venue robustness path:** add Flickr30k-style retrieval and a lower-alpha or explicitly cited residual-direction/orthogonal-projection ablation only if the target requires stronger guardrails.
3. **External-validity path:** add another binding benchmark before making any broader generality claim.
4. **Mechanism path:** add slot ablation or slot-responsibility analysis before claiming interpretable relation-specialized slots.

## Top-Venue Readiness Addendum

A hard external review on 2026-04-29 scored the package `5/10` for a top ML venue. After R031-R033, the review status improved to `6.5/10` with an `ALMOST` top-venue verdict because R033 showed that a no-slot residual control collapses guardrails. R034 and R035-R040 then closed the immediate planned follow-ups, but they did not remove the main blockers.

- The current package is suitable as a solid workshop/class-project/thesis result if written honestly.
- It is not yet a top-venue main-track result because COCO guardrails remain seed-fragile and Winoground is not positive.
- R028 still pays an alignment tax: COCO I2T R@1 drops from R003 `0.7990` to `0.7340`, and from R005 `0.7790` to `0.7340`.
- R035-R039 show that `lambda_anchor=20` lowers residual drift, but the COCO pass rate is still only `3/5`.
- R034 closes the missing Winoground access blocker as a negative result: R028 does not beat OpenCLIP on text or group score.
- R033/R040 reduce the simple-residual ambiguity but still do not prove interpretable slot specialization.
- The phrase "relational slots" should be treated as an architectural description, not a proven mechanistic explanation.
- The 2026-04-30 Round 1 review returned `5/10` and `NOT READY`; it confirms the project is a bounded workshop/class-project/thesis result unless the external-validity and mechanism blockers are addressed.
- Round 2 after reframing returned `5.5/10` and explicitly recommended: stop and write.
- The reviewer-suggested ROP pivot was checked and should remain future work or an ablation because the mechanism overlaps strongly with null-space/principal-subspace PEFT literature.
- Do not put "slot" in the title or abstract unless a slot-responsibility analysis is added; use bottlenecked residual adaptation language instead.
- R041-R045 were run after this recommendation. They strengthen the same conclusion rather than changing it: slot responsibility was diffuse for R028/R035/R038, the R043 ROP pilot failed the relation gate (`0.7441` relation with COCO `0.7360` and ImageNet `0.6566`), R044 was correctly skipped, and Flickr30k shows an external retrieval tax for RRB checkpoints.

## Consolidated Experiment Plan and Tracker Archive

This section absorbs the useful information from the previous `EXPERIMENT_PLAN.md` and `EXPERIMENT_TRACKER.md`. Those files were rewritten after consolidation; the old milestone narrative is no longer the active execution plan.

### Archived Planning Decisions

- The original M0-M2 sanity, baseline, and capacity stages were useful for building the scaffold, but they are no longer the active paper path.
- The hard token bottleneck stage is retained only as diagnostic evidence. It showed relation signal but failed the guardrails.
- The RRB M3a/M3b, active R027-R030, M7, M8, and Winoground stages are the current evidence base:
  - R019 is the cleanest guarded point.
  - R025 is the relation-stable but COCO-fragile replicate.
  - R027 is the third full-RRB seed; it confirms relation gain and COCO fragility.
  - R023 is the no-targeted anti-claim control.
  - R024 is the generic-hard-negative anti-claim control.
  - R028 is the selected no-targeted stronger-anchor checkpoint.
  - R029 is the matched stronger-anchor full-RRB control.
  - R030 is the final diagnostics export for R028.
  - R031-R032 replicate the selected no-targeted stronger-anchor setting and show relation stability with COCO fragility.
  - R033/R040 are no-slot residual controls; both remain guardrail-fragile.
  - R034 is a negative Winoground external-validity result.
  - R035-R039 are the anchor-20 stricter-preservation sweep; relation is stable, but COCO pass rate is still `3/5`.
- The old plan's frontier/data-pipeline and overbuilt-variant blocks are appendix-only. They should not delay the core paper table.

### Archived Run Accounting

| Runs | Status After Consolidation | Keep / Cut Decision |
|---|---|---|
| R001-R002 | edit-audit and filter-audit scaffolding | Keep only if the paper foregrounds the data pipeline; otherwise appendix/cut |
| R003, R005 | OpenCLIP and plain-adapter baselines | Keep in every main table |
| R006, R011, R012, R014 | hard token bottleneck diagnostics | Keep as evidence that replacement bottlenecking is unsafe |
| R004, R018 | earlier planning or incomplete runs | Cut from active plan unless a specific appendix need reappears |
| R007-R010, R013, R015-R017 | completed superseded adapter runs | Keep only as historical/diagnostic evidence; not paper-facing RRB evidence |
| R019-R025 | current RRB evidence base | Keep as the core result set |
| R026 | consolidation | Keep as this consolidated report |
| R027 | full-RRB seed 3 | Keep in seed-stability summary |
| R028 | no-targeted stronger-anchor RRB | Keep as selected final checkpoint |
| R029 | matched full stronger-anchor control | Keep as control for R028 |
| R030 | diagnostics for R028 | Keep as error-analysis source |
| R031-R032 | selected-checkpoint seed replications | Keep in stronger-anchor seed summary |
| R033 | no-slot residual adapter control | Keep as safety-prior control |
| R034 | Winoground external-validity evaluation | Keep as negative external-validity result |
| R035-R039 | anchor-20 no-targeted sweep | Keep as stricter-preservation evidence and limitation |
| R040 | constrained no-slot residual control | Keep as constrained safety-prior control |
| R041-R045 | slot probe, ROP pilot, Flickr30k retrieval | Keep as negative top-venue follow-up evidence; do not use to expand claims |

### Completed Active Planning Scope

The active plan started from the completed evidence rather than replaying the whole project. Its priorities are now complete:

1. consolidate the existing R003/R005/R011/R014/R019-R025 evidence into final paper tables,
2. run the minimum additional RRB checks needed to address COCO fragility and seed stability,
3. run review-driven external-validity and no-slot controls,
4. export diagnostics for figures after the final checkpoint choice is locked.

## Final Paper Stance

The safest story is:

> RRB is a parser-free structured-counterfactual residual adaptation method that improves CLIP relation sensitivity while bounding, but not eliminating, retrieval and zero-shot degradation. The current evidence supports structured relational residual adaptation, not targeted-slot-supervision necessity.

The final selected checkpoint is still R028, the no-targeted stronger-anchor RRB variant. It gives the cleanest single-run tradeoff: relation `0.7734` versus R005 `0.6956`, COCO I2T R@1 `0.7340`, ImageNet top-1 `0.6342`, residual magnitude `0.0804`, and z0 cosine `0.9963`. R031/R032 and R035-R039 show that relation gains are stable, but COCO remains seed-fragile. R034 shows no broad Winoground generalization. R041-R045 add mechanism and external-retrieval negatives, not a new positive route: slot responsibility remains unproven, ROP misses the relation gate, and Flickr30k I2T R@1 drops from OpenCLIP `0.8620` to R028 `0.8260`. The final writeup should be analytical and limitation-forward, not top-venue-ready robustness evidence. The external review loop's stopping recommendation is to write this as a characterization of the relation-vs-guardrail tradeoff rather than launch more low-return adapter sweeps.
