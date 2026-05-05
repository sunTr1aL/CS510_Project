# Experiment Results

**Date**: 2026-04-29  
**Last updated**: 2026-04-30  
**Canonical report**: `refine-logs/CONSOLIDATED_METHOD_RESULTS_2026-04-30.md`

This file is a compact result ledger. The consolidated report is the source of truth for full analysis, per-source tables, claim boundaries, and redundant-document cleanup.

## Raw Public-Suite Table

Pilot gate: relation accuracy greater than R005, COCO I2T R@1 at least `0.70`, and ImageNet top-1 at least `0.55`.

| Run | System | Relation Acc. | Delta vs R005 | COCO I2T R@1 | ImageNet Top-1 | Residual Mag. | z0 Cosine | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R003 | OpenCLIP | 0.6564 | -0.0392 | 0.7990 | 0.6670 | - | - | baseline |
| R005 | plain adapter | 0.6956 | +0.0000 | 0.7790 | 0.6438 | - | - | safe baseline |
| R006 | hard token bottleneck only | 0.6674 | -0.0282 | 0.4210 | 0.0644 | - | - | collapse |
| R011 | hard token targeted seed 1 | 0.7488 | +0.0532 | 0.3920 | 0.0660 | - | - | collapse |
| R012 | hard token targeted seed 2 | 0.7483 | +0.0527 | 0.3740 | 0.0912 | - | - | collapse |
| R014 | hard token no targeted | 0.7332 | +0.0376 | 0.4110 | 0.0602 | - | - | collapse |
| R019 | RRB full seed 1 | 0.7697 | +0.0741 | 0.7190 | 0.6152 | 0.1328 | 0.9900 | pass |
| R020 | RRB alpha 0.10 | 0.7700 | +0.0744 | 0.4850 | 0.3922 | 0.2139 | 0.9730 | fail |
| R021 | RRB alpha 0.10 anchor 10 | 0.7734 | +0.0778 | 0.5300 | 0.4822 | 0.1803 | 0.9811 | fail |
| R022 | RRB learned gate | 0.7820 | +0.0864 | 0.6790 | 0.6086 | 0.1255 | 0.9912 | COCO fail |
| R023 | RRB no targeted | 0.7922 | +0.0966 | 0.6820 | 0.5982 | 0.0886 | 0.9953 | COCO fail |
| R024 | RRB generic hard negatives | 0.6736 | -0.0220 | 0.7990 | 0.6676 | 0.0484 | 0.9988 | relation fail |
| R025 | RRB full seed 2 | 0.7809 | +0.0853 | 0.6980 | 0.6026 | 0.1020 | 0.9932 | COCO near miss |
| R027 | RRB full seed 3 | 0.7799 | +0.0843 | 0.6610 | 0.5516 | 0.1247 | 0.9911 | COCO fail |
| R028 | no-targeted stronger-anchor RRB | 0.7734 | +0.0778 | 0.7340 | 0.6342 | 0.0804 | 0.9963 | selected |
| R029 | matched full stronger-anchor RRB | 0.7731 | +0.0775 | 0.7280 | 0.6462 | 0.1073 | 0.9931 | pass |
| R031 | no-targeted stronger-anchor seed 2 | 0.7809 | +0.0853 | 0.7260 | 0.5942 | 0.0747 | 0.9967 | pass |
| R032 | no-targeted stronger-anchor seed 3 | 0.7804 | +0.0848 | 0.6730 | 0.5814 | 0.0846 | 0.9961 | COCO fail |
| R033 | no-slot residual control | 0.7825 | +0.0869 | 0.3740 | 0.3308 | 0.2062 | 0.9754 | collapse |
| R035 | no-targeted RRB anchor 20 seed 1 | 0.7802 | +0.0846 | 0.7130 | 0.6128 | 0.0623 | 0.9978 | pass |
| R036 | no-targeted RRB anchor 20 seed 2 | 0.7799 | +0.0843 | 0.6820 | 0.5764 | 0.0677 | 0.9974 | COCO fail |
| R037 | no-targeted RRB anchor 20 seed 3 | 0.7815 | +0.0859 | 0.7140 | 0.6030 | 0.0651 | 0.9976 | pass |
| R038 | no-targeted RRB anchor 20 seed 4 | 0.7734 | +0.0778 | 0.7430 | 0.6694 | 0.0674 | 0.9974 | pass |
| R039 | no-targeted RRB anchor 20 seed 5 | 0.7794 | +0.0838 | 0.6940 | 0.6196 | 0.0707 | 0.9972 | COCO near miss |
| R040 | constrained no-slot residual control | 0.7773 | +0.0817 | 0.4820 | 0.4798 | 0.0969 | 0.9942 | guardrail-fragile |

## Seed Statistics

| Setting | Metric | Mean | Sample std | Pass Rate |
|---|---|---:|---:|---:|
| Full RRB R019/R025/R027 | Relation | 0.7768 | 0.0062 | 3/3 relation-positive |
| Full RRB R019/R025/R027 | COCO I2T R@1 | 0.6927 | 0.0294 | 1/3 pass |
| No-targeted anchor 10 R028/R031/R032 | Relation | 0.7782 | 0.0042 | 3/3 relation-positive |
| No-targeted anchor 10 R028/R031/R032 | COCO I2T R@1 | 0.7110 | 0.0332 | 2/3 pass |
| No-targeted anchor 20 R035-R039 | Relation | 0.7789 | 0.0032 | 5/5 relation-positive |
| No-targeted anchor 20 R035-R039 | COCO I2T R@1 | 0.7092 | 0.0232 | 3/5 pass |

## Winoground R034

| System | Text Score | Image Score | Group Score | Interpretation |
|---|---:|---:|---:|---|
| R003 OpenCLIP | 0.2775 | 0.1075 | 0.0825 | baseline |
| R005 plain adapter | 0.2700 | 0.0850 | 0.0650 | below OpenCLIP |
| R028 selected RRB | 0.2400 | 0.1100 | 0.0700 | image slightly up, text/group down |

## Key Findings

1. Hard token bottlenecking proves relation signal is reachable, but it destroys COCO/ImageNet guardrails.
2. R028 remains the best single checkpoint for the bounded paper story: relation `0.7734`, COCO `0.7340`, ImageNet `0.6342`.
3. Stronger anchoring lowers residual drift, but COCO remains seed-fragile even at `lambda_anchor=20`.
4. R033/R040 show that no-slot residual controls can learn relation metrics while remaining unsafe for broad CLIP behavior.
5. R034 does not support broad Winoground binding/role generalization.
6. The 2026-04-30 review scored the package `5/10` for a top ML venue; the bounded writeup should foreground Winoground, seed fragility, and the lack of direct slot-mechanism evidence.

## Top-Venue Follow-up R041-R045

Remote root: `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_topvenue_20260430_163958`  
Local summary copy: `outputs/remote/rrb_topvenue_20260430_163958/analysis/R041_R045_results.md`

### R041 Slot Responsibility

| Checkpoint | Examples | Entropy | Top-1 Slot Conc. | Edited Overlap | Slot-Delta Lift | Verdict |
|---|---:|---:|---:|---:|---:|---|
| R028 | 1252 | 1.0000 | 0.1259 | 0.1397 | 0.0001 | diffuse_or_unproven |
| R035 | 1252 | 1.0000 | 0.1259 | 0.1396 | 0.0001 | diffuse_or_unproven |
| R038 | 1252 | 1.0000 | 0.1261 | 0.1395 | -0.0000 | diffuse_or_unproven |

Interpretation: the slot-responsibility probe does not support interpretable relation-specialized slot language.

### R042-R044 ROP Ablation

R042 built a rank-64 protected PCA basis from 4096 frozen OpenCLIP anchor embeddings, with cumulative variance explained `0.6969`.

| Run | Relation Acc. | COCO I2T R@1 | ImageNet Top-1 | Residual Mag. | Protected Energy | Status |
|---|---:|---:|---:|---:|---:|---|
| R043 | 0.7441 | 0.7360 | 0.6566 | 0.0792 | 0.0001 | gate fail |

R043 improved guardrails relative to R028 only marginally, but relation accuracy fell below the `0.76` pilot gate. R044 replication seeds were skipped by design.

### R045 Flickr30k Retrieval

| System | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---:|---:|---:|---:|---:|---:|
| OpenCLIP | 0.8620 | 0.9800 | 0.9950 | 0.6980 | 0.9040 | 0.9456 |
| R005 | 0.8600 | 0.9720 | 0.9900 | 0.6736 | 0.8904 | 0.9362 |
| R028 | 0.8260 | 0.9580 | 0.9800 | 0.6962 | 0.8988 | 0.9444 |
| R038 | 0.8440 | 0.9660 | 0.9870 | 0.6964 | 0.9016 | 0.9442 |
| R043 | 0.8330 | 0.9650 | 0.9870 | 0.6988 | 0.9032 | 0.9448 |

Interpretation: Flickr30k confirms a non-COCO retrieval tax for the RRB checkpoints, especially image-to-text R@1.

## Remote Roots

- `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/updated_end2end_20260428_115746`
- `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_20260428_131556`
- `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m3b_20260428_201251`
- `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_active_20260429_155721`
- `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m7_gpu0_20260429_164404`
- `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m7_gpu1_20260429_164404`
- `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m8_gpu0_20260429_211743`
- `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_m8_gpu1_20260429_211743`
- `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/winoground_r034_20260429_215139`
- `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_topvenue_20260430_163958`
