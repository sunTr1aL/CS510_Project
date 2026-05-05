# R030 Diagnostics

Selected run: `R028`
Checkpoint: `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_active_20260429_155721/R028/adapter.pt`
Status: `done`

## Metrics

| Metric | Value |
|---|---:|
| Relation accuracy | 0.7734 |
| COCO I2T R@1 | 0.7340 |
| ImageNet top-1 | 0.6342 |
| Residual relative magnitude mean | 0.0804 |
| z0 cosine mean | 0.9963 |

## Failure Taxonomy

| Source | Edit type | Failures | Total | Failure rate | Mean margin |
|---|---|---:|---:|---:|---:|
| aro_visual_relation | feeding | 1 | 1 | 1.0000 | -0.0053 |
| aro_visual_relation | hanging off | 1 | 1 | 1.0000 | -0.0237 |
| aro_visual_relation | standing behind | 1 | 1 | 1.0000 | -0.0011 |
| aro_visual_relation | touching | 1 | 1 | 1.0000 | -0.0066 |
| aro_visual_relation | of | 5 | 7 | 0.7143 | -0.0054 |
| aro_visual_relation | behind | 7 | 12 | 0.5833 | -0.0042 |
| aro_visual_relation | next to | 4 | 7 | 0.5714 | -0.0011 |
| aro_visual_relation | to the right of | 89 | 164 | 0.5427 | -0.0006 |
| aro_visual_relation | to the left of | 87 | 163 | 0.5337 | -0.0009 |
| aro_visual_relation | above | 4 | 8 | 0.5000 | 0.0191 |
| aro_visual_relation | on top of | 4 | 8 | 0.5000 | 0.0051 |
| aro_visual_relation | under | 2 | 4 | 0.5000 | 0.0047 |
| aro_visual_relation | near | 10 | 22 | 0.4545 | 0.0026 |
| aro_visual_relation | below | 3 | 7 | 0.4286 | -0.0023 |
| aro_visual_relation | with | 2 | 6 | 0.3333 | 0.0158 |
| aro_visual_relation | on | 11 | 37 | 0.2973 | 0.0189 |
| sugarcrepe_swap_obj | swap_obj | 73 | 246 | 0.2967 | 0.0139 |
| sugarcrepe_swap_att | swap_att | 142 | 512 | 0.2773 | 0.0144 |
| aro_visual_attribution | aro_visual_attribution | 141 | 512 | 0.2754 | 0.0138 |
| aro_visual_relation | in front of | 3 | 13 | 0.2308 | 0.0149 |
| aro_visual_relation | in | 2 | 10 | 0.2000 | 0.0150 |
| sugarcrepe_replace_rel | replace_rel | 98 | 512 | 0.1914 | 0.0237 |
| aro_coco_order | word_order | 89 | 512 | 0.1738 | 0.0134 |
| sugarcrepe_replace_att | replace_att | 61 | 512 | 0.1191 | 0.0374 |
| aro_visual_relation | wearing | 1 | 18 | 0.0556 | 0.0216 |
| sugarcrepe_replace_obj | replace_obj | 26 | 512 | 0.0508 | 0.0748 |
| aro_visual_relation | covered in | 0 | 1 | 0.0000 | 0.0120 |
| aro_visual_relation | full of | 0 | 1 | 0.0000 | 0.0011 |
| aro_visual_relation | holding | 0 | 2 | 0.0000 | 0.0618 |
| aro_visual_relation | inside | 0 | 1 | 0.0000 | 0.0150 |
| aro_visual_relation | leaning against | 0 | 1 | 0.0000 | 0.0102 |
| aro_visual_relation | leaning on | 0 | 1 | 0.0000 | 0.0183 |
| aro_visual_relation | looking at | 0 | 1 | 0.0000 | 0.0451 |
| aro_visual_relation | lying on | 0 | 1 | 0.0000 | 0.0340 |
| aro_visual_relation | parked in front of | 0 | 1 | 0.0000 | 0.0276 |
| aro_visual_relation | parked next to | 0 | 1 | 0.0000 | 0.0190 |
| aro_visual_relation | parked on | 0 | 1 | 0.0000 | 0.0455 |
| aro_visual_relation | playing with | 0 | 2 | 0.0000 | 0.0154 |
| aro_visual_relation | sitting at | 0 | 1 | 0.0000 | 0.0299 |
| aro_visual_relation | sitting in | 0 | 1 | 0.0000 | 0.0451 |
| aro_visual_relation | sitting near | 0 | 2 | 0.0000 | 0.0505 |
| aro_visual_relation | sitting on | 0 | 1 | 0.0000 | 0.0373 |
| aro_visual_relation | using | 0 | 1 | 0.0000 | 0.0310 |
| aro_visual_relation | walking on | 0 | 1 | 0.0000 | 0.0011 |
| aro_visual_relation | watching | 0 | 1 | 0.0000 | 0.0087 |

## Hardest Failures

| Index | Source | Edit type | Margin | Positive | Best negative |
|---:|---|---|---:|---:|---:|
| 201 | sugarcrepe_replace_rel | replace_rel | -0.0837 | 0.1711 | 0.2547 |
| 393 | sugarcrepe_replace_rel | replace_rel | -0.0670 | 0.2788 | 0.3458 |
| 2936 | aro_visual_attribution | aro_visual_attribution | -0.0644 | 0.1836 | 0.2480 |
| 2416 | aro_visual_relation | near | -0.0604 | 0.2046 | 0.2651 |
| 2850 | aro_visual_attribution | aro_visual_attribution | -0.0583 | 0.1669 | 0.2252 |
| 2026 | sugarcrepe_swap_att | swap_att | -0.0583 | 0.1654 | 0.2237 |
| 2361 | aro_visual_relation | to the right of | -0.0576 | 0.2769 | 0.3345 |
| 2364 | aro_visual_relation | to the right of | -0.0576 | 0.2769 | 0.3345 |
| 2851 | aro_visual_attribution | aro_visual_attribution | -0.0536 | 0.1491 | 0.2026 |
| 1551 | sugarcrepe_swap_att | swap_att | -0.0480 | 0.2439 | 0.2919 |

## Closest Correct Cases

| Index | Source | Edit type | Margin | Positive | Best negative |
|---:|---|---|---:|---:|---:|
| 3325 | aro_coco_order | word_order | 0.0000 | 0.3648 | 0.3648 |
| 3331 | aro_coco_order | word_order | 0.0000 | 0.3410 | 0.3410 |
| 3343 | aro_coco_order | word_order | 0.0000 | 0.3161 | 0.3161 |
| 3344 | aro_coco_order | word_order | 0.0000 | 0.3494 | 0.3494 |
| 3360 | aro_coco_order | word_order | 0.0000 | 0.3514 | 0.3514 |
| 3368 | aro_coco_order | word_order | 0.0000 | 0.2583 | 0.2583 |
| 3377 | aro_coco_order | word_order | 0.0000 | 0.2856 | 0.2856 |
| 3378 | aro_coco_order | word_order | 0.0000 | 0.3584 | 0.3584 |
| 3397 | aro_coco_order | word_order | 0.0000 | 0.2636 | 0.2636 |
| 3398 | aro_coco_order | word_order | 0.0000 | 0.2498 | 0.2498 |
