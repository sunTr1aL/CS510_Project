# R041-R045 Initial Results

Output root: `/shared/nas/data/m1/zixuans8/relbottleneck/outputs/rrb_topvenue_20260430_163958`

## R041 Slot Responsibility

| Checkpoint | Examples | Entropy | Top-1 conc. | Edited overlap | Slot-delta lift | Verdict |
|---|---:|---:|---:|---:|---:|---|
| R028 | 1252 | 1.0000 | 0.1259 | 0.1397 | 0.0001 | diffuse_or_unproven |
| R035 | 1252 | 1.0000 | 0.1259 | 0.1396 | 0.0001 | diffuse_or_unproven |
| R038 | 1252 | 1.0000 | 0.1261 | 0.1395 | -0.0000 | diffuse_or_unproven |

## R042-R044 ROP Ablation

- R042 protected rank: 64; cumulative variance: 0.6969
- R043 gate: FAIL; relation 0.7441, COCO I2T R@1 0.7360, ImageNet top-1 0.6566

| Run | Relation acc. | COCO I2T R@1 | ImageNet top-1 | Residual mag. | Protected energy | ROP loss |
|---|---:|---:|---:|---:|---:|---:|
| R043 | 0.7441 | 0.7360 | 0.6566 | 0.0792 | 0.0001 | 0.0001 |

## R045 Flickr30k Retrieval

| System | Images | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenCLIP | 1000 | 0.8620 | 0.9800 | 0.9950 | 0.6980 | 0.9040 | 0.9456 |
| R005 | 1000 | 0.8600 | 0.9720 | 0.9900 | 0.6736 | 0.8904 | 0.9362 |
| R028 | 1000 | 0.8260 | 0.9580 | 0.9800 | 0.6962 | 0.8988 | 0.9444 |
| R038 | 1000 | 0.8440 | 0.9660 | 0.9870 | 0.6964 | 0.9016 | 0.9442 |
| R043 | 1000 | 0.8330 | 0.9650 | 0.9870 | 0.6988 | 0.9032 | 0.9448 |

