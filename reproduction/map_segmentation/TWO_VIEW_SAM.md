# Two-view confirmation with MobileSAM

This experiment keeps the semantic priority `large_objects > small_objects > floor_stairs` and adds a conservative confirmation rule for small-object evidence.

## Pipeline

1. ConvNeXt-L/SED dense features are compared with the configured open-vocabulary text prompts.
2. Pixels predicted as the three added small-object classes form candidate connected components.
3. MobileSAM is prompted with each candidate box and refines the candidate region. SAM supplies a region boundary; it does not supply the semantic class.
4. RGB-D pixels are projected with the original camera intrinsics and the saved Habitat pose into the OneMap XY grid.
5. Evidence is counted independently per input view. A view supports a class only when its class fraction for the cell passes the fixed rule in `run_instance_evidence_comparison.py`.
6. A small-object class is confirmed only when at least two distinct views support that same map cell. Only confirmed classes enter the original priority reduction.

The current implementation is a reproducible 2D-cell ablation. “Two views” means two distinct camera frames support the same XY cell; it is not yet a full 3D instance tracker. The separate `run_3d_instance_pipeline.py` is the experimental 3D proposal prototype.

## Run

The experiment needs the already prepared local HM3D room outputs, cached ConvNeXt-L features, and `weights/mobile_sam.pt`:

```bash
MPLBACKEND=Agg HF_HUB_OFFLINE=1 HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet \
python reproduction/map_segmentation/run_instance_evidence_comparison.py --sam
```

Results are written to `reproduction/map_segmentation/multi_room/instance_evidence/summary.json`. The evaluation uses the same three rooms, eight RGB-D views per room, fixed GT masks, and fixed GT labels as the other semantic-elevation experiments. GT is loaded only for scoring.

## Measured result

On the prepared three-room experiment, two-view confirmation with SAM produced:

| metric | value |
|---|---:|
| overall semantic agreement | 64.68% |
| small-object recall | 16/46 = 34.78% |
| small-object precision | 16/43 = 37.21% |

The result is intentionally conservative: requiring two views improves precision and rejects single-view false positives, but can miss an object visible clearly in only one view. Thresholds and the evaluation domain are not changed by the script.

## References

The design is inspired by region-to-3D fusion in [Open3DIS](https://openaccess.thecvf.com/content/CVPR2024/papers/Nguyen_Open3DIS_Open-Vocabulary_3D_Instance_Segmentation_with_2D_Mask_Guidance_CVPR2024_paper.pdf), open-vocabulary 3D instance retrieval in [OVIR-3D](https://github.com/shiyoung77/OVIR-3D), and contextual/best-view instance reasoning in [OV3D-CG](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_OV3D-CG_Open-vocabulary_3D_Instance_Segmentation_with_Contextual_Guidance_ICCV_2025_paper.pdf).
