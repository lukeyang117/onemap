# Instance-evidence comparison

Same three rooms, eight RGB-D views per room, fixed GT and fixed evaluation masks.

The script `../run_instance_evidence_comparison.py` compares view-balanced class evidence while keeping `large > small > floor/stairs`. The `--sam` option loads MobileSAM and refines small-object candidate components before the same evidence fusion. It is a conservative SAM ablation, not a full Open3DIS implementation.

The current best fixed rule is `margin_confirm`: 67.48% overall agreement, 14/46 small-object recall (30.43%), and 6.39% small-object precision. The existing strict layered rule is 64.54%, 7/46 (15.22%), and 2.52%. The conservative SAM ablation produced the same aggregate metrics on these cached inputs, so it did not add measurable value in this form. This does not rule out a full 3D instance-proposal method; it means that refining already predicted small-object seed components with MobileSAM did not change the evaluated evidence.

References motivating the design: Open3DIS (CVPR 2024), OVIR-3D (CoRL 2023), and OV3D-CG (ICCV 2025). Their central distinction is 3D instance proposal/aggregation before open-vocabulary recognition.
