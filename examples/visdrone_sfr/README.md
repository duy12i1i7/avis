# YOLO26 VisDrone Attack Build

This directory now contains four VisDrone-focused paths:

- the earlier routed `SFR` prototype for ablations
- the `P2 + tiny-object-aware training` attack recipe for upper-bound recall
- the current default `RSPB` recipe that injects selective P2 detail into P3 without a full P2 detection head
- the new `SPD` recipe that adds a frozen dense teacher during training and keeps the lightweight student at inference
- the new `SFR host-module bench` path that tests `SparseSubpixelExpert` under `C2f`, `C3k`, and `C3k2` hosts

## Files

- `ultralytics/cfg/models/26/yolo26n-spd-visdrone.yaml`: shadow-pyramid distilled student, the current default candidate
- `ultralytics/cfg/models/26/yolo26n-rspb-visdrone.yaml`: route-selective P2 bridge, the current default candidate
- `ultralytics/cfg/models/26/yolo26n-p2-visdrone.yaml`: fixed `n` model for strong tiny-human recall
- `ultralytics/cfg/models/26/yolo26s-p2-visdrone.yaml`: heavier `s` variant for accuracy attacks
- `ultralytics/nn/modules/routed.py`: sparse routed expert and `SFRC2f`
- `ultralytics/nn/modules/routed.py`: sparse routed expert plus `SFRC2f`, `SFRC3k`, and `SFRC3k2`
- `ultralytics/cfg/models/26/yolo26-sfr-visdrone.yaml`: routed YOLO26 variant
- `ultralytics/cfg/models/26/yolo26n-sfr-visdrone.yaml`: fixed `n` variant without scale-warning
- `ultralytics/cfg/models/26/yolo26n-sfrc2f-visdrone.yaml`: explicit `SFRC2f` host benchmark
- `ultralytics/cfg/models/26/yolo26n-sfrc3k-visdrone.yaml`: explicit `SFRC3k` host benchmark
- `ultralytics/cfg/models/26/yolo26n-sfrc3k2-visdrone.yaml`: explicit `SFRC3k2` host benchmark
- `ultralytics/cfg/models/11/yolo11n-sfrc2f-visdrone.yaml`: cross-YOLO `SFRC2f` transfer benchmark
- `ultralytics/cfg/models/v8/yolov8n-sfrc2f-visdrone.yaml`: `YOLOv8n` transfer benchmark
- `ultralytics/cfg/models/v10/yolov10n-sfrc2f-visdrone.yaml`: `YOLOv10n` transfer benchmark
- `ultralytics/cfg/models/12/yolo12n-sfrc2f-visdrone.yaml`: `YOLO12n` compatibility-limited transfer benchmark
- `ultralytics/models/yolo/detect/visdrone.py`: VisDrone trainer with tiny-image oversampling and batch upscale bias
- `examples/visdrone_sfr/train_psr_yolo26.py`: training entrypoint for the attack recipe
- `examples/visdrone_sfr/train_sfr_module_bench.py`: training entrypoint for host-module transfer studies
- `examples/visdrone_sfr/run_sfr_full_matrix.sh`: one-shot launcher for the full SFR train/eval matrix
- `examples/visdrone_sfr/run_sfr_dataset_suite.sh`: one-shot launcher for running the full matrix across VisDrone, AI-TOD-v2, and TinyPerson
- `examples/visdrone_sfr/prepare_coco_detection_dataset.py`: convert COCO-style datasets such as AI-TOD-v2 and TinyPerson into YOLO labels plus a dataset YAML
- `examples/visdrone_sfr/val_psr_yolo26.py`: validation entrypoint
- `examples/visdrone_sfr/tiny_human_eval.py`: computes `tiny-human AP` for `pedestrian` and `people`
- `examples/visdrone_sfr/run_kaggle_dual_t4.sh`: launcher for Kaggle 2xT4
- `examples/visdrone_sfr/KAGGLE.md`: Kaggle notebook instructions
- `docs/visdrone_sfr_experiment_protocol.md`: full SFR host-module experiment protocol

## Recommended training

```bash
cd /Users/udy/avis/ultralytics
python3 examples/visdrone_sfr/train_psr_yolo26.py \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 16 \
  --device 0
```

For a harder accuracy push on 2xT4:

```bash
cd /Users/udy/avis/ultralytics
python3 examples/visdrone_sfr/train_psr_yolo26.py \
  --model ultralytics/cfg/models/26/yolo26s-p2-visdrone.yaml \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 8 \
  --device 0,1
```

## Validation and tiny-human AP

```bash
cd /Users/udy/avis/ultralytics
python3 examples/visdrone_sfr/val_psr_yolo26.py \
  --model runs/visdrone/yolo26_sfr_visdrone/weights/best.pt \
  --data VisDrone.yaml \
  --imgsz 960 \
  --save-json

python3 examples/visdrone_sfr/tiny_human_eval.py \
  --pred-json runs/visdrone/yolo26_sfr_val/predictions.json \
  --data VisDrone.yaml
```

## Notes

- The scripts assume the standard Ultralytics `VisDrone.yaml` split: `train 6471 / val 548 / test 1610`.
- For multi-dataset studies, keep `VisDrone.yaml` as-is and convert `AI-TOD-v2` / `TinyPerson` with `prepare_coco_detection_dataset.py`.
- The current default attack recipe is `YOLO26n-SPD` with transfer learning from `yolo26n.pt`.
- `SPD` uses a frozen `YOLO26n-P2` teacher only during training and keeps the student inference graph lightweight.
- `RSPB` keeps only P3-P5 detection outputs and uses a selective P2 bridge for small-object detail recovery.
- `VisDroneAttackTrainer` only changes training-time behavior; inference graph still comes from the YAML model.
- Compare against `yolo26.yaml` and `yolo26-p2.yaml` at the same image size for a fair study.
