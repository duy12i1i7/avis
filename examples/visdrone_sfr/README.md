# YOLO26 VisDrone Attack Build

This directory now contains two VisDrone-focused paths:

- the earlier routed `SFR` prototype for ablations
- the current `P2 + tiny-object-aware training` attack recipe for maximum validation mAP

## Files

- `ultralytics/cfg/models/26/yolo26n-p2-visdrone.yaml`: fixed `n` model for strong tiny-human recall
- `ultralytics/cfg/models/26/yolo26s-p2-visdrone.yaml`: heavier `s` variant for accuracy attacks
- `ultralytics/nn/modules/routed.py`: sparse routed expert and `SFRC2f`
- `ultralytics/cfg/models/26/yolo26-sfr-visdrone.yaml`: routed YOLO26 variant
- `ultralytics/cfg/models/26/yolo26n-sfr-visdrone.yaml`: fixed `n` variant without scale-warning
- `ultralytics/models/yolo/detect/visdrone.py`: VisDrone trainer with tiny-image oversampling and batch upscale bias
- `examples/visdrone_sfr/train_psr_yolo26.py`: training entrypoint for the attack recipe
- `examples/visdrone_sfr/val_psr_yolo26.py`: validation entrypoint
- `examples/visdrone_sfr/tiny_human_eval.py`: computes `tiny-human AP` for `pedestrian` and `people`
- `examples/visdrone_sfr/run_kaggle_dual_t4.sh`: launcher for Kaggle 2xT4
- `examples/visdrone_sfr/KAGGLE.md`: Kaggle notebook instructions

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
- The current attack recipe defaults to `YOLO26n-P2` with transfer learning from `yolo26n.pt`.
- `VisDroneAttackTrainer` only changes training-time behavior; inference graph still comes from the YAML model.
- Compare against `yolo26.yaml` and `yolo26-p2.yaml` at the same image size for a fair study.
