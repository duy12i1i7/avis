# Routed YOLO26 for VisDrone

This prototype adds a `Perspective-Scale Routed` block to YOLO26 for tiny-object detection on VisDrone.

## Files

- `ultralytics/nn/modules/routed.py`: sparse routed expert and `SFRC2f`
- `ultralytics/cfg/models/26/yolo26-sfr-visdrone.yaml`: routed YOLO26 variant
- `examples/visdrone_sfr/train_psr_yolo26.py`: training entrypoint
- `examples/visdrone_sfr/val_psr_yolo26.py`: validation entrypoint
- `examples/visdrone_sfr/tiny_human_eval.py`: computes `tiny-human AP` for `pedestrian` and `people`

## Recommended training

```bash
cd /Users/udy/avis/ultralytics
python3 examples/visdrone_sfr/train_psr_yolo26.py \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 16 \
  --device 0
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
- The prototype keeps the standard three-scale detector and concentrates extra compute on the small-object pathway.
- Compare against `yolo26.yaml` and `yolo26-p2.yaml` at the same image size for a fair Pareto study.
