# SFR Multi-Dataset Suite

This note extends the SFR host-module benchmark beyond `VisDrone` so the same matrix can be trained on:

- `VisDrone`
- `AI-TOD-v2`
- `TinyPerson`

The intended claim is stronger than a single-dataset gain:

> the routed `SparseSubpixelExpert` remains useful under multiple YOLO host modules and does not collapse once the benchmark changes from a UAV detection dataset to a tiny-object or tiny-person dataset.

## 1. Prepare AI-TOD-v2 and TinyPerson

`VisDrone` already has `VisDrone.yaml`. The other two datasets should be normalized into YOLO labels first.

Use:

- [prepare_coco_detection_dataset.py](/Users/udy/avis/ultralytics/examples/visdrone_sfr/prepare_coco_detection_dataset.py)

This script expects:

- a train image directory
- a train COCO-style annotation JSON
- a val image directory
- a val COCO-style annotation JSON

It writes:

- `images/{split}`
- `labels/{split}`
- a dataset YAML

### AI-TOD-v2 example

```bash
cd /Users/udy/avis/ultralytics

python3 examples/visdrone_sfr/prepare_coco_detection_dataset.py \
  --name aitodv2 \
  --output /data/aitodv2_yolo \
  --train-images /data/AI-TOD-v2/train/images \
  --train-json /data/AI-TOD-v2/train.json \
  --val-images /data/AI-TOD-v2/val/images \
  --val-json /data/AI-TOD-v2/val.json
```

### TinyPerson example

```bash
cd /Users/udy/avis/ultralytics

python3 examples/visdrone_sfr/prepare_coco_detection_dataset.py \
  --name tinyperson \
  --output /data/tinyperson_yolo \
  --train-images /data/TinyPerson/train/images \
  --train-json /data/TinyPerson/train.json \
  --val-images /data/TinyPerson/val/images \
  --val-json /data/TinyPerson/val.json
```

The script prints the generated YAML path. Use that path in the suite launcher.

## 2. Run the full three-dataset suite

Use:

- [run_sfr_dataset_suite.sh](/Users/udy/avis/ultralytics/examples/visdrone_sfr/run_sfr_dataset_suite.sh)

Example:

```bash
cd /Users/udy/avis/ultralytics

bash examples/visdrone_sfr/run_sfr_dataset_suite.sh \
  --stage train \
  --device 0 \
  --epochs 300 \
  --patience 80 \
  --workers 4 \
  --visdrone-data VisDrone.yaml \
  --aitodv2-data /data/aitodv2.yaml \
  --tinyperson-data /data/tinyperson.yaml \
  --batch 8 \
  --imgsz 960
```

This launches the full model matrix on each dataset:

- `YOLO26n` baseline
- `YOLO26n-SFRC2f`
- `YOLO26n-SFRC3k`
- `YOLO26n-SFRC3k2`
- `YOLO11n` baseline
- `YOLO11n-SFRC2f`
- `YOLOv8n` baseline
- `YOLOv8n-SFRC2f`
- `YOLOv10n` baseline
- `YOLOv10n-SFRC2f`
- `YOLO12n` baseline
- `YOLO12n-SFRC2f`

## One-shot local setup

If you want one command that:

- creates a local `venv`
- installs the repo
- prepares `AI-TOD-v2` and `TinyPerson` from COCO-style annotations
- launches the three-dataset suite

use:

- [setup_and_run_multidataset.sh](/Users/udy/avis/ultralytics/examples/visdrone_sfr/setup_and_run_multidataset.sh)

Example:

```bash
cd /Users/udy/avis/ultralytics

bash examples/visdrone_sfr/setup_and_run_multidataset.sh \
  --device 0 \
  --epochs 300 \
  --batch 8 \
  --imgsz 960 \
  --visdrone-data VisDrone.yaml \
  --aitodv2-output /data/aitodv2_yolo \
  --aitodv2-train-images /data/AI-TOD-v2/train/images \
  --aitodv2-train-json /data/AI-TOD-v2/train.json \
  --aitodv2-val-images /data/AI-TOD-v2/val/images \
  --aitodv2-val-json /data/AI-TOD-v2/val.json \
  --tinyperson-output /data/tinyperson_yolo \
  --tinyperson-train-images /data/TinyPerson/train/images \
  --tinyperson-train-json /data/TinyPerson/train.json \
  --tinyperson-val-images /data/TinyPerson/val/images \
  --tinyperson-val-json /data/TinyPerson/val.json
```

## One-shot from clone

If you want the process to start from repository bootstrap as well, use:

- [bootstrap_sfr_multidataset.sh](/Users/udy/avis/ultralytics/bootstrap_sfr_multidataset.sh)

Example:

```bash
bash bootstrap_sfr_multidataset.sh --repo-dir /workspace/avis -- \
  --device 0 \
  --epochs 300 \
  --batch 8 \
  --imgsz 960 \
  --visdrone-data VisDrone.yaml \
  --aitodv2-data /data/aitodv2.yaml \
  --tinyperson-data /data/tinyperson.yaml
```

Each dataset gets its own project subtree:

- `runs/sfr_suite/visdrone`
- `runs/sfr_suite/aitodv2`
- `runs/sfr_suite/tinyperson`

## 3. Resume and eval

The suite launcher delegates to:

- [run_sfr_full_matrix.sh](/Users/udy/avis/ultralytics/examples/visdrone_sfr/run_sfr_full_matrix.sh)

So resume/skip logic still applies:

- finished runs are skipped
- interrupted runs resume from `last.pt`
- suffixed reruns like `name2`, `name3` are resolved automatically to the run with the deepest history

To evaluate only:

```bash
cd /Users/udy/avis/ultralytics

bash examples/visdrone_sfr/run_sfr_dataset_suite.sh \
  --stage eval \
  --device 0 \
  --visdrone-data VisDrone.yaml \
  --aitodv2-data /data/aitodv2.yaml \
  --tinyperson-data /data/tinyperson.yaml \
  --batch 8 \
  --imgsz 960
```

## 4. Tiny-human metrics

`tiny_human_eval.py` now recognizes:

- `pedestrian`
- `people`
- `person`

So:

- `VisDrone` tiny-human eval works
- `TinyPerson` tiny-human eval works
- `AI-TOD-v2` tiny-human eval is skipped automatically when no human class is present
