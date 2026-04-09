# SFR Multi-Dataset Suite

This note extends the SFR host-module benchmark beyond `VisDrone` so the same matrix can be trained on:

- `VisDrone`
- `TinyPerson`

The intended claim is stronger than a single-dataset gain:

> the routed `SparseSubpixelExpert` remains useful under multiple YOLO host modules and does not collapse once the benchmark changes from a UAV detection dataset to a tiny-object or tiny-person dataset.

## 1. Prepare TinyPerson

`VisDrone` already has `VisDrone.yaml`.

The repo now also ships:

- [TinyPerson.yaml](/Users/udy/avis/ultralytics/ultralytics/cfg/datasets/TinyPerson.yaml)

Behavior:

- `TinyPerson.yaml` can auto-download from the official Google Drive release and convert to YOLO.
If you already have raw COCO-style directories, you can still normalize TinyPerson manually.

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

## 2. Run the two-dataset suite

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
  --tinyperson-data TinyPerson.yaml \
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

## 2a. Canonical in-repo runner

The canonical "run all" entrypoint inside the repo is:

- [run_sfr_multidataset.sh](/Users/udy/avis/ultralytics/run_sfr_multidataset.sh)

It handles:

- creates a local `venv`
- installs the repo
- prepares `TinyPerson` if you provide raw COCO-style annotations
- launches the two-dataset suite

Example:

```bash
cd /Users/udy/avis/ultralytics

bash run_sfr_multidataset.sh \
  --stage train \
  --device 0 \
  --epochs 300 \
  --batch 8 \
  --imgsz 960 \
  --visdrone-data VisDrone.yaml
```

For evaluation only:

```bash
cd /Users/udy/avis/ultralytics

bash run_sfr_multidataset.sh \
  --stage eval \
  --device 0 \
  --visdrone-data VisDrone.yaml \
  --tinyperson-data TinyPerson.yaml \
  --batch 8 \
  --imgsz 960
```

For `TinyPerson.yaml`, you can optionally override the raw cache location:

```bash
export TINYPERSON_RAW_ROOT=/data/tinyperson_raw
```

## 2b. Bootstrap clone-and-run wrapper

If you want one extra script that can clone/pull the repo and then call the canonical runner, use:

- [bootstrap_sfr_multidataset.sh](/Users/udy/avis/ultralytics/bootstrap_sfr_multidataset.sh)

Example:

```bash
bash bootstrap_sfr_multidataset.sh --repo-dir /workspace/avis -- \
  --stage train \
  --device 0 \
  --epochs 300 \
  --batch 8 \
  --imgsz 960 \
  --visdrone-data VisDrone.yaml \
  --tinyperson-data TinyPerson.yaml
```

The older example helper:

- [setup_and_run_multidataset.sh](/Users/udy/avis/ultralytics/examples/visdrone_sfr/setup_and_run_multidataset.sh)

is now only a compatibility shim that forwards to the canonical root runner.

Each dataset gets its own project subtree:

- `runs/sfr_suite/visdrone`
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
  --tinyperson-data TinyPerson.yaml \
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
