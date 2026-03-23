# Kaggle 2xT4 Guide

Use a Kaggle Notebook, not an interactive Python cell, for multi-GPU training. Ultralytics will spawn distributed workers when you pass `--device 0,1`, and that is more reliable when launched as a script.

## 1. Notebook settings

- Accelerator: `GPU`
- GPU count: `2`
- Internet: `ON` if you want GitHub clone and automatic VisDrone download

Check that Kaggle sees both GPUs:

```bash
!nvidia-smi -L
```

You should see two `Tesla T4` devices.

## 2. Pull the repo

If internet is ON:

```bash
!git clone https://github.com/<github-user>/<repo-name>.git /kaggle/working/ultralytics-sfr
%cd /kaggle/working/ultralytics-sfr
```

If internet is OFF:

- upload your repo as a Kaggle Dataset
- attach it as an input
- copy/unzip it into `/kaggle/working`

## 3. Install the project

```bash
%cd /kaggle/working/ultralytics-sfr
!python3 -m pip install --upgrade pip
!python3 -m pip install -e .
!python3 -m pip install pycocotools
```

## 4. Train on 2xT4

If internet is ON, `VisDrone.yaml` can auto-download and convert the dataset.

```bash
%cd /kaggle/working/ultralytics-sfr
!bash examples/visdrone_sfr/run_kaggle_dual_t4.sh \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 16 \
  --epochs 500
```

Notes:

- `--device 0,1` is already baked into the launcher.
- The default launcher now uses `yolo26n-p2-visdrone.yaml` plus pretrained transfer from `yolo26n.pt`.
- `--batch 16` is a safe first try for `2xT4`; if OOM, try `12` or `8`.
- If throughput is unstable, reduce `--workers` from `4` to `2`.

For a heavier accuracy-first run:

```bash
!bash examples/visdrone_sfr/run_kaggle_dual_t4.sh \
  --model ultralytics/cfg/models/26/yolo26s-p2-visdrone.yaml \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 8 \
  --epochs 500
```

## 5. If internet is OFF and you attach VisDrone as Kaggle Input

Create a small YAML in a notebook cell:

```python
from pathlib import Path

yaml_text = """
path: /kaggle/input/visdrone/VisDrone
train: images/train
val: images/val
test: images/test
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
Path("/kaggle/working/VisDrone-kaggle.yaml").write_text(yaml_text)
```

Then train with:

```bash
!bash examples/visdrone_sfr/run_kaggle_dual_t4.sh \
  --data /kaggle/working/VisDrone-kaggle.yaml \
  --imgsz 960 \
  --batch 16 \
  --epochs 500
```

## 6. Validate and tiny-human AP

```bash
%cd /kaggle/working/ultralytics-sfr
!python3 examples/visdrone_sfr/val_psr_yolo26.py \
  --model /kaggle/working/runs/visdrone/yolo26_sfr_visdrone_kaggle/weights/best.pt \
  --data VisDrone.yaml \
  --imgsz 960 \
  --save-json

!python3 examples/visdrone_sfr/tiny_human_eval.py \
  --pred-json /kaggle/working/runs/visdrone/yolo26_sfr_val/predictions.json \
  --data VisDrone.yaml
```

## 7. Save outputs

Kaggle only preserves files under `/kaggle/working`. Your weights and logs are already written there by default:

- checkpoints: `/kaggle/working/runs/visdrone/.../weights`
- validation json: `/kaggle/working/runs/visdrone/.../predictions.json`

Before ending the session, use `Save Version` in Kaggle so the artifacts persist.
