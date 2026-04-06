# SFR SparseSubpixelExpert Experiment Protocol

This note turns the SFR paper idea into an experiment matrix that can defend a stronger claim:

> `SparseSubpixelExpert` is not a one-off improvement attached only to `C2f`.
> It remains effective when hosted by multiple YOLO extractor families and stays portable across recent YOLO backbones.

## Core claim to prove

The paper should not stop at:

- `YOLO26n-SFR beats YOLO26n`

That is too weak because a reviewer can still say:

- the gain may come from a lucky placement;
- the gain may be specific to one host block only;
- the gain may not transfer beyond the current YOLO26n build.

The stronger claim is:

- `SparseSubpixelExpert` is a reusable sparse-detail mechanism;
- it works when inserted into multiple extractor shells;
- it stays useful under the same VisDrone tiny-human recipe across more than one modern YOLO family.

## Experimental questions

### Q1. Host-module breadth on YOLO26

Does the same `SparseSubpixelExpert` remain beneficial when the host block changes?

Recommended variants on the same YOLO26n recipe:

- `YOLO26n` baseline
- `YOLO26n-SFRC2f`
- `YOLO26n-SFRC3k`
- `YOLO26n-SFRC3k2`

This is the most important section for the SFR paper.

### Q2. Portability across recent YOLO backbones

Does the same routed expert still help after the backbone family changes?

Recommended initial cross-family check:

- `YOLO26n` vs `YOLO26n-SFRC2f`
- `YOLO11n` vs `YOLO11n-SFRC2f`
- `YOLOv8n` vs `YOLOv8n-SFRC2f`
- `YOLOv10n` vs `YOLOv10n-SFRC2f`
- `YOLO12n` vs `YOLO12n-SFRC2f`

If more compute is available, extend later to:

- larger scales or additional host placements once non-`C2f` blocks are adapted

### Q3. Stability, not just peak score

Even if mean AP improves, does the method remain stable across seeds and training runs?

Each key variant should be run with at least:

- `seed 0`
- `seed 1`
- `seed 2`

Report:

- mean
- standard deviation
- best run

### Q4. Mechanism sanity

Do the routed experts actually run sparsely and consistently?

Recommended evidence:

- average route density per expert block
- route density variance across validation batches
- optional visualization of routed regions on tiny-human images

This is not optional if the paper wants to claim the mechanism itself matters.

## Controlled training recipe

All host-module ablations should use the same recipe:

- dataset: `VisDrone.yaml`
- image size: `960`
- epochs: `300` for module screening, `500` for finalists
- patience: `150`
- optimizer: `auto`
- trainer: `VisDroneAttackTrainer`
- pretrained initialization from the closest official checkpoint
- same `tiny_obj` settings
- same `visdrone_attack` settings

The point is to keep everything constant except the host block.

## Recommended experiment matrix

### Phase A. Host-module breadth on YOLO26n

Run the following models with identical recipe:

1. `ultralytics/cfg/models/26/yolo26n.yaml`
2. `ultralytics/cfg/models/26/yolo26n-sfrc2f-visdrone.yaml`
3. `ultralytics/cfg/models/26/yolo26n-sfrc3k-visdrone.yaml`
4. `ultralytics/cfg/models/26/yolo26n-sfrc3k2-visdrone.yaml`

Interpretation goal:

- if all three routed hosts improve over baseline, the expert is broad;
- if only one host improves, the claim should be narrowed;
- if `SFRC2f` is strongest, that supports the current SFR design choice;
- if `SFRC3k` or `SFRC3k2` wins, the paper should pivot to that host.

### Phase B. Cross-YOLO transfer of SFRC2f

Run:

1. `ultralytics/cfg/models/26/yolo26n.yaml`
2. `ultralytics/cfg/models/26/yolo26n-sfrc2f-visdrone.yaml`
3. `ultralytics/cfg/models/11/yolo11.yaml`
4. `ultralytics/cfg/models/11/yolo11n-sfrc2f-visdrone.yaml`
5. `ultralytics/cfg/models/v8/yolov8.yaml`
6. `ultralytics/cfg/models/v8/yolov8n-sfrc2f-visdrone.yaml`
7. `ultralytics/cfg/models/v10/yolov10n.yaml`
8. `ultralytics/cfg/models/v10/yolov10n-sfrc2f-visdrone.yaml`
9. `ultralytics/cfg/models/12/yolo12.yaml`
10. `ultralytics/cfg/models/12/yolo12n-sfrc2f-visdrone.yaml`

Interpretation goal:

- `SFRC2f` should improve the `YOLO26n`, `YOLO11n`, `YOLOv8n`, and `YOLOv10n` baselines;
- even if the absolute gain differs, the direction should stay positive;
- `YOLO12n` is a compatibility-limited transfer case because the current backbone is dominated by `A2C2f`, so the first port only swaps the `C3k2`-compatible stages;
- if gains disappear repeatedly outside `YOLO26n`, the paper should stop claiming broad portability.

## What to report

For every model:

- Precision
- Recall
- `mAP50`
- `mAP50-95`
- parameters
- GFLOPs
- `AP/GFLOP`
- tiny-human `AP50`
- tiny-human `AP50-95`
- latency or FPS on one deployment device if available

For every ablation group:

- mean over 3 seeds
- standard deviation over 3 seeds

## Required tables for the SFR paper

### Table 1. Main host-module comparison on YOLO26n

Rows:

- baseline
- SFRC2f
- SFRC3k
- SFRC3k2

Columns:

- params
- GFLOPs
- P
- R
- AP50
- AP50:95
- tiny AP50
- tiny AP50:95
- seed mean and std if space allows

### Table 2. Cross-family transfer of SFRC2f

Rows:

- YOLO26n baseline
- YOLO26n + SFRC2f
- YOLO11n baseline
- YOLO11n + SFRC2f
- YOLOv8n baseline
- YOLOv8n + SFRC2f
- YOLOv10n baseline
- YOLOv10n + SFRC2f
- YOLO12n baseline
- YOLO12n + SFRC2f

Columns:

- params
- GFLOPs
- AP50
- AP50:95
- tiny AP50
- tiny AP50:95

### Table 3. Route-density sanity check

Rows:

- each routed block family

Columns:

- mean route density
- std route density
- min
- max
- AP50:95

This table is how the paper proves the expert is actually sparse and not collapsing to dense behavior.

## Required figures

### Figure A. Host-block schematic

One compact figure comparing:

- `SFRC2f`
- `SFRC3k`
- `SFRC3k2`

Only show:

- outer shell
- repeated unit
- where `SparseSubpixelExpert` is injected

The reviewer should understand in one glance that the expert is being tested under different host modules, not just one.

### Figure B. Cross-family gain plot

A small bar plot or dot plot:

- x-axis: baseline YOLO families
- bars: `baseline` vs `+SFRC2f`
- y-axis: `mAP50-95` or tiny-human `AP50-95`

This figure is stronger than paragraphs of text.

### Figure C. Routed-region qualitative examples

For a few VisDrone images:

- input image
- predicted boxes
- routed-region mask or highlighted patches

Without this, the expert still feels abstract.

## Minimal run order

If GPU budget is limited, do it in this order:

1. `YOLO26n` baseline
2. `YOLO26n-SFRC2f`
3. `YOLO26n-SFRC3k`
4. `YOLO26n-SFRC3k2`
5. `YOLO11n` baseline
6. `YOLO11n-SFRC2f`
7. `YOLOv8n` baseline
8. `YOLOv8n-SFRC2f`
9. `YOLOv10n` baseline
10. `YOLOv10n-SFRC2f`
11. `YOLO12n` baseline
12. `YOLO12n-SFRC2f`

If only the first four are possible, the paper can still argue host-module breadth inside YOLO26.

## Training commands

YOLO26n host-module ablations:

```bash
cd /Users/udy/avis/ultralytics

python3 examples/visdrone_sfr/train_sfr_module_bench.py \
  --model ultralytics/cfg/models/26/yolo26n-sfrc2f-visdrone.yaml \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 16 \
  --epochs 300 \
  --device 0 \
  --name yolo26n_sfrc2f_visdrone

python3 examples/visdrone_sfr/train_sfr_module_bench.py \
  --model ultralytics/cfg/models/26/yolo26n-sfrc3k-visdrone.yaml \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 16 \
  --epochs 300 \
  --device 0 \
  --name yolo26n_sfrc3k_visdrone

python3 examples/visdrone_sfr/train_sfr_module_bench.py \
  --model ultralytics/cfg/models/26/yolo26n-sfrc3k2-visdrone.yaml \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 16 \
  --epochs 300 \
  --device 0 \
  --name yolo26n_sfrc3k2_visdrone
```

Cross-YOLO transfer:

```bash
cd /Users/udy/avis/ultralytics

python3 examples/visdrone_sfr/train_sfr_module_bench.py \
  --model ultralytics/cfg/models/11/yolo11n-sfrc2f-visdrone.yaml \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 16 \
  --epochs 300 \
  --device 0 \
  --name yolo11n_sfrc2f_visdrone

python3 examples/visdrone_sfr/train_sfr_module_bench.py \
  --model ultralytics/cfg/models/v8/yolov8n-sfrc2f-visdrone.yaml \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 16 \
  --epochs 300 \
  --device 0 \
  --name yolov8n_sfrc2f_visdrone

python3 examples/visdrone_sfr/train_sfr_module_bench.py \
  --model ultralytics/cfg/models/v10/yolov10n-sfrc2f-visdrone.yaml \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 16 \
  --epochs 300 \
  --device 0 \
  --name yolov10n_sfrc2f_visdrone

python3 examples/visdrone_sfr/train_sfr_module_bench.py \
  --model ultralytics/cfg/models/12/yolo12n-sfrc2f-visdrone.yaml \
  --data VisDrone.yaml \
  --imgsz 960 \
  --batch 16 \
  --epochs 300 \
  --device 0 \
  --name yolo12n_sfrc2f_visdrone
```

Validation:

```bash
python3 examples/visdrone_sfr/val_psr_yolo26.py \
  --model runs/visdrone/yolo26n_sfrc2f_visdrone/weights/best.pt \
  --data VisDrone.yaml \
  --imgsz 960 \
  --save-json \
  --name yolo26n_sfrc2f_val

python3 examples/visdrone_sfr/tiny_human_eval.py \
  --pred-json runs/visdrone/yolo26n_sfrc2f_val/predictions.json \
  --data VisDrone.yaml
```

## How to write the claim safely

If results are strong:

- `SparseSubpixelExpert consistently improves YOLO26n when hosted by C2f-, C3k-, and C3k2-style extractors, with the C2f host achieving the best efficiency-accuracy tradeoff.`

If only `SFRC2f` is strong:

- `SparseSubpixelExpert is transferable across multiple YOLO backbones, while C2f-style shells provide the most stable host among the tested extractor families.`

Avoid claiming:

- universal superiority across all hosts;
- superiority across all YOLO versions unless all results are positive.

## Decision rule after experiments

- Keep `SFRC2f` as the main paper path if it is best or tied-best with the most stable variance.
- Promote `SFRC3k` or `SFRC3k2` only if they win clearly enough to justify the extra complexity.
- If all routed hosts help on YOLO26n but only `SFRC2f` transfers to YOLO11n, then the paper narrative should be:
  - broad host viability inside YOLO26;
  - strongest portable version is `SFRC2f`.
- If `YOLO12n-SFRC2f` underperforms, frame it as a partial-transfer case caused by host incompatibility with `A2C2f`, not as evidence that the expert itself is invalid.
