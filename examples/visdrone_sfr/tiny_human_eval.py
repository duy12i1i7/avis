from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require_dependency(name: str, install_hint: str) -> None:
    try:
        __import__(name)
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit(f"Missing dependency '{name}'. Install it with: {install_hint}") from exc


def load_data_yaml(path_or_name: str) -> dict:
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    yaml_path = check_yaml(path_or_name)
    return YAML.load(yaml_path)


def build_tiny_gt(data: dict, split: str, area_thr: float, height_thr: float) -> dict:
    from PIL import Image

    names = data["names"]
    name_lookup = list(names.values()) if isinstance(names, dict) else list(names)
    target_classes = {idx + 1 for idx, name in enumerate(name_lookup) if name in {"pedestrian", "people", "person"}}

    root = Path(data["path"])
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    images = []
    annotations = []
    ann_id = 1
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        image_id = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0])) + 1
            if cls not in target_classes:
                continue
            xc, yc, w_norm, h_norm = map(float, parts[1:5])
            w_px = w_norm * width
            h_px = h_norm * height
            area = w_px * h_px
            if area > area_thr or h_px > height_thr:
                continue
            x0 = (xc - w_norm / 2.0) * width
            y0 = (yc - h_norm / 2.0) * height
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls,
                    "bbox": [x0, y0, w_px, h_px],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(name_lookup) if idx + 1 in target_classes]
    return {"images": images, "annotations": annotations, "categories": categories}


def filter_predictions(pred_json: Path, gt: dict, area_thr: float, height_thr: float) -> list[dict]:
    categories = {category["id"] for category in gt["categories"]}
    if not pred_json.exists():
        raise SystemExit(f"Prediction JSON not found: {pred_json}")
    if pred_json.stat().st_size == 0:
        raise SystemExit(f"Prediction JSON is empty: {pred_json}")
    try:
        with pred_json.open("r", encoding="utf-8") as handle:
            predictions = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Prediction JSON is invalid: {pred_json}") from exc
    filtered = []
    for pred in predictions:
        if pred["category_id"] not in categories:
            continue
        _, _, w_px, h_px = pred["bbox"]
        if (w_px * h_px) > area_thr or h_px > height_thr:
            continue
        filtered.append(pred)
    return filtered


def evaluate_tiny_humans(gt: dict, predictions: list[dict]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not gt["annotations"]:
        return {
            "tiny_human_ap50_95": 0.0,
            "tiny_human_ap50": 0.0,
            "tiny_human_ar1": 0.0,
            "tiny_human_ar10": 0.0,
        }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        gt_path = tmp_dir / "tiny_gt.json"
        pred_path = tmp_dir / "tiny_pred.json"
        gt_path.write_text(json.dumps(gt), encoding="utf-8")
        pred_path.write_text(json.dumps(predictions), encoding="utf-8")

        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.catIds = [category["id"] for category in gt["categories"]]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return {
            "tiny_human_ap50_95": float(evaluator.stats[0]),
            "tiny_human_ap50": float(evaluator.stats[1]),
            "tiny_human_ar1": float(evaluator.stats[6]),
            "tiny_human_ar10": float(evaluator.stats[7]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tiny-human AP on VisDrone predictions.json.")
    parser.add_argument("--pred-json", required=True, help="Path to predictions.json generated by Ultralytics val.")
    parser.add_argument("--data", default="VisDrone.yaml", help="Dataset YAML.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    parser.add_argument("--area-thr", type=float, default=32.0 * 32.0, help="Maximum bbox area in pixels.")
    parser.add_argument("--height-thr", type=float, default=20.0, help="Maximum bbox height in pixels.")
    parser.add_argument("--save", default=None, help="Optional path to save the tiny-human metrics JSON.")
    return parser.parse_args()


def main() -> None:
    require_dependency("PIL", "pip install pillow")
    require_dependency("pycocotools", "pip install pycocotools")

    args = parse_args()
    data = load_data_yaml(args.data)
    gt = build_tiny_gt(data, args.split, args.area_thr, args.height_thr)
    predictions = filter_predictions(Path(args.pred_json), gt, args.area_thr, args.height_thr)
    metrics = evaluate_tiny_humans(gt, predictions)
    print(json.dumps(metrics, indent=2))
    if args.save:
        Path(args.save).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
