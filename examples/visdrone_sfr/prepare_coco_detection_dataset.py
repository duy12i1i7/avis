from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a COCO-style detection dataset into YOLO labels and YAML.")
    parser.add_argument("--name", required=True, help="Dataset name tag, e.g. aitodv2 or tinyperson.")
    parser.add_argument("--output", required=True, help="Output directory for YOLO-style dataset.")
    parser.add_argument("--train-images", required=True, help="Directory containing train images.")
    parser.add_argument("--train-json", required=True, help="COCO-style train annotation JSON.")
    parser.add_argument("--val-images", required=True, help="Directory containing val images.")
    parser.add_argument("--val-json", required=True, help="COCO-style val annotation JSON.")
    parser.add_argument("--test-images", default=None, help="Optional directory containing test images.")
    parser.add_argument("--test-json", default=None, help="Optional COCO-style test annotation JSON.")
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Whether to symlink or copy images into the YOLO dataset root.",
    )
    parser.add_argument("--yaml-path", default=None, help="Optional explicit output YAML path.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_category_mapping(annotation_paths: list[Path]) -> tuple[dict[int, int], list[str]]:
    categories: dict[int, str] = {}
    for path in annotation_paths:
        if path is None or not path.exists():
            continue
        for category in load_json(path).get("categories", []):
            category_id = int(category["id"])
            category_name = str(category["name"])
            if category_id in categories and categories[category_id] != category_name:
                raise ValueError(f"Category id clash for {category_id}: {categories[category_id]} vs {category_name}")
            categories[category_id] = category_name
    ordered = sorted(categories.items())
    cat_id_to_idx = {cat_id: idx for idx, (cat_id, _) in enumerate(ordered)}
    names = [name for _, name in ordered]
    return cat_id_to_idx, names


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def place_image(src: Path, dst: Path, mode: str) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def resolve_image_path(images_dir: Path, file_name: str) -> Path:
    rel = Path(file_name)
    candidates = [
        images_dir / rel,
        images_dir / rel.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image '{file_name}' under {images_dir}")


def convert_split(
    split: str,
    images_dir: Path,
    ann_path: Path,
    out_root: Path,
    cat_id_to_idx: dict[int, int],
    image_mode: str,
) -> None:
    data = load_json(ann_path)
    images = {int(img["id"]): img for img in data["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for image_id, image in images.items():
        file_name = str(image["file_name"])
        src_image = resolve_image_path(images_dir, file_name)
        dst_image = out_images / file_name
        dst_label = (out_labels / file_name).with_suffix(".txt")

        place_image(src_image, dst_image, image_mode)
        ensure_parent(dst_label)

        width = float(image["width"])
        height = float(image["height"])
        lines: list[str] = []
        for ann in anns_by_image.get(image_id, []):
            if ann.get("iscrowd", False):
                continue
            category_id = int(ann["category_id"])
            if category_id not in cat_id_to_idx:
                continue
            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            xc = (x + w / 2.0) / width
            yc = (y + h / 2.0) / height
            wn = w / width
            hn = h / height
            cls = cat_id_to_idx[category_id]
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        dst_label.write_text("".join(lines), encoding="utf-8")


def write_yaml(yaml_path: Path, out_root: Path, names: list[str], has_test: bool) -> None:
    lines = [
        f"path: {out_root}",
        "train: images/train",
        "val: images/val",
    ]
    if has_test:
        lines.append("test: images/test")
    lines.append("names:")
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    train_images = Path(args.train_images).expanduser().resolve()
    train_json = Path(args.train_json).expanduser().resolve()
    val_images = Path(args.val_images).expanduser().resolve()
    val_json = Path(args.val_json).expanduser().resolve()
    test_images = Path(args.test_images).expanduser().resolve() if args.test_images else None
    test_json = Path(args.test_json).expanduser().resolve() if args.test_json else None

    if bool(test_images) ^ bool(test_json):
        raise SystemExit("--test-images and --test-json must be provided together")

    annotation_paths = [train_json, val_json] + ([test_json] if test_json else [])
    cat_id_to_idx, names = build_category_mapping(annotation_paths)

    convert_split("train", train_images, train_json, output, cat_id_to_idx, args.image_mode)
    convert_split("val", val_images, val_json, output, cat_id_to_idx, args.image_mode)
    if test_images and test_json:
        convert_split("test", test_images, test_json, output, cat_id_to_idx, args.image_mode)

    yaml_path = Path(args.yaml_path).expanduser().resolve() if args.yaml_path else output.parent / f"{args.name}.yaml"
    write_yaml(yaml_path, output, names, has_test=bool(test_images and test_json))

    print(json.dumps({"dataset_root": str(output), "yaml": str(yaml_path), "names": names}, indent=2))


if __name__ == "__main__":
    main()
