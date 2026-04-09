from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

AITODV2_ANNOTATIONS_FOLDER = "https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY?usp=sharing"
AITOD_WO_XVIEW_FOLDER = "https://drive.google.com/drive/folders/1uNY_rcOO5LrWibXRY6l2dvqSbK6xikJp?usp=sharing"
AI_TOD_OFFICIAL_REPO = "https://github.com/jwwangchn/AI-TOD.git"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    candidates = [images_dir / rel, images_dir / rel.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image '{file_name}' under {images_dir}")


def build_category_mapping(
    annotation_paths: list[Path],
    *,
    collapse_to_supercategory: bool = False,
    name_overrides: dict[str, str] | None = None,
) -> tuple[dict[int, int], list[str]]:
    categories: dict[int, str] = {}
    ordered_names: list[str] = []
    name_to_idx: dict[str, int] = {}
    cat_id_to_idx: dict[int, int] = {}

    for path in annotation_paths:
        if path is None or not path.exists():
            continue
        for category in load_json(path).get("categories", []):
            category_id = int(category["id"])
            category_name = str(category.get("name", category_id))
            if collapse_to_supercategory and category.get("supercategory"):
                category_name = str(category["supercategory"])
            if name_overrides:
                category_name = name_overrides.get(category_name, category_name)
            if category_id in categories and categories[category_id] != category_name:
                raise ValueError(f"Category id clash for {category_id}: {categories[category_id]} vs {category_name}")
            categories[category_id] = category_name

    for category_id, category_name in sorted(categories.items()):
        if category_name not in name_to_idx:
            name_to_idx[category_name] = len(ordered_names)
            ordered_names.append(category_name)
        cat_id_to_idx[category_id] = name_to_idx[category_name]

    return cat_id_to_idx, ordered_names


def convert_coco_split(
    split: str,
    images_dir: Path,
    ann_path: Path,
    out_root: Path,
    cat_id_to_idx: dict[int, int],
    *,
    image_mode: str = "symlink",
    skip_iscrowd: bool = True,
    ignore_true_keys: tuple[str, ...] = (),
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
            if skip_iscrowd and ann.get("iscrowd", False):
                continue
            if any(bool(ann.get(key, False)) for key in ignore_true_keys):
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


def write_detection_yaml(yaml_path: Path, out_root: Path, names: list[str], *, has_test: bool = False) -> None:
    lines = [f"path: {out_root}", "train: images/train", "val: images/val"]
    if has_test:
        lines.append("test: images/test")
    lines.append("names:")
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_output_dir(path: Path) -> Path:
    name = path.name
    if name.endswith(".tar.gz"):
        return path.with_name(name[:-7])
    if name.endswith(".tgz"):
        return path.with_name(name[:-4])
    return path.with_suffix("")


def extract_archives(root: Path) -> None:
    root = root.expanduser().resolve()
    for archive in sorted(root.rglob("*")):
        if not archive.is_file():
            continue
        lower = archive.name.lower()
        if not (
            lower.endswith(".zip")
            or lower.endswith(".tar")
            or lower.endswith(".tar.gz")
            or lower.endswith(".tgz")
        ):
            continue
        dest = archive_output_dir(archive)
        if dest.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if lower.endswith(".tgz"):
            shutil.unpack_archive(str(archive), str(dest), format="gztar")
        else:
            shutil.unpack_archive(str(archive), str(dest))


def find_first_existing(root: Path, rel_candidates: list[str]) -> Path | None:
    for rel in rel_candidates:
        candidate = (root / rel).expanduser().resolve()
        if candidate.exists():
            return candidate
    return None


def ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)


def download_google_drive_folder(folder: str, output_dir: Path) -> None:
    ensure_gdown()
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "gdown", "--folder", folder, "-O", str(output_dir)], check=True)


def ensure_python_packages(packages: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=True)


def git_clone_shallow(repo_url: str, target_dir: Path) -> None:
    target_dir = target_dir.expanduser().resolve()
    if (target_dir / ".git").exists():
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)], check=True)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        return
    if src.is_dir():
        os.symlink(src.resolve(), dst, target_is_directory=True)
    else:
        os.symlink(src.resolve(), dst)


def find_named_file(root: Path, names: list[str]) -> Path | None:
    lower_map = {name.lower(): name for name in names}
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in lower_map:
            return path.resolve()
    return None


def find_split_image_dir(root: Path, split: str) -> Path | None:
    preferred = [
        f"{split}/images",
        f"images/{split}",
        f"aitod/images/{split}",
        f"aitod_wo_xview_{split}_imgs",
    ]
    direct = find_first_existing(root, preferred)
    if direct:
        return direct

    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name.lower()
        parts = [part.lower() for part in path.parts]
        if split.lower() not in parts and split.lower() not in name:
            continue
        try:
            has_image = any(child.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} for child in path.iterdir())
        except OSError:
            continue
        if has_image:
            candidates.append(path.resolve())
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if "wo_xview" in str(p).lower() else 1, len(p.parts)))
    return candidates[0]


def download_aitod_public_assets(raw_root: Path) -> None:
    raw_root = raw_root.expanduser().resolve()
    annotations_root = raw_root / "aitodv2_annotations"
    wo_xview_root = raw_root / "aitod_wo_xview_public"
    if not annotations_root.exists() or not any(annotations_root.iterdir()):
        download_google_drive_folder(AITODV2_ANNOTATIONS_FOLDER, annotations_root)
    if not wo_xview_root.exists() or not any(wo_xview_root.iterdir()):
        download_google_drive_folder(AITOD_WO_XVIEW_FOLDER, wo_xview_root)
    extract_archives(raw_root)


def maybe_generate_aitod_images_from_xview(raw_root: Path) -> None:
    raw_root = raw_root.expanduser().resolve()
    if find_split_image_dir(raw_root, "train") and find_split_image_dir(raw_root, "val"):
        return

    xview_train_images = os.environ.get("XVIEW_TRAIN_IMAGES")
    xview_geojson = os.environ.get("XVIEW_GEOJSON")
    if not xview_train_images or not xview_geojson:
        return

    xview_train_images_path = Path(xview_train_images).expanduser().resolve()
    xview_geojson_path = Path(xview_geojson).expanduser().resolve()
    if not xview_train_images_path.exists() or not xview_geojson_path.exists():
        return

    public_root = raw_root / "aitod_wo_xview_public"
    annotations_dir = find_first_existing(public_root, ["complete_annotations"])
    if annotations_dir is None:
        ann_file = find_named_file(public_root, ["aitod_train.json"])
        annotations_dir = ann_file.parent if ann_file else None
    train_public = find_split_image_dir(public_root, "train")
    val_public = find_split_image_dir(public_root, "val")
    trainval_public = find_split_image_dir(public_root, "trainval")
    test_public = find_split_image_dir(public_root, "test")
    if not all([annotations_dir, train_public, val_public, trainval_public, test_public]):
        return

    toolkit_root = raw_root / "_aitod_toolkit"
    git_clone_shallow(AI_TOD_OFFICIAL_REPO, toolkit_root)

    work_root = raw_root / "_aitod_build"
    ensure_clean_dir(work_root)
    (work_root / "aitod" / "images").mkdir(parents=True, exist_ok=True)
    (work_root / "xview" / "ori").mkdir(parents=True, exist_ok=True)
    shutil.copy2(toolkit_root / "aitodtoolkit" / "generate_aitod_imgs.py", work_root / "generate_aitod_imgs.py")
    shutil.copytree(toolkit_root / "aitodtoolkit" / "aitod_xview", work_root / "aitod_xview", dirs_exist_ok=True)
    shutil.copytree(Path(annotations_dir), work_root / "aitod" / "annotations", dirs_exist_ok=True)

    split_sources = {
        "train": Path(train_public),
        "val": Path(val_public),
        "trainval": Path(trainval_public),
        "test": Path(test_public),
    }
    for split, src_dir in split_sources.items():
        dst_dir = work_root / "aitod" / "images" / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        for image in src_dir.rglob("*"):
            if image.is_file() and image.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                symlink_or_copy(image, dst_dir / image.name)

    symlink_or_copy(xview_train_images_path, work_root / "xview" / "ori" / "train_images")
    symlink_or_copy(xview_geojson_path, work_root / "xview" / "xView_train.geojson")

    ensure_python_packages(
        [
            "gdown",
            "opencv-python-headless",
            "scikit-image",
            "mmcv-lite",
            "git+https://github.com/jwwangchn/wwtool.git",
        ]
    )
    subprocess.run([sys.executable, "generate_aitod_imgs.py"], cwd=str(work_root), check=True)

    generated_root = work_root / "aitod" / "images"
    for split in ("train", "val", "trainval", "test"):
        src = generated_root / split
        if src.exists():
            dst = raw_root / "aitod" / "images" / split
            dst.mkdir(parents=True, exist_ok=True)
            for image in src.iterdir():
                if image.is_file():
                    symlink_or_copy(image, dst / image.name)


def prepare_tinyperson_from_raw(raw_root: Path, output_root: Path, *, yaml_path: Path, image_mode: str = "symlink") -> None:
    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    train_images = find_first_existing(
        raw_root,
        [
            "tiny_set/erase_with_uncertain_dataset/train",
            "tiny_set/train",
            "erase_with_uncertain_dataset/train",
            "train",
        ],
    )
    val_images = find_first_existing(raw_root, ["tiny_set/test", "test"])
    train_json = find_first_existing(
        raw_root,
        [
            "tiny_set/erase_with_uncertain_dataset/annotations/tiny_set_train.json",
            "tiny_set/annotations/tiny_set_train.json",
            "erase_with_uncertain_dataset/annotations/tiny_set_train.json",
            "annotations/tiny_set_train.json",
        ],
    )
    val_json = find_first_existing(
        raw_root,
        [
            "tiny_set/annotations/task/tiny_set_test_all.json",
            "tiny_set/annotations/tiny_set_test_all.json",
            "annotations/task/tiny_set_test_all.json",
            "annotations/tiny_set_test_all.json",
        ],
    )

    if not all([train_images, val_images, train_json, val_json]):
        raise FileNotFoundError(
            "TinyPerson raw root is incomplete. Expected a layout containing tiny_set/train, tiny_set/test, "
            "tiny_set/annotations/tiny_set_train.json, and tiny_set/annotations/task/tiny_set_test_all.json."
        )

    cat_id_to_idx, names = build_category_mapping(
        [train_json, val_json],
        collapse_to_supercategory=True,
        name_overrides={"person": "person"},
    )
    convert_coco_split(
        "train",
        train_images,
        train_json,
        output_root,
        cat_id_to_idx,
        image_mode=image_mode,
        ignore_true_keys=("ignore", "uncertain", "in_dense_image"),
    )
    convert_coco_split(
        "val",
        val_images,
        val_json,
        output_root,
        cat_id_to_idx,
        image_mode=image_mode,
        ignore_true_keys=("ignore", "uncertain", "in_dense_image"),
    )
    write_detection_yaml(yaml_path, output_root, names)


def prepare_aitodv2_from_raw(raw_root: Path, output_root: Path, *, yaml_path: Path, image_mode: str = "symlink") -> None:
    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    download_aitod_public_assets(raw_root)
    maybe_generate_aitod_images_from_xview(raw_root)

    train_images = find_split_image_dir(raw_root, "train")
    val_images = find_split_image_dir(raw_root, "val")
    train_json = find_first_existing(
        raw_root,
        [
            "aitodv2_annotations/aitodv2_train.json",
            "train.json",
            "annotations/train.json",
            "aitod/annotations/train.json",
            "annotations/aitodv2_train.json",
            "annotations/train_v2.json",
        ],
    )
    val_json = find_first_existing(
        raw_root,
        [
            "aitodv2_annotations/aitodv2_val.json",
            "val.json",
            "annotations/val.json",
            "aitod/annotations/val.json",
            "annotations/aitodv2_val.json",
            "annotations/val_v2.json",
        ],
    )

    if not all([train_images, val_images, train_json, val_json]):
        raise FileNotFoundError(
            "AI-TOD-v2 could not be prepared automatically. The pipeline already tried to download the public "
            "AI-TOD-v2 annotations and AI-TOD_wo_xview assets. The remaining missing piece is usually the xView "
            "training set required to synthesize the full AI-TOD image corpus. Set XVIEW_TRAIN_IMAGES and "
            "XVIEW_GEOJSON to let the official generator run automatically, or place a prepared raw root under "
            "AITODV2_RAW_ROOT with train/val images plus aitodv2_train.json and aitodv2_val.json."
        )

    cat_id_to_idx, names = build_category_mapping([train_json, val_json])
    convert_coco_split("train", train_images, train_json, output_root, cat_id_to_idx, image_mode=image_mode)
    convert_coco_split("val", val_images, val_json, output_root, cat_id_to_idx, image_mode=image_mode)
    write_detection_yaml(yaml_path, output_root, names)
