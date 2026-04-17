from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.request import HTTPCookieProcessor, Request, build_opener


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


class GoogleDriveDownloadFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_download_form = False
        self.action: str | None = None
        self.fields: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "form" and attrs_dict.get("id") == "download-form":
            self.in_download_form = True
            self.action = attrs_dict.get("action")
            return
        if self.in_download_form and tag == "input":
            name = attrs_dict.get("name")
            value = attrs_dict.get("value", "")
            if name:
                self.fields[name] = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self.in_download_form:
            self.in_download_form = False


def stream_response_to_file(response, output_path: Path) -> None:
    with output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_google_drive_file(file_or_url: str, output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return

    if "://" not in file_or_url:
        file_or_url = f"https://drive.google.com/uc?id={file_or_url}&export=download"

    opener = build_opener(HTTPCookieProcessor())
    headers = {"User-Agent": "Mozilla/5.0"}
    response = opener.open(Request(file_or_url, headers=headers), timeout=120)
    content_type = response.headers.get("Content-Type", "").lower()
    content_disposition = response.headers.get("Content-Disposition", "")

    if "text/html" not in content_type and "attachment" in content_disposition.lower():
        stream_response_to_file(response, output_path)
        return

    html = response.read().decode("utf-8", "ignore")
    if "Virus scan warning" not in html or 'id="download-form"' not in html:
        raise RuntimeError(f"Could not download Google Drive file from {file_or_url}")

    parser = GoogleDriveDownloadFormParser()
    parser.feed(html)
    if not parser.action or not parser.fields:
        raise RuntimeError(f"Could not parse Google Drive confirmation form for {file_or_url}")

    confirm_url = parser.action
    if not confirm_url.startswith("http"):
        confirm_url = urljoin(response.geturl(), confirm_url)
    confirm_url = f"{confirm_url}?{urlencode(parser.fields)}"
    confirmed = opener.open(Request(confirm_url, headers=headers), timeout=120)
    stream_response_to_file(confirmed, output_path)


def download_google_drive_folder(folder: str, output_dir: Path) -> None:
    ensure_gdown()
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "gdown", "--folder", folder, "-O", str(output_dir)], check=True)


TINYPERSON_MINIMAL_ASSETS = {
    "tiny_set/train.tar.gz": "1tqGbW7_3X_-CpQvZ9ls3tYTJafsoKOr4",
    "tiny_set/test.tar.gz": "1uq148D2Nxs3JiHJmZW8zT1DEnd6cPelF",
    "tiny_set/annotations/tiny_set_train.json": "1vo-ggU2lltIIze9tIMhBCRpFEz3h2Hv4",
    "tiny_set/annotations/task/tiny_set_test_all.json": "16mIDH58dukozi2iQwBqDTURYBDNNWrxy",
}


def download_tinyperson_minimal_assets(raw_root: Path) -> None:
    raw_root = raw_root.expanduser().resolve()
    for rel_path, file_id in TINYPERSON_MINIMAL_ASSETS.items():
        download_google_drive_file(f"https://drive.google.com/uc?id={file_id}", raw_root / rel_path)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def directory_has_images(path: Path) -> bool:
    if path is None or not path.is_dir():
        return False
    return any(child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES for child in path.iterdir())


def normalize_image_dir(path: Path | None, split_name: str) -> Path | None:
    if path is None or not path.exists():
        return None
    path = path.expanduser().resolve()
    if directory_has_images(path):
        return path
    nested = path / split_name
    if directory_has_images(nested):
        return nested
    for candidate in sorted(path.rglob("*")):
        if candidate.is_dir() and directory_has_images(candidate):
            return candidate
    return None


def find_image_dir(root: Path, split_name: str) -> Path | None:
    root = root.expanduser().resolve()
    if not root.exists():
        return None
    direct = root / split_name
    direct = normalize_image_dir(direct, split_name)
    if direct is not None:
        return direct
    for candidate in root.rglob(split_name):
        candidate = normalize_image_dir(candidate, split_name)
        if candidate is not None:
            return candidate
    return None


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
    train_images = normalize_image_dir(train_images, "train")
    val_images = normalize_image_dir(val_images, "test")
    if train_images is None:
        train_images = find_image_dir(raw_root / "tiny_set", "train")
    if val_images is None:
        val_images = find_image_dir(raw_root / "tiny_set", "test")
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
