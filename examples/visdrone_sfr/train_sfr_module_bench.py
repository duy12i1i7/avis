from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect import VisDroneAttackTrainer  # noqa: E402


def infer_pretrained_weights(model_path: str) -> str:
    """Map a custom YAML name to the closest official pretrained checkpoint."""
    name = Path(model_path).name.lower()
    for family in ("yolov10", "yolov8", "yolo26", "yolo12", "yolo11"):
        for scale in ("x", "l", "m", "s", "n"):
            if f"{family}{scale}" in name:
                return f"{family}{scale}.pt"
    return "yolo26n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFR host-module ablations on VisDrone.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26n-sfrc2f-visdrone.yaml"),
        help="Path to the custom model YAML.",
    )
    parser.add_argument(
        "--weights",
        default="auto",
        help="Pretrained weights path, 'auto' to infer from model name, or 'none' to train from scratch.",
    )
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs" / "visdrone"))
    parser.add_argument("--name", default="sfr_module_bench")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--multi-scale", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--cos-lr", dest="cos_lr", action="store_true")
    parser.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
    parser.set_defaults(cos_lr=True, amp=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_devices = [x for x in str(args.device).replace("cuda:", "").split(",") if x]
    if len(requested_devices) > 1 and torch.cuda.device_count() < len(requested_devices):
        raise RuntimeError(
            f"Requested {len(requested_devices)} CUDA devices via --device {args.device}, "
            f"but only {torch.cuda.device_count()} are visible."
        )
    if torch.cuda.is_available():
        print(
            "CUDA preflight:",
            {
                "requested": requested_devices or [str(args.device)],
                "visible_count": torch.cuda.device_count(),
                "visible_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            },
        )

    if args.resume:
        resume_path = Path(args.resume).resolve()
        resume_run_dir = resume_path.parents[1]
        model = YOLO(args.resume)
        train_kwargs = dict(
            trainer=VisDroneAttackTrainer,
            resume=True,
            data=args.data,
            project=str(resume_run_dir.parent),
            name=resume_run_dir.name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            cache=args.cache,
            optimizer=args.optimizer,
            patience=args.patience,
            seed=args.seed,
            amp=args.amp,
            cos_lr=args.cos_lr,
            close_mosaic=args.close_mosaic,
            multi_scale=args.multi_scale,
            mixup=args.mixup,
        )
        if args.lr0 is not None:
            train_kwargs["lr0"] = args.lr0
        model.train(**train_kwargs)
        return

    model = YOLO(args.model)
    pretrained: str | bool
    if args.weights.lower() == "auto":
        pretrained = infer_pretrained_weights(args.model)
    elif args.weights.lower() == "none":
        pretrained = False
    else:
        pretrained = args.weights

    train_kwargs = dict(
        trainer=VisDroneAttackTrainer,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        cache=args.cache,
        pretrained=pretrained,
        seed=args.seed,
        optimizer=args.optimizer,
        amp=args.amp,
        patience=args.patience,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        multi_scale=args.multi_scale,
        mixup=args.mixup,
        degrees=0.0,
        copy_paste=0.0,
        save_json=False,
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
