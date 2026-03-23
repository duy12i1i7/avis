from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect import ShadowDistillTrainer  # noqa: E402


def infer_pretrained_weights(model_path: str) -> str:
    """Map a custom YAML name to the closest official YOLO26 pretrained checkpoint."""
    name = Path(model_path).name.lower()
    for scale in ("x", "l", "m", "s", "n"):
        if f"yolo26{scale}" in name:
            return f"yolo26{scale}.pt"
    return "yolo26n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VisDrone YOLO26 attack recipe.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26n-spd-visdrone.yaml"),
        help="Path to the custom model YAML.",
    )
    parser.add_argument(
        "--weights",
        default="auto",
        help="Pretrained weights path, 'auto' to infer from model name, or 'none' to train from scratch.",
    )
    parser.add_argument("--data", default="VisDrone.yaml", help="Dataset YAML. Defaults to Ultralytics VisDrone config.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs" / "visdrone"))
    parser.add_argument("--name", default="yolo26_spd_visdrone_attack")
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--multi-scale", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--teacher-model", default=None, help="Optional teacher model YAML/PT for shadow distillation.")
    parser.add_argument("--teacher-weights", default=None, help="Optional teacher weights checkpoint.")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM for faster training.")
    parser.add_argument("--resume", default=None, help="Resume from a previous checkpoint if provided.")
    parser.add_argument("--cos-lr", dest="cos_lr", action="store_true", help="Enable cosine LR schedule.")
    parser.add_argument("--no-cos-lr", dest="cos_lr", action="store_false", help="Disable cosine LR schedule.")
    parser.set_defaults(cos_lr=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume:
        model = YOLO(args.resume)
        model.train(
            trainer=ShadowDistillTrainer,
            resume=True,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            cache=args.cache,
            patience=args.patience,
            teacher_model=args.teacher_model,
            teacher_weights=args.teacher_weights,
        )
        return

    model = YOLO(args.model)
    pretrained: str | bool
    if args.weights.lower() == "auto":
        pretrained = infer_pretrained_weights(args.model)
    elif args.weights.lower() == "none":
        pretrained = False
    else:
        pretrained = args.weights

    model.train(
        trainer=ShadowDistillTrainer,
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
        optimizer=args.optimizer,
        patience=args.patience,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        multi_scale=args.multi_scale,
        mixup=args.mixup,
        teacher_model=args.teacher_model,
        teacher_weights=args.teacher_weights,
        degrees=0.0,
        copy_paste=0.0,
        save_json=False,
    )


if __name__ == "__main__":
    main()
