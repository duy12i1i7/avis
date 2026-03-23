from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ultralytics.data import build_dataloader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class _IndexedDataset(Dataset):
    """Dataset wrapper that repeats selected samples without touching the base dataset cache."""

    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices
        if hasattr(dataset, "collate_fn"):
            self.collate_fn = dataset.collate_fn

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def close_mosaic(self, hyp) -> None:
        if hasattr(self.dataset, "close_mosaic"):
            self.dataset.close_mosaic(hyp)

    @property
    def mosaic(self):
        return getattr(self.dataset, "mosaic", None)

    @mosaic.setter
    def mosaic(self, value) -> None:
        if hasattr(self.dataset, "mosaic"):
            self.dataset.mosaic = value

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)


class VisDroneAttackTrainer(DetectionTrainer):
    """Detection trainer with train-time only tricks for tiny-human-heavy VisDrone runs."""

    def _attack_cfg(self) -> dict:
        model = unwrap_model(self.model)
        return getattr(model, "yaml", {}).get("visdrone_attack", {}) or {}

    def _tiny_cls_ids(self) -> np.ndarray:
        return np.asarray(self._attack_cfg().get("tiny_cls_ids", [0, 1]), dtype=np.int64)

    def _match_tiny_objects(
        self,
        classes: np.ndarray,
        boxes_xywh: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Match tiny target objects using original-image pixel thresholds."""
        if classes.size == 0 or boxes_xywh.size == 0:
            return np.zeros(0, dtype=bool)

        cfg = self._attack_cfg()
        cls_ids = self._tiny_cls_ids()
        h0, w0 = image_shape
        widths = boxes_xywh[:, 2] * float(w0)
        heights = boxes_xywh[:, 3] * float(h0)
        areas = widths * heights
        return (
            np.isin(classes, cls_ids)
            & (areas <= float(cfg.get("tiny_area", 32.0 * 32.0)))
            & (heights <= float(cfg.get("tiny_height", 20.0)))
            & (areas > 0.0)
        )

    def _maybe_wrap_oversampled_dataset(self, dataset):
        cfg = self._attack_cfg()
        base_repeat = int(cfg.get("tiny_repeat", 0))
        repeat_cap = int(cfg.get("tiny_repeat_cap", base_repeat))
        if base_repeat <= 0 or repeat_cap < 0:
            return dataset

        indices: list[int] = []
        boosted = 0
        for i, label in enumerate(dataset.labels):
            classes = label["cls"].reshape(-1).astype(np.int64, copy=False)
            boxes = label["bboxes"]
            tiny_mask = self._match_tiny_objects(classes, boxes, label["shape"])
            tiny_count = int(tiny_mask.sum())
            extra = min(base_repeat + max(tiny_count - 1, 0) // 4, repeat_cap) if tiny_count else 0
            indices.extend([i] * (1 + extra))
            boosted += int(extra > 0)

        if len(indices) == len(dataset):
            return dataset

        LOGGER.info(
            "VisDrone tiny-image oversampling expanded train set from "
            f"{len(dataset)} to {len(indices)} samples ({boosted} images boosted)."
        )
        return _IndexedDataset(dataset, indices)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Build a dataloader and optionally oversample tiny-human-heavy images for training."""
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        if mode == "train":
            dataset = self._maybe_wrap_oversampled_dataset(dataset)

        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False

        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def _batch_has_tiny_targets(self, batch: dict) -> bool:
        """Check whether a formatted batch contains configured tiny-human targets."""
        classes = batch["cls"].view(-1)
        boxes = batch["bboxes"].view(-1, 4)
        if classes.numel() == 0 or boxes.numel() == 0:
            return False

        cfg = self._attack_cfg()
        img_h, img_w = batch["img"].shape[-2:]
        area_thr = float(cfg.get("tiny_area", 32.0 * 32.0)) / max(float(img_h * img_w), 1.0)
        height_thr = float(cfg.get("tiny_height", 20.0)) / max(float(img_h), 1.0)
        cls_mask = torch.zeros_like(classes, dtype=torch.bool)
        for cls_id in self._tiny_cls_ids().tolist():
            cls_mask |= classes.eq(float(cls_id))
        areas = boxes[:, 2] * boxes[:, 3]
        return bool((cls_mask & (areas <= area_thr) & (boxes[:, 3] <= height_thr) & areas.gt(0)).any())

    def preprocess_batch(self, batch: dict) -> dict:
        """Bias train-time resolution upward for batches that contain tiny humans."""
        batch = super().preprocess_batch(batch)
        cfg = self._attack_cfg()
        scale_prob = float(cfg.get("tiny_scale_prob", 0.0))
        scale_gain = float(cfg.get("tiny_scale_gain", 1.0))
        max_gain = float(cfg.get("max_scale_gain", scale_gain))
        if scale_prob <= 0.0 or scale_gain <= 1.0 or random.random() >= scale_prob:
            return batch
        if not self._batch_has_tiny_targets(batch):
            return batch

        gain = random.uniform(1.0, scale_gain)
        current = int(max(batch["img"].shape[-2:]))
        target = int(round(current * gain / self.stride) * self.stride)
        target_cap = int(round(self.args.imgsz * max_gain / self.stride) * self.stride)
        target = min(max(target, current), max(target_cap, current))
        if target == current:
            return batch

        batch["img"] = F.interpolate(batch["img"], size=(target, target), mode="bilinear", align_corners=False)
        return batch
