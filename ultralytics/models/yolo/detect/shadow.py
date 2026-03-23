from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOCAL_RANK, LOGGER, RANK

from .visdrone import VisDroneAttackTrainer


class ShadowDistillTrainer(VisDroneAttackTrainer):
    """VisDrone trainer with a frozen dense teacher used only during training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = None

    @property
    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[4]

    def _shadow_cfg(self) -> dict:
        model = self.model.module if hasattr(self.model, "module") else self.model
        return getattr(model, "yaml", {}).get("shadow_distill", {}) or {}

    def _resolve_path(self, value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        if path.exists():
            return str(path)
        repo_path = self._repo_root / value
        if repo_path.exists():
            return str(repo_path)
        return value

    def _build_teacher(self):
        """Construct the frozen teacher model from cfg/weights arguments."""
        shadow_cfg = self._shadow_cfg()
        teacher_model = self._resolve_path(getattr(self.args, "teacher_model", None) or shadow_cfg.get("teacher_model"))
        if not teacher_model:
            return None

        teacher_weights = self._resolve_path(
            getattr(self.args, "teacher_weights", None) or shadow_cfg.get("teacher_weights", None)
        )
        cfg, weights = teacher_model, None
        if str(teacher_model).endswith(".pt"):
            weights, _ = load_checkpoint(teacher_model)
            cfg = weights.yaml
        elif teacher_weights:
            weights, _ = load_checkpoint(teacher_weights)

        model = self.get_model(cfg=cfg, weights=weights, verbose=False)
        model = model.to(self.device).eval()
        model.requires_grad_(False)
        for p in model.parameters():
            p.requires_grad = False
        model.model[-1].training = True  # return raw preds while keeping the rest of the network in eval mode
        LOGGER.info(
            f"Shadow distillation teacher ready: cfg={teacher_model}"
            + (f", weights={teacher_weights}" if teacher_weights else "")
        )
        return model

    def _setup_train(self):
        """Build the student via the standard path, then attach the frozen teacher."""
        super()._setup_train()
        LOGGER.info(
            "DDP runtime "
            f"rank={RANK} local_rank={LOCAL_RANK} world_size={self.world_size} "
            f"device={self.device} batch_per_rank={self.batch_size // max(self.world_size, 1)}"
        )
        if self.teacher_model is None:
            self.teacher_model = self._build_teacher()
        if self.teacher_model is not None:
            LOGGER.info(f"Teacher runtime rank={RANK} device={next(self.teacher_model.parameters()).device}")

    def preprocess_batch(self, batch: dict) -> dict:
        """Attach teacher predictions to the training batch for train-time-only distillation losses."""
        batch = super().preprocess_batch(batch)
        if self.teacher_model is None:
            return batch

        with torch.no_grad():
            teacher_preds = self.teacher_model(batch["img"])
        batch["_teacher_preds"] = teacher_preds
        return batch
