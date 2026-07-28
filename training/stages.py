"""Validate the two-stage cloud super-resolution training contract."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingStage:
    number: int
    name: str
    dataset_type: str
    synthetic_degradation: bool


_STAGES = {
    1: TrainingStage(
        number=1,
        name="paired-cross-sensor",
        dataset_type="realesrgan_paired",
        synthetic_degradation=False,
    ),
    2: TrainingStage(
        number=2,
        name="hr-synthetic-degradation",
        dataset_type="realesrgan",
        synthetic_degradation=True,
    ),
}


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    getter = getattr(mapping, "get", None)
    if getter is not None:
        return getter(key, default)
    return mapping[key] if key in mapping else default


def validate_training_stage(configs: Any) -> TrainingStage:
    """Return the configured stage or raise for an unsafe stage/data mismatch."""
    stage_number = int(_get(configs.train, "stage", 0))
    if stage_number not in _STAGES:
        raise ValueError("train.stage must be explicitly set to 1 or 2")

    stage = _STAGES[stage_number]
    dataset_type = str(configs.data.train.type)
    degradation_enabled = bool(_get(configs.degradation, "enabled", False))

    if dataset_type != stage.dataset_type:
        raise ValueError(
            f"Stage {stage.number} requires data.train.type={stage.dataset_type!r}, "
            f"but received {dataset_type!r}"
        )
    if degradation_enabled != stage.synthetic_degradation:
        raise ValueError(
            f"Stage {stage.number} requires degradation.enabled="
            f"{stage.synthetic_degradation}, but received {degradation_enabled}"
        )

    if stage.number == 1:
        params = configs.data.train.params
        enabled_spatial = [
            name
            for name in ("use_hflip", "use_rot", "random_crop")
            if bool(_get(params, name, False))
        ]
        if enabled_spatial:
            options = ", ".join(enabled_spatial)
            raise ValueError(
                "Stage 1 must preserve matched pairs without random spatial "
                f"augmentation; disable: {options}"
            )

    return stage
