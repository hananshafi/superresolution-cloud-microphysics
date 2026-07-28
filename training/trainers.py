"""Explicit trainers for the two-stage cloud super-resolution curriculum."""

from trainer_orig import TrainerSDTurboSR as SyntheticDegradationTrainer
from trainer_paired import TrainerSDTurboSR as PairedTrainer
from training.stages import validate_training_stage


class Stage1Trainer(PairedTrainer):
    """Train on real cross-sensor pairs without synthetic augmentation."""

    def __init__(self, configs):
        self.training_stage = validate_training_stage(configs)
        if self.training_stage.number != 1:
            raise ValueError("Stage1Trainer requires train.stage=1")
        super().__init__(configs)


class Stage2Trainer(SyntheticDegradationTrainer):
    """Fine-tune the Stage 1 prior using HR-only synthetic degradations."""

    def __init__(self, configs):
        self.training_stage = validate_training_stage(configs)
        if self.training_stage.number != 2:
            raise ValueError("Stage2Trainer requires train.stage=2")
        super().__init__(configs)
