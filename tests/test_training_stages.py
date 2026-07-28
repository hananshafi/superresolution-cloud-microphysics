from pathlib import Path
import unittest

from omegaconf import OmegaConf

from training.stages import validate_training_stage


ROOT = Path(__file__).resolve().parents[1]


class TrainingStageTests(unittest.TestCase):
    def test_training_configs_have_safe_stage_contract(self):
        cases = (
            ("sd_turbo-sr-ldis-pairwise.yaml", 1, False),
            ("sd_turbo-sr-ldis-pairwise-msg-mtg.yaml", 1, False),
            ("sd-turbo-sr-ldis.yaml", 2, True),
            ("sd-turbo-sr-ldis-msg-mtg.yaml", 2, True),
        )
        for config_name, expected_stage, expected_degradation in cases:
            with self.subTest(config=config_name):
                configs = OmegaConf.load(ROOT / "configs" / config_name)
                stage = validate_training_stage(configs)
                self.assertEqual(stage.number, expected_stage)
                self.assertIs(
                    stage.synthetic_degradation, expected_degradation
                )

    def test_stage1_rejects_synthetic_degradation(self):
        configs = OmegaConf.load(
            ROOT / "configs" / "sd_turbo-sr-ldis-pairwise.yaml"
        )
        configs.degradation.enabled = True

        with self.assertRaisesRegex(ValueError, "degradation.enabled=False"):
            validate_training_stage(configs)

    def test_stage1_rejects_random_spatial_augmentation(self):
        configs = OmegaConf.load(
            ROOT / "configs" / "sd_turbo-sr-ldis-pairwise.yaml"
        )
        configs.data.train.params.use_hflip = True

        with self.assertRaisesRegex(ValueError, "disable: use_hflip"):
            validate_training_stage(configs)


if __name__ == "__main__":
    unittest.main()
