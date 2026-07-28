from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from basicsr.data.realesrgan_paired_dataset import RealESRGANPairedDataset


def _dataset_options(lq_dir, gt_dir):
    return {
        "dataroot_lq": str(lq_dir),
        "dataroot_gt": str(gt_dir),
        "io_backend": {"type": "disk"},
        "gt_size": 8,
        "scale": 4,
        "use_hflip": False,
        "use_rot": False,
        "random_crop": False,
    }


class PairedDatasetTests(unittest.TestCase):
    def test_uint16_pairs_keep_dynamic_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lq_dir = root / "lq"
            gt_dir = root / "gt"
            lq_dir.mkdir()
            gt_dir.mkdir()

            values = np.linspace(0, 65535, 64, dtype=np.uint16).reshape(8, 8)
            Image.fromarray(values).save(lq_dir / "msg_1.png")
            Image.fromarray(values).save(gt_dir / "mtg_1.png")

            dataset = RealESRGANPairedDataset(
                _dataset_options(lq_dir, gt_dir)
            )
            first = dataset[0]
            second = dataset[0]

            self.assertEqual(first["lq"].shape, (3, 2, 2))
            self.assertEqual(first["gt"].shape, (3, 8, 8))
            self.assertLess(first["lq"].min(), first["lq"].max())
            self.assertEqual(first["gt"].min(), 0)
            self.assertEqual(first["gt"].max(), 1)
            self.assertTrue(torch.equal(first["lq"], second["lq"]))
            self.assertTrue(torch.equal(first["gt"], second["gt"]))

    def test_rgb_channel_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lq_dir = root / "lq"
            gt_dir = root / "gt"
            lq_dir.mkdir()
            gt_dir.mkdir()

            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            rgb[..., 0] = 10
            rgb[..., 1] = 20
            rgb[..., 2] = 30
            Image.fromarray(rgb, mode="RGB").save(lq_dir / "seviri_1.png")
            Image.fromarray(rgb, mode="RGB").save(gt_dir / "viirs_1.png")

            sample = RealESRGANPairedDataset(
                _dataset_options(lq_dir, gt_dir)
            )[0]
            means = sample["gt"].mean(dim=(1, 2))

            self.assertTrue(
                torch.allclose(means, torch.tensor([10, 20, 30]) / 255)
            )


if __name__ == "__main__":
    unittest.main()
