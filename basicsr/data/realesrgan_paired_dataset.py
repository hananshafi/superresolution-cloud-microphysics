import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import io
from PIL import Image
import numpy as np


def _decode_rgb_float(content, size=None):
    """Decode uint8/uint16 imagery as RGB float32 without per-image scaling."""
    with Image.open(io.BytesIO(content)) as image:
        if size is not None:
            image = image.resize(size, Image.Resampling.BICUBIC)

        if image.mode.startswith("I") or image.mode == "F":
            image_array = np.asarray(image)
            if image_array.ndim == 2:
                image_array = np.repeat(image_array[:, :, None], 3, axis=2)
        else:
            image_array = np.asarray(image.convert("RGB"))

    if np.issubdtype(image_array.dtype, np.integer):
        if image_array.dtype == np.uint8:
            denominator = 255.0
        elif image_array.dtype == np.uint16:
            denominator = 65535.0
        elif image_array.min() >= 0 and image_array.max() <= 65535:
            denominator = 65535.0
        else:
            denominator = float(np.iinfo(image_array.dtype).max)
        image_array = image_array.astype(np.float32) / denominator
    else:
        image_array = image_array.astype(np.float32)

    return np.ascontiguousarray(np.clip(image_array, 0.0, 1.0))


def _paired_center_crop(img_gt, img_lq, gt_size, scale, gt_path):
    """Center-crop aligned GT/LQ arrays using scale-aligned GT coordinates."""
    height, width = img_gt.shape[:2]
    if height < gt_size or width < gt_size:
        raise ValueError(
            f"GT image {gt_path} is smaller than gt_size={gt_size}: "
            f"{width}x{height}"
        )

    top = ((height - gt_size) // 2 // scale) * scale
    left = ((width - gt_size) // 2 // scale) * scale
    lq_size = gt_size // scale
    img_gt = img_gt[top : top + gt_size, left : left + gt_size]
    img_lq = img_lq[
        top // scale : top // scale + lq_size,
        left // scale : left // scale + lq_size,
    ]
    return img_gt, img_lq


@DATASET_REGISTRY.register(suffix='basicsr')
class RealESRGANPairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self,opt, mode='training'):
        super(RealESRGANPairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip() for line in fin]
            self.paths = []
            for path in paths:
                gt_path, lq_path = path.split(', ')
                gt_path = os.path.join(self.gt_folder, gt_path)
                lq_path = os.path.join(self.lq_folder, lq_path)
                self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path)]))
        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        self.mode = mode

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = int(self.opt.get('scale', 4))

        # Load gt and lq images. Dimension order: HWC; channel order: RGB;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = _decode_rgb_float(img_bytes)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        lq_size = (img_gt.shape[1] // scale, img_gt.shape[0] // scale)
        img_lq = _decode_rgb_float(img_bytes, size=lq_size)

        # Stage 1 configs disable random crop, flip, and rotation.
        if self.mode == 'training':
            gt_size = self.opt['gt_size']
            if self.opt.get('random_crop', False):
                img_gt, img_lq = paired_random_crop(
                    img_gt, img_lq, gt_size, scale, gt_path
                )
            else:
                img_gt, img_lq = _paired_center_crop(
                    img_gt, img_lq, gt_size, scale, gt_path
                )
            if self.opt.get('use_hflip', False) or self.opt.get('use_rot', False):
                img_gt, img_lq = augment(
                    [img_gt, img_lq],
                    self.opt.get('use_hflip', False),
                    self.opt.get('use_rot', False),
                )

        # HWC to CHW, numpy to tensor. Arrays are already RGB.
        img_gt, img_lq = img2tensor(
            [img_gt, img_lq], bgr2rgb=False, float32=True
        )
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'txt': ""}

    def __len__(self):
        return len(self.paths)
