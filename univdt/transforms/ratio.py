import logging
import random
from typing import Any

import cv2
import numpy as np
from albumentations.augmentations.geometric.functional import resize, pad_with_params, keypoints_scale
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from albumentations.core.transforms_interface import DualTransform

logger = logging.getLogger(__name__)


class RandomRatio(DualTransform):
    """"
    Randomly resize the input image and mask with a scale ratio applied asymmetrically to height or width,
    followed by padding to maintain a fixed output size.

    This transform simulates variable aspect ratios while preserving the final dimensions. It is useful for
    training models to be robust against objects of varying scales and shapes.

    Args:
        height (int): Target output height after padding.
        width (int): Target output width after padding.
        ratio_range (float | tuple[float, float]): If float, treated as (ratio_range, 1.0); range of resize ratio.
                                                   Must be between 0 and 1.
        interpolation (int): Interpolation method for resizing the image (e.g., cv2.INTER_LINEAR).
        pad_val (int): Padding value for images.
        pad_val_mask (int): Padding value for masks.
        debug (bool): If True, enables verbose logging for debugging.
        p (float): Probability of applying the transform.
    """

    def __init__(self,
                 height: int = 768, width: int = 768,
                 ratio_range: float | tuple[float, float] = 0.7,
                 interpolation: int = cv2.INTER_LINEAR,
                 pad_val: int = 0, pad_val_mask: int = 0,
                 debug: bool = False,
                 p: float = 0.5):
        super().__init__(p)
        self.height = height
        self.width = width
        self.ratio_range = self._process_ratio_range(ratio_range)
        self.interpolation = interpolation
        self.pad_val = pad_val
        self.pad_val_mask = pad_val_mask
        self.debug = debug

    def _process_ratio_range(self, ratio_range: float | tuple[float, float]) -> tuple[float, float]:
        if isinstance(ratio_range, float):
            return (ratio_range, 1.0)
        elif isinstance(ratio_range, (tuple, list)) and len(ratio_range) == 2:
            return tuple(ratio_range)
        else:
            raise ValueError("ratio_range must be float or tuple of two floats between 0 and 1")

    def get_params(self) -> dict[str, Any]:
        ratio = random.uniform(*self.ratio_range)
        return {"ratio": ratio}

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image: np.ndarray = data["image"]
        h, w = image.shape[:2]
        ratio = params["ratio"]

        if random.random() < 0.5:
            scale_h, scale_w = self.height / h * ratio, self.width / w
        else:
            scale_h, scale_w = self.height / h, self.width / w * ratio

        new_h = int(h * scale_h)
        new_w = int(w * scale_w)

        pad_top, pad_bottom = self._calculate_padding(self.height, new_h)
        pad_left, pad_right = self._calculate_padding(self.width, new_w)

        if self.debug:
            logger.debug(f"[RandomRatio] Original (h, w): ({h}, {w})")
            logger.debug(f"[RandomRatio] Resize to (new_h, new_w): ({new_h}, {new_w})")
            logger.debug(
                f"[RandomRatio] Padding - top:{pad_top}, bottom:{pad_bottom}, left:{pad_left}, right:{pad_right}")

        return {'new_h': new_h, 'new_w': new_w,
                'pad_top': pad_top, 'pad_bottom': pad_bottom,
                'pad_left': pad_left, 'pad_right': pad_right,
                'rows': h, 'cols': w,
                'shape': image.shape}

    def _calculate_padding(self, target: int, current: int) -> tuple[int, int]:
        pad = max(0, (target - current) // 2)
        return pad, target - current - pad

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        resized = resize(img, (params['new_h'], params['new_w']), interpolation=self.interpolation)
        return pad_with_params(resized,
                               params['pad_top'], params['pad_bottom'],
                               params['pad_left'], params['pad_right'],
                               border_mode=cv2.BORDER_CONSTANT, value=self.pad_val)

    def apply_to_mask(self, img: np.ndarray, **params: Any) -> np.ndarray:
        resized = resize(img, (params['new_h'], params['new_w']), interpolation=cv2.INTER_NEAREST)
        return pad_with_params(resized,
                               params['pad_top'], params['pad_bottom'],
                               params['pad_left'], params['pad_right'],
                               border_mode=cv2.BORDER_CONSTANT, value=self.pad_val_mask)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        if len(bboxes) == 0:
            return bboxes
        denorm = denormalize_bboxes(bboxes, (params['rows'], params['cols']))
        denorm[:, [0, 2]] *= params['new_w'] / params['cols']
        denorm[:, [1, 3]] *= params['new_h'] / params['rows']
        denorm[:, [0, 2]] += params['pad_left']
        denorm[:, [1, 3]] += params['pad_top']
        return normalize_bboxes(denorm, (self.height, self.width))

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        if len(keypoints) == 0:
            return keypoints
        keypoints = np.asarray(keypoints)
        rows, cols = params['rows'], params['cols']
        scaled = keypoints_scale(keypoints, params['new_w'] / cols, params['new_h'] / rows)
        scaled[:, 0] += params['pad_left']
        scaled[:, 1] += params['pad_top']
        return scaled

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('height', 'width', 'ratio_range', 'interpolation', 'pad_val', 'pad_val_mask', 'debug')

    def __call__(self, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        if self.debug:
            logger.debug(f"RandomRatio.__call__ invoked with force_apply={force_apply}")
            if 'image' in data:
                logger.debug(f"Input image shape: {data['image'].shape}")
        return super(RandomRatio, self).__call__(force_apply=force_apply, **data)
