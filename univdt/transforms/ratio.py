import random
from typing import Any

from albumentations.augmentations.geometric import functional as geo_f
import albumentations.core.bbox_utils as bbox_utils
from albumentations.core.transforms_interface import DualTransform
import cv2
import numpy as np


class RandomRatio(DualTransform):
    """
    Apply specific ratio resize & padding
    ratio_range (Union[float, Tuple[float, float]]):
            If float, the range will be (ratio_range, 1).
            If tuple, it will be used as (min_ratio, max_ratio).
            All values should be between 0 and 1.
    """
    # DEFAULT_RATIOS = (0.66666, 0.785714, 0.823529, 0.833333)

    def __init__(self, height: int = 768, width: int = 768,
                 ratio_range: float | tuple[float, float] = 0.7,
                 interpolation: int = cv2.INTER_LINEAR,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.ratio_range = self._process_ratio_range(ratio_range)
        self.value = 0
        self.mask_value = 0
        self.interpolation = interpolation

    def _process_ratio_range(self, ratio_range: float | tuple[float, float]) -> tuple[float, float]:
        if isinstance(ratio_range, float):
            if not 0 <= ratio_range <= 1:
                raise ValueError("ratio_range must be between 0 and 1")
            return (ratio_range, 1.0)
        elif isinstance(ratio_range, (tuple, list)) and len(ratio_range) == 2:
            if not all(0 <= r <= 1 for r in ratio_range):
                raise ValueError("All values in ratio_range must be between 0 and 1")
            return tuple(ratio_range)
        else:
            raise ValueError("ratio_range must be a float or a tuple of two floats")

    def get_params(self) -> dict[str, float]:
        ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
        return {"ratio": ratio}

    def update_params(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        params = super().update_params(params, **kwargs)
        params.update(self._calculate_resize_and_pad_params(params))
        return params

    def _calculate_resize_and_pad_params(self, params: dict[str, Any]) -> dict[str, Any]:
        height, width = params["rows"], params["cols"]
        ratio = params["ratio"]

        if random.random() < 0.5:
            scale_h, scale_w = self.height / height * ratio, self.width / width
        else:
            scale_h, scale_w = self.height / height, self.width / width * ratio

        rows, cols = int(height * scale_h), int(width * scale_w)

        h_pad_top, h_pad_bottom = self._calculate_padding(self.height, rows)
        w_pad_left, w_pad_right = self._calculate_padding(self.width, cols)

        return {'rows': rows, 'cols': cols,
                'scale_h': scale_h, 'scale_w': scale_w,
                'pad_top': h_pad_top, 'pad_bottom': h_pad_bottom,
                'pad_left': w_pad_left, 'pad_right': w_pad_right}

    @ staticmethod
    def _calculate_padding(target_size: int, current_size: int) -> tuple[int, int]:
        if current_size < target_size:
            pad = int((target_size - current_size) / 2)
            return pad, target_size - current_size - pad
        return 0, 0

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        rows, cols = params['rows'], params['cols']
        pad_top, pad_bottom = params['pad_top'], params['pad_bottom']
        pad_left, pad_right = params['pad_left'], params['pad_right']
        img = geo_f.resize(img, rows, cols, self.interpolation)

        return geo_f.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right,
                                     border_mode=cv2.BORDER_CONSTANT,
                                     value=self.value)

    def apply_to_mask(self, img: np.ndarray, **params: Any) -> np.ndarray:
        rows, cols = params['rows'], params['cols']
        pad_top, pad_bottom = params['pad_top'], params['pad_bottom']
        pad_left, pad_right = params['pad_left'], params['pad_right']

        img = geo_f.resize(img, rows, cols, cv2.INTER_NEAREST)

        return geo_f.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right,
                                     border_mode=cv2.BORDER_CONSTANT, value=self.mask_value)

    def apply_to_bbox(self, bbox: tuple[float, float, float, float], **params: Any) -> tuple[float, float, float, float]:
        rows, cols = params['rows'], params['cols']
        pad_top, pad_bottom = params['pad_top'], params['pad_bottom']
        pad_left, pad_right = params['pad_left'], params['pad_right']
        x_min, y_min, x_max, y_max = bbox_utils.denormalize_bbox(bbox, rows, cols)
        bbox = x_min+pad_left, y_min+pad_top, x_max+pad_left, y_max+pad_top
        return bbox_utils.normalize_bbox(bbox, rows+pad_top+pad_bottom, cols+pad_left+pad_right)

    def apply_to_keypoint(self, keypoint: tuple[float, float, float, float], **params: Any) -> tuple[float, float, float, float]:
        x, y, angle, scale = keypoint
        rows, cols = params['rows'], params['cols']
        pad_top, pad_left = params['pad_top'], params['pad_left']

        x, y, angle, scale = geo_f.keypoint_scale((x, y, angle, scale), rows/self.height, cols/self.width)
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('height', 'width', 'ratio_range', )
