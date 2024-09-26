import random
from typing import Any

import albumentations.augmentations.crops.functional as cf
import albumentations.augmentations.geometric.functional as gf
import albumentations.core.bbox_utils as bbox_utils
import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform

DEFAULT_PAD_VAL = 0
DEFAULT_PAD_VAL_MASK = 255
DEFAULT_PROBABILITY = 1.0


def random_zoom(scale: float | tuple[float, float] = 0.1,
                pad_val: int = 0, pad_val_mask: int = 255, p: float = 1.0) -> 'RandomZoom':
    return RandomZoom(scale, pad_val, pad_val_mask, p=p)


class RandomZoom(DualTransform):
    """
    Randomly zoom the input. Output image size is same as input image size.
    If scale is a single float value, the range will be (1-scale, 1+scale).
    If scale < 1.0, the image will be zoomed out with padding.
    If scale > 1.0, the image will be zoomed in with random cropping.

    Args:
        scale (float | tuple[float, float]): Scale factor range for random resize.
        pad_val (int): Padding value for image if zooming out.
        pad_val_mask (int): Padding value for mask if zooming out.
        always_apply (bool): Always apply the transform.
        p (float): Probability of applying the transform. Default: 1.0.
    """

    def __init__(self, scale: float | tuple[float, float] = 0.1,
                 pad_val: int = DEFAULT_PAD_VAL, pad_val_mask: int = DEFAULT_PAD_VAL_MASK,
                 always_apply: bool = False, p: float = DEFAULT_PROBABILITY):
        super().__init__(always_apply, p)
        self.scale = scale if isinstance(scale, tuple) else (1-scale, 1+scale)
        self.pad_val = pad_val
        self.pad_val_mask = pad_val_mask

    def get_params(self) -> dict[str, Any]:
        return {'scale': random.uniform(self.scale[0], self.scale[1])}

    def get_pad_size(self, size: int, target_size: int) -> tuple[int, int]:
        if size > target_size:
            return 0, 0
        pad = target_size - size
        pad_a = random.randint(0, pad)  # Randomly set padding start position
        pad_b = pad - pad_a
        return pad_a, pad_b

    def update_params(self, params: dict[str, Any], **kwargs) -> dict[str, Any]:
        params = super().update_params(params, **kwargs)
        scale = params['scale']
        target_h = int(params['rows'] * scale)
        target_w = int(params['cols'] * scale)
        params.update({'target_h': target_h, 'target_w': target_w})

        if scale < 1.0:
            # Set padding options for zoom out
            params.update({'h_pad_top': self.get_pad_size(target_h, params['rows'])[0],
                           'h_pad_bottom': self.get_pad_size(target_h, params['rows'])[1],
                           'w_pad_left': self.get_pad_size(target_w, params['cols'])[0],
                           'w_pad_right': self.get_pad_size(target_w, params['cols'])[1]
                           })
        else:
            # Set cropping options for zoom in
            params.update({'h_crop_start': random.random(), 'w_crop_start': random.random()})
        return params

    def apply(self, img: np.ndarray, interpolation: int = cv2.INTER_LINEAR, **params) -> np.ndarray:
        return self._apply_zoom(img, interpolation, **params)

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return self._apply_zoom(img, cv2.INTER_NEAREST, **params)

    def _apply_zoom(self, img: np.ndarray, interpolation: int, **params) -> np.ndarray:
        img = gf.resize(img, params['target_h'], params['target_w'], interpolation)
        if params['scale'] < 1.0:
            return gf.pad_with_params(img=img,
                                      h_pad_top=params['h_pad_top'], h_pad_bottom=params['h_pad_bottom'],
                                      w_pad_left=params['w_pad_left'], w_pad_right=params['w_pad_right'],
                                      border_mode=cv2.BORDER_CONSTANT,
                                      value=self.pad_val if interpolation != cv2.INTER_NEAREST else self.pad_val_mask)
        return cf.random_crop(img, params['rows'], params['cols'], params['h_crop_start'], params['w_crop_start'])

    def apply_to_bbox(self, bbox: tuple[float, float, float, float], **params) -> tuple[float, float, float, float]:
        x_min, y_min, x_max, y_max = bbox_utils.denormalize_bbox(bbox, params['target_h'], params['target_w'])
        if params['scale'] < 1.0:
            # Add padding
            x_min, x_max = x_min + params['w_pad_left'], x_max + params['w_pad_left']
            y_min, y_max = y_min + params['h_pad_top'], y_max + params['h_pad_top']
            return bbox_utils.normalize_bbox((x_min, y_min, x_max, y_max),
                                             params['rows'], params['cols'])
        # Random crop image for zoom in
        return cf.bbox_random_crop((x_min, y_min, x_max, y_max),
                                   params['rows'], params['cols'],
                                   params['h_crop_start'], params['w_crop_start'])

    def apply_to_keypoint(self, keypoint: tuple[float, float, float, float], **params) -> tuple[float, float, float, float]:
        x, y, angle, scale = keypoint
        rows, cols = params['rows'], params['cols']
        zoom_scale = params['scale']

        # Adjust keypoint position based on zoom scale
        x *= zoom_scale
        y *= zoom_scale

        if zoom_scale < 1.0:  # Zoom out (padding)
            # Adjust for padding
            x += params['w_pad_left']
            y += params['h_pad_top']
        else:  # Zoom in (cropping)
            # Adjust for cropping
            crop_x = int(cols * params['w_crop_start'])
            crop_y = int(rows * params['h_crop_start'])
            x -= crop_x
            y -= crop_y

        # Ensure the keypoint is within the image boundaries
        x = max(0, min(x, cols - 1))
        y = max(0, min(y, rows - 1))

        # Normalize the coordinates
        x /= cols
        y /= rows

        return (x, y, angle, scale)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('scale', 'pad_val', 'pad_val_mask')
