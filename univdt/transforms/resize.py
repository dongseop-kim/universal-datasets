from typing import Any, Dict

import albumentations as A
from albumentations import Resize
import albumentations.augmentations.geometric.functional as F
import albumentations.core.bbox_utils as bbox_utils
import cv2
import numpy as np


class Letterbox(A.DualTransform):
    def __init__(self, height: int, width: int, interpolation=cv2.INTER_LINEAR,
                 pad_cval: int = 0, pad_cval_mask: int = 255,
                 always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.pad_cval = pad_cval
        self.pad_cval_mask = pad_cval_mask
        self.interpolation = interpolation

    def apply(self, img: np.ndarray, interpolation=cv2.INTER_LINEAR, **params):
        img = F.resize(img, params['height'], params['width'], interpolation)
        img = F.pad_with_params(img=img,
                                h_pad_top=params['h_pad_top'], h_pad_bottom=params['h_pad_bottom'],
                                w_pad_left=params['w_pad_left'], w_pad_right=params['w_pad_right'],
                                border_mode=cv2.BORDER_CONSTANT, value=self.pad_cval)
        return img

    def apply_to_mask(self, img: np.ndarray, **params):
        img = F.resize(img, params['height'], params['width'], cv2.INTER_NEAREST)
        img = F.pad_with_params(img=img,
                                h_pad_top=params['h_pad_top'], h_pad_bottom=params['h_pad_bottom'],
                                w_pad_left=params['w_pad_left'], w_pad_right=params['w_pad_right'],
                                border_mode=cv2.BORDER_CONSTANT, value=self.pad_cval_mask)
        return img

    def apply_to_bbox(self, bbox, **params):
        # denormalize to target size
        h, w = params['height'], params['width']
        x_min, y_min, x_max, y_max = bbox_utils.denormalize_bbox(bbox, h, w)
        # add padding
        x_min, x_max = x_min + params['w_pad_left'], x_max + params['w_pad_left']
        y_min, y_max = y_min + params['h_pad_top'], y_max + params['h_pad_top']
        # normalize to padded size
        return bbox_utils.normalize_bbox((x_min, y_min, x_max, y_max),
                                         h+params['h_pad_top']+params['h_pad_bottom'],
                                         w+params['w_pad_left']+params['w_pad_right'])

    # NOTE: not sure if this is correct
    def apply_to_keypoint(self, keypoint, **params):
        scale_x: float = self.width / params['cols']
        scale_y: float = self.height / params['rows']
        keypoint = F.keypoint_scale(keypoint, scale_x, scale_y)
        x, y, angle, scale = keypoint
        x += params['w_pad_left']
        y += params['h_pad_top']
        return x, y, angle, scale

    def _get_pad_size(self, size: int, target_size: int):
        if size > target_size:
            return 0, 0
        else:
            pad = target_size - size
            pad_a = pad // 2
            pad_b = pad - pad_a
            return pad_a, pad_b

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        params: dict[str, Any] = super().update_params(params, **kwargs)
        h, w = params['rows'], params['cols']
        scale = min(self.width / w, self.height / h)
        new_w, new_h = int(params['cols'] * scale), int(params['rows'] * scale)

        pad_top, pad_bottom = self._get_pad_size(new_h, self.height)
        pad_left, pad_right = self._get_pad_size(new_w, self.width)

        params.update({'height': new_h, 'width': new_w, 'scale': scale,
                       'h_pad_top': pad_top, 'h_pad_bottom': pad_bottom,
                       'w_pad_left': pad_left, 'w_pad_right': pad_right})
        return params
