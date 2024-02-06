import random
from typing import Any

import albumentations.augmentations.crops.functional as cf
import albumentations.augmentations.geometric.functional as gf
import albumentations.core.bbox_utils as bbox_utils
import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform


def random_zoom(scale: float | tuple[float, float] = 0.1,
                pad_val: int = 0, pad_val_mask: int = 255, p=1.0):
    return RandomZoom(scale, pad_val, pad_val_mask, p=p)


class RandomZoom(DualTransform):
    """
    Randomly zoom the input. output image size is same as input image size.
    If scale is a single float value, the range will be (1-scale, 1+scale).
    if scale < 1.0, the image will be zoomed out with padding.
    if scale > 1.0, the image will be zoomed in with cropping.

    Args:
        scale (float | tuple[float, float]): scale factor range for random resize. 

        always_apply (bool): always apply the transform.
        p (float): probability of applying the transform. Default: 1.0.
    """

    def __init__(self, scale: float | tuple[float, float] = 0.1,
                 pad_val: int = 0, pad_val_mask: int = 255,
                 always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.scale = scale if isinstance(scale, tuple) else (1-scale, 1+scale)
        self.pad_val = pad_val
        self.pad_val_mask = pad_val_mask

    def get_pad_size(self, size: int, target_size: int):
        if size > target_size:
            return 0, 0
        else:
            pad = target_size - size
            pad_a = pad // 2
            pad_b = pad - pad_a
            return pad_a, pad_b

    def update_params(self, params: dict[str, Any], **kwargs) -> dict[str, Any]:
        params: dict[str, Any] = super().update_params(params, **kwargs)
        scale = random.uniform(self.scale[0], self.scale[1])
        target_h = int(params['rows'] * scale)
        target_w = int(params['cols'] * scale)
        params.update({'scale': scale, 'target_h': target_h, 'target_w': target_w})
        if scale < 1.0:
            # set padding options for zoom out
            pad_top, pad_bottom = self.get_pad_size(target_h, params['rows'], )
            pad_left, pad_right = self.get_pad_size(target_w, params['cols'])
            params.update({'h_pad_top': pad_top, 'h_pad_bottom': pad_bottom,
                           'w_pad_left': pad_left, 'w_pad_right': pad_right})
        else:
            # set cropping options for zoom in
            h_crop_start = random.random()
            w_crop_start = random.random()
            params.update({'h_crop_start': h_crop_start, 'w_crop_start': w_crop_start})
        return params

    def get_transform_init_args_names(self):
        return ('scale', 'pad_val', 'pad_val_mask')

    def apply(self, img: np.ndarray, interpolation=cv2.INTER_LINEAR,  **params):
        # resize image with scale factor
        scale, target_h, target_w = params['scale'], params['target_h'], params['target_w']
        img = gf.resize(img, target_h, target_w, interpolation)
        if scale < 1.0:
            # pad image for zoom out
            img = gf.pad_with_params(img=img,
                                     h_pad_top=params['h_pad_top'], h_pad_bottom=params['h_pad_bottom'],
                                     w_pad_left=params['w_pad_left'], w_pad_right=params['w_pad_right'],
                                     border_mode=cv2.BORDER_CONSTANT, value=self.pad_val)
        else:
            # crop image for zoom in
            img = cf.random_crop(img, params['rows'], params['cols'],
                                 params['h_crop_start'], params['w_crop_start'])
        return img

    def apply_to_mask(self, img: np.ndarray, **params):
        # resize image with scale factor
        scale, target_h, target_w = params['scale'], params['target_h'], params['target_w']
        img = gf.resize(img, target_h, target_w, cv2.INTER_NEAREST)

        if scale < 1.0:
            # pad image for zoom out
            img = gf.pad_with_params(img=img,
                                     h_pad_top=params['h_pad_top'], h_pad_bottom=params['h_pad_bottom'],
                                     w_pad_left=params['w_pad_left'], w_pad_right=params['w_pad_right'],
                                     border_mode=cv2.BORDER_CONSTANT, value=self.pad_val_mask)
        else:
            # crop image for zoom in
            img = cf.random_crop(img, params['rows'], params['cols'],
                                 params['h_crop_start'], params['w_crop_start'])
        return img

    # NOTE: not sure if this is correct
    def apply_to_bbox(self, bbox, **params):
        scale, target_h, target_w = params['scale'], params['target_h'], params['target_w']
        x_min, y_min, x_max, y_max = bbox_utils.denormalize_bbox(bbox, target_h, target_w)
        if scale < 1.0:
            # add padding
            x_min, x_max = x_min + params['w_pad_left'], x_max + params['w_pad_left']
            y_min, y_max = y_min + params['h_pad_top'], y_max + params['h_pad_top']
            return bbox_utils.normalize_bbox((x_min, y_min, x_max, y_max),
                                             params['rows'], params['cols'])
        else:
            # random crop image for zoom in
            return cf.bbox_random_crop((x_min, y_min, x_max, y_max),
                                       params['rows'], params['cols'],
                                       params['h_crop_start'], params['w_crop_start'])

    # TODO: implement this
    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError
