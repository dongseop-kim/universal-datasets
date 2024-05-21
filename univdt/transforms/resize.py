from typing import Any

import albumentations as A
import albumentations.augmentations.geometric.functional as gf
import albumentations.core.bbox_utils as bbox_utils
import cv2
import numpy as np


def random_resize(height: int, width: int, pad_val: int = 0, pad_val_mask: int = 255, p: float = 1.0):
    """
    Randomly select and apply a resize algorithm with the given size.
    The algorithm is chosen from A.Resize and Letterbox.

    Args:
        height (int): desired height of the output image.
        width (int): desired width of the output image.
        pad_val (int): padding value if border_mode is cv2.BORDER_CONSTANT. Default: 0.
        pad_val_mask (int): padding value for mask if border_mode is cv2.BORDER_CONSTANT. Default: 255.
        p (float): probability of applying the transform. Default: 1.0.
    """
    resize = A.Resize(height, width, cv2.INTER_LINEAR)
    letterbox = Letterbox(height, width, cv2.INTER_LINEAR, pad_val, pad_val_mask)
    return A.OneOf([resize, letterbox], p=p)


class Letterbox(A.DualTransform):
    """
    Resize the input to the specified height and width while maintaining the aspect ratio. 
    padding is added to the image to preserve the aspect ratio.

    Args:
        height (int): desired height of the output image.
        width (int): desired width of the output image.
        pad_val (int): padding value if border_mode is cv2.BORDER_CONSTANT. Default: 0.
        pad_val_mask (int): padding value for mask if border_mode is cv2.BORDER_CONSTANT. Default: 255.
        always_apply (bool): always apply the transform.
        p (float): probability of applying the transform. Default: 1.0.
    """

    def __init__(self, height: int, width: int,
                 pad_val: int = 0, pad_val_mask: int = 255,
                 always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
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
        h, w = params['rows'], params['cols']
        scale = min(self.width / w, self.height / h)
        target_h, target_w = int(h * scale), int(w * scale)

        pad_top, pad_bottom = self.get_pad_size(target_h, self.height)
        pad_left, pad_right = self.get_pad_size(target_w, self.width)

        params.update({'scale': scale, 'target_h': target_h, 'target_w': target_w,
                       'h_pad_top': pad_top, 'h_pad_bottom': pad_bottom,
                       'w_pad_left': pad_left, 'w_pad_right': pad_right})
        return params

    def get_transform_init_args_names(self):
        return ('height', 'width', 'pad_val', 'pad_val_mask')

    def apply(self, img: np.ndarray, interpolation=cv2.INTER_LINEAR, **params):
        img = gf.resize(img, params['target_h'], params['target_w'], interpolation)
        img = gf.pad_with_params(img=img,
                                 h_pad_top=params['h_pad_top'], h_pad_bottom=params['h_pad_bottom'],
                                 w_pad_left=params['w_pad_left'], w_pad_right=params['w_pad_right'],
                                 border_mode=cv2.BORDER_CONSTANT, value=self.pad_val)
        return img

    def apply_to_mask(self, img: np.ndarray, **params):
        img = gf.resize(img, params['target_h'], params['target_w'], cv2.INTER_NEAREST)
        img = gf.pad_with_params(img=img,
                                 h_pad_top=params['h_pad_top'], h_pad_bottom=params['h_pad_bottom'],
                                 w_pad_left=params['w_pad_left'], w_pad_right=params['w_pad_right'],
                                 border_mode=cv2.BORDER_CONSTANT, value=self.pad_val_mask)
        return img

    # NOTE: not sure if this is correct
    def apply_to_bbox(self, bbox, **params):
        # denormalize to target size
        h, w = params['target_h'], params['target_w']
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
        keypoint = gf.keypoint_scale(keypoint, scale_x, scale_y)
        x, y, angle, scale = keypoint
        x += params['w_pad_left']
        y += params['h_pad_top']
        return x, y, angle, scale
