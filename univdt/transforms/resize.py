from typing import Any

import albumentations as A
import albumentations.augmentations.geometric.functional as gf
import albumentations.core.bbox_utils as bbox_utils
import cv2
import numpy as np

DEFAULT_PAD_VAL = 0
DEFAULT_PAD_VAL_MASK = 255
DEFAULT_PROBABILITY = 1.0


def random_resize(height: int, width: int, pad_val: int = DEFAULT_PAD_VAL,
                  pad_val_mask: int = DEFAULT_PAD_VAL_MASK,
                  p: float = DEFAULT_PROBABILITY) -> A.OneOf:
    """
    Randomly select and apply a resize algorithm with the given size.

    Args:
        height (int): Desired height of the output image.
        width (int): Desired width of the output image.
        pad_val (int): Padding value for image if border_mode is cv2.BORDER_CONSTANT.
        pad_val_mask (int): Padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        p (float): Probability of applying the transform.

    Returns:
        A.OneOf: Albumentations OneOf transform with Resize and Letterbox.
    """
    resize = A.Resize(height, width, interpolation=cv2.INTER_LINEAR)
    letterbox = Letterbox(height, width, pad_val=pad_val, pad_val_mask=pad_val_mask)
    return A.OneOf([resize, letterbox], p=p)


class Letterbox(A.DualTransform):
    """
    Resize the input to the specified height and width while maintaining the aspect ratio.
    Padding is added to the image to preserve the aspect ratio.

    Args:
        height (int): Desired height of the output image.
        width (int): Desired width of the output image.
        pad_val (int): Padding value for image if border_mode is cv2.BORDER_CONSTANT.
        pad_val_mask (int): Padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        always_apply (bool): Whether to always apply the transform.
        p (float): Probability of applying the transform.
    """

    def __init__(self, height: int, width: int,
                 pad_val: int = DEFAULT_PAD_VAL,
                 pad_val_mask: int = DEFAULT_PAD_VAL_MASK,
                 always_apply: bool = False,
                 p: float = DEFAULT_PROBABILITY):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.pad_val = pad_val
        self.pad_val_mask = pad_val_mask

    @staticmethod
    def get_pad_size(size: int, target_size: int) -> tuple[int, int]:
        """Calculate padding size for a given dimension."""
        if size > target_size:
            return 0, 0
        pad = target_size - size
        pad_a, pad_b = pad // 2, pad - pad // 2
        return pad_a, pad_b

    def update_params(self, params: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Update transform parameters."""
        params = super().update_params(params, **kwargs)
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

    def apply(self, img: np.ndarray, interpolation: int = cv2.INTER_LINEAR, **params) -> np.ndarray:
        """Apply the transform to an image."""
        img = gf.resize(img, params['target_h'], params['target_w'], interpolation)
        return self._pad_image(img, params, self.pad_val)

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply the transform to a mask."""
        img = gf.resize(img, params['target_h'], params['target_w'], cv2.INTER_NEAREST)
        return self._pad_image(img, params, self.pad_val_mask)

    def apply_to_bbox(self, bbox: list[float], **params) -> list[float]:
        """Apply the transform to a bounding box."""
        h, w = params['target_h'], params['target_w']
        x_min, y_min, x_max, y_max = bbox_utils.denormalize_bbox(bbox, h, w)
        x_min, x_max = x_min + params['w_pad_left'], x_max + params['w_pad_left']
        y_min, y_max = y_min + params['h_pad_top'], y_max + params['h_pad_top']
        return bbox_utils.normalize_bbox((x_min, y_min, x_max, y_max),
                                         h + params['h_pad_top'] + params['h_pad_bottom'],
                                         w + params['w_pad_left'] + params['w_pad_right'])

    def apply_to_keypoint(self, keypoint: tuple[float, float, float, float], **params) -> tuple[float, float, float, float]:
        """Apply the transform to a keypoint."""
        scale_x, scale_y = self.width / params['cols'], self.height / params['rows']
        keypoint = gf.keypoint_scale(keypoint, scale_x, scale_y)
        x, y, angle, scale = keypoint
        return x + params['w_pad_left'], y + params['h_pad_top'], angle, scale

    def _pad_image(self, img: np.ndarray, params: dict[str, Any], pad_val: int) -> np.ndarray:
        """Pad the image with the specified value."""
        return gf.pad_with_params(img=img,
                                  h_pad_top=params['h_pad_top'], h_pad_bottom=params['h_pad_bottom'],
                                  w_pad_left=params['w_pad_left'], w_pad_right=params['w_pad_right'],
                                  border_mode=cv2.BORDER_CONSTANT, value=pad_val)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Get the names of the arguments used in __init__."""
        return ('height', 'width', 'pad_val', 'pad_val_mask')
