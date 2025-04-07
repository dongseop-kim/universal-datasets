import random
from typing import Any

import albumentations as A
import cv2
import numpy as np
from albumentations.augmentations.geometric.functional import (keypoints_scale,
                                                               pad_with_params,
                                                               resize)
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from albumentations.core.transforms_interface import DualTransform

DEFAULT_PAD_VAL = 0
DEFAULT_PAD_VAL_MASK = 0


class RandomResize(DualTransform):
    """
    Randomly apply one of several resize methods (including Letterbox) to an image.

    Args:
        height (int): Target height.
        width (int): Target width.
        interpolations (list[int]): List of OpenCV interpolation methods (e.g., [cv2.INTER_LINEAR, ...]).
        letterbox_pad_val (int): Padding value for image in Letterbox.
        letterbox_pad_val_mask (int): Padding value for mask in Letterbox.
        p (float): Probability of applying transform.
    """

    def __init__(self,
                 height: int = 768,
                 width: int = 768,
                 interpolations: list[int] = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4],
                 letterbox_pad_val: int = 0,
                 letterbox_pad_val_mask: int = 0,
                 p: float = 0.5):
        super().__init__(p)
        self.height = height
        self.width = width
        self.interpolations = interpolations
        self.letterbox_pad_val = letterbox_pad_val
        self.letterbox_pad_val_mask = letterbox_pad_val_mask
        self._create_transforms()

    def _create_transforms(self):
        """Create candidate resize transforms to choose from at call time."""
        self.resize_transforms = [A.Resize(height=self.height,
                                           width=self.width,
                                           interpolation=interp) for interp in self.interpolations]

        # Include Letterbox as an alternative resize method
        self.resize_transforms.append(Letterbox(height=self.height,
                                                width=self.width,
                                                pad_val=self.letterbox_pad_val,
                                                pad_val_mask=self.letterbox_pad_val_mask))
        # Wrap all resize options under OneOf
        self.transform = A.OneOf(self.resize_transforms, p=1.0)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return self.transform(image=img, **params)["image"]

    def apply_to_bboxes(self, bboxes: list[list[float]], **params: Any) -> list[list[float]]:
        """Apply resize or letterbox transform to bounding boxes."""
        transformed = self.transform(bboxes=bboxes, **params)
        return transformed['bboxes']

    def apply_to_masks(self, masks, *args, **params):
        """Apply resize or letterbox transform to masks."""
        transformed = self.transform(masks=masks, **params)
        return transformed['masks']

    def apply_to_keypoints(self, keypoints: list[list[float]], **params: Any) -> list[list[float]]:
        """Apply resize or letterbox transform to keypoints."""
        transformed = self.transform(keypoints=keypoints, **params)
        return transformed['keypoints']

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('height', 'width', 'interpolations', 'letterbox_pad_val', 'letterbox_pad_val_mask')


class Letterbox(DualTransform):
    """
    Resize the input to fit within the target (height, width) while preserving aspect ratio.
    Adds padding to maintain target size. Supports image, mask, bboxes, keypoints.

    Args:
        height (int): Target output height.
        width (int): Target output width.
        pad_val (int): Padding value for image.
        pad_val_mask (int): Padding value for mask.
        border_mode (int): Border mode for padding (OpenCV constant).
        always_apply (bool): If True, always apply the transform.
        p (float): Probability of applying the transform.
    """

    def __init__(self,
                 height: int, width: int,
                 pad_val: int = 0, pad_val_mask: int = 0,
                 border_mode: int = cv2.BORDER_CONSTANT,
                 p: float = 1.0):
        super().__init__(p)
        self.height = height
        self.width = width
        self.pad_val = pad_val
        self.pad_val_mask = pad_val_mask
        self.border_mode = border_mode

    def get_transform_init_args_names(self):
        return ('height', 'width', 'pad_val', 'pad_val_mask', 'border_mode',)

    def get_params(self) -> dict[str, Any]:
        """ No random params needed, so return empty dict. """
        return {}

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """
        Compute resize scale and padding required to fit original image into target size.
        """
        h, w = data['image'].shape[:2]
        scale = min(self.width / w, self.height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        pad_w = self.width - new_w
        pad_h = self.height - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        return {'new_h': new_h, 'new_w': new_w,
                'pad_top': pad_top, 'pad_bottom': pad_bottom,
                'pad_left': pad_left, 'pad_right': pad_right,
                'scale': scale}

    def apply(self, img: np.ndarray, new_h: int, new_w: int,
              pad_top: int, pad_bottom: int,
              pad_left: int, pad_right: int,
              interpolation: int = cv2.INTER_LINEAR, **params) -> np.ndarray:
        """
        Apply resize and padding to image.
        """
        resized = resize(img, new_h, new_w, interpolation=interpolation)
        return pad_with_params(resized, pad_top, pad_bottom, pad_left, pad_right,
                               border_mode=self.border_mode, value=self.pad_val)

    def apply_to_mask(self, img: np.ndarray, new_h: int, new_w: int,
                      pad_top: int, pad_bottom: int,
                      pad_left: int, pad_right: int,
                      **params) -> np.ndarray:
        """
        Apply resize and padding to mask.
        Use nearest interpolation for discrete label preservation.
        """
        resized = resize(img, new_h, new_w, interpolation=cv2.INTER_NEAREST)
        return pad_with_params(resized, pad_top, pad_bottom, pad_left, pad_right,
                               border_mode=self.border_mode, value=self.pad_val_mask)

    def apply_to_bboxes(self, bboxes: np.ndarray, new_h: int, new_w: int,
                        pad_top: int, pad_bottom: int,
                        pad_left: int, pad_right: int,
                        rows: int, cols: int, **params) -> np.ndarray:
        """
        Adjust bounding box coordinates after resize and padding.
        Input bboxes are normalized in [0, 1] (x_min, y_min, x_max, y_max, ...)
        """
        denorm = denormalize_bboxes(bboxes, (rows, cols))
        denorm[:, [0, 2]] *= new_w / cols
        denorm[:, [1, 3]] *= new_h / rows
        denorm[:, [0, 2]] += pad_left
        denorm[:, [1, 3]] += pad_top
        return normalize_bboxes(denorm, (self.height, self.width))

    def apply_to_keypoints(self, keypoints: np.ndarray, new_h: int, new_w: int,
                           pad_top: int, pad_bottom: int,
                           pad_left: int, pad_right: int,
                           cols: int, rows: int, **params) -> np.ndarray:
        """
        Adjust keypoints after resize and padding.
        Keypoints are of shape (N, 5+) format: (x, y, z, angle, scale, ...)
        """
        scaled = keypoints_scale(keypoints, new_w / cols, new_h / rows)
        scaled[:, 0] += pad_left  # x
        scaled[:, 1] += pad_top   # y
        return scaled
