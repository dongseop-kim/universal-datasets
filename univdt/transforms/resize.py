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
    def __init__(self,
                 height: int = 768,
                 width: int = 768,
                 interpolations: list[int] = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4],
                 letterbox_pad_val: int = 0,
                 letterbox_pad_val_mask: int = 0,
                 p: float = 0.5):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.interpolations = interpolations
        self.letterbox_pad_val = letterbox_pad_val
        self.letterbox_pad_val_mask = letterbox_pad_val_mask

        self._resize_transforms1 = [A.Resize(height=height, width=width, interpolation=interp)
                                    for interp in interpolations]
        self._resize_transforms2 = Letterbox(height=height, width=width, pad_val=letterbox_pad_val,
                                             pad_val_mask=letterbox_pad_val_mask)

    def __call__(self, force_apply=False, **data):
        if not self.should_apply(force_apply=force_apply):
            return data

        if random.random() < 0.5:
            # Apply resize with interpolation
            transform = random.choice(self._resize_transforms1)
            data = transform(**data)
        else:
            # Apply letterbox resize
            print("Applying letterbox resize")
            data = self._resize_transforms2(**data)
        return data

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("height", "width", "interpolations", "letterbox_pad_val", "letterbox_pad_val_mask")


class Letterbox(DualTransform):
    def __init__(self,
                 height: int,
                 width: int,
                 pad_val: int = 0,
                 pad_val_mask: int = 0,
                 border_mode: int = cv2.BORDER_CONSTANT,
                 p: float = 1.0) -> None:
        super().__init__(p)
        self.height = height
        self.width = width
        self.pad_val = pad_val
        self.pad_val_mask = pad_val_mask
        self.border_mode = border_mode

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('height', 'width', 'pad_val', 'pad_val_mask', 'border_mode')

    def get_params(self) -> dict[str, Any]:
        return {}

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        image = data["image"]
        h, w = image.shape[:2]
        scale = min(self.width / w, self.height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        pad_w = self.width - new_w
        pad_h = self.height - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        return {
            'new_h': new_h, 'new_w': new_w,
            'pad_top': pad_top, 'pad_bottom': pad_bottom,
            'pad_left': pad_left, 'pad_right': pad_right,
            'scale': scale,
            'rows': h,  # original shape 전달
            'cols': w,
            'shape': image.shape,
        }

    def apply(self,
              img: np.ndarray,
              new_h: int,
              new_w: int,
              pad_top: int,
              pad_bottom: int,
              pad_left: int,
              pad_right: int,
              interpolation: int = cv2.INTER_LINEAR,
              **params: Any) -> np.ndarray:
        resized = resize(img, (new_h, new_w), interpolation=interpolation)
        return pad_with_params(resized, pad_top, pad_bottom, pad_left, pad_right,
                               border_mode=self.border_mode, value=self.pad_val)

    def apply_to_mask(self,
                      img: np.ndarray,
                      new_h: int,
                      new_w: int,
                      pad_top: int,
                      pad_bottom: int,
                      pad_left: int,
                      pad_right: int,
                      **params: Any) -> np.ndarray:
        resized = resize(img, (new_h, new_w), interpolation=cv2.INTER_NEAREST)
        return pad_with_params(resized, pad_top, pad_bottom, pad_left, pad_right,
                               border_mode=self.border_mode, value=self.pad_val_mask)

    def apply_to_bboxes(self, bboxes: np.ndarray, new_h: int, new_w: int,
                        pad_top: int, pad_bottom: int,
                        pad_left: int, pad_right: int,
                        rows: int, cols: int, **params) -> np.ndarray:
        denorm = denormalize_bboxes(bboxes, (rows, cols))
        denorm[:, [0, 2]] *= new_w / cols
        denorm[:, [1, 3]] *= new_h / rows
        denorm[:, [0, 2]] += pad_left
        denorm[:, [1, 3]] += pad_top
        return normalize_bboxes(denorm, (self.height, self.width))

    def apply_to_keypoints(self,
                           keypoints: np.ndarray,
                           new_h: int,
                           new_w: int,
                           pad_top: int,
                           pad_bottom: int,
                           pad_left: int,
                           pad_right: int,
                           **params: Any) -> np.ndarray:
        keypoints = np.asarray(keypoints)
        rows, cols = params["shape"][:2]
        scaled = keypoints_scale(keypoints, new_w / cols, new_h / rows)
        scaled[:, 0] += pad_left
        scaled[:, 1] += pad_top
        return scaled
