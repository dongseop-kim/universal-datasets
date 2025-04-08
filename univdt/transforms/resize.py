import logging
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

# 모듈 수준에서 로거 설정
logger = logging.getLogger(__name__)

DEFAULT_PAD_VAL = 0
DEFAULT_PAD_VAL_MASK = 0


class RandomResize(DualTransform):
    """
    Apply either regular resize or letterbox resize randomly
    Args:
        height (int): target height
        width (int): target width
        interpolations (list[int]): list of interpolation methods to choose from
        letterbox_pad_val (int): padding value for letterbox (image)
        letterbox_pad_val_mask (int): padding value for letterbox (mask)
        p (float): probability of applying the transform
        debug (bool): Enable debug logging
    """

    def __init__(self,
                 height: int = 768,
                 width: int = 768,
                 interpolations: list[int] = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4],
                 letterbox_pad_val: int = 0,
                 letterbox_pad_val_mask: int = 0,
                 p: float = 0.5,
                 debug: bool = False):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.interpolations = interpolations
        self.letterbox_pad_val = letterbox_pad_val
        self.letterbox_pad_val_mask = letterbox_pad_val_mask
        self.debug = debug
        if self.debug:
            logger.debug(f"Initialized RandomResize: height={height}, width={width}, "
                         f"interpolations={interpolations}, letterbox_pad_val={letterbox_pad_val}, "
                         f"letterbox_pad_val_mask={letterbox_pad_val_mask}, p={p}")

        self._resize_transforms1 = [A.Resize(height=height, width=width, interpolation=interp)
                                    for interp in interpolations]
        self._resize_transforms2 = Letterbox(height=height, width=width,
                                             pad_val=letterbox_pad_val,
                                             pad_val_mask=letterbox_pad_val_mask,
                                             debug=debug)

    def __call__(self, force_apply=False, **data):
        if self.debug:
            logger.debug(f"RandomResize.__call__ invoked with force_apply={force_apply}")
            if "image" in data:
                img_shape = data["image"].shape
                logger.debug(f"Input image shape: {img_shape}")

        if not self.should_apply(force_apply=force_apply):
            if self.debug:
                logger.debug("Transform will be skipped (should_apply=False)")
            return data

        use_resize = random.random() < 0.5

        if self.debug:
            transform_type = "resize with interpolation" if use_resize else "letterbox resize"
            logger.debug(f"Selected transform type: {transform_type}")

        if use_resize:
            # Apply resize with interpolation
            chosen_transform = random.choice(self._resize_transforms1)
            if self.debug:
                chosen_interpolation = [i for i, t in enumerate(self._resize_transforms1) if t == chosen_transform][0]
                interp_name = {
                    cv2.INTER_LINEAR: "INTER_LINEAR",
                    cv2.INTER_CUBIC: "INTER_CUBIC",
                    cv2.INTER_LANCZOS4: "INTER_LANCZOS4"
                }.get(self.interpolations[chosen_interpolation], f"Unknown ({self.interpolations[chosen_interpolation]})")
                logger.debug(f"Selected interpolation method: {interp_name}")

            data = chosen_transform(**data)
        else:
            # Apply letterbox resize
            data = self._resize_transforms2(**data)

        if self.debug and "image" in data:
            logger.debug(f"Output image shape: {data['image'].shape}")

        return data

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("height", "width", "interpolations", "letterbox_pad_val", "letterbox_pad_val_mask", "debug")


class Letterbox(DualTransform):
    """
    Apply letterbox resize to inputs
    Args:
        height (int): target height
        width (int): target width
        pad_val (int): padding value for image
        pad_val_mask (int): padding value for mask
        border_mode (int): border mode for padding
        p (float): probability of applying the transform
        debug (bool): Enable debug logging
    """

    def __init__(self,
                 height: int,
                 width: int,
                 pad_val: int = 0,
                 pad_val_mask: int = 0,
                 border_mode: int = cv2.BORDER_CONSTANT,
                 p: float = 1.0,
                 debug: bool = False) -> None:
        super().__init__(p)
        self.height = height
        self.width = width
        self.pad_val = pad_val
        self.pad_val_mask = pad_val_mask
        self.border_mode = border_mode
        self.debug = debug

        # Configure logger if debug is enabled
        if self.debug:
            logger = logging.getLogger("Letterbox")
            logger.setLevel(logging.DEBUG)

            # Create handler if not already exists
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            # Get border mode name for better logging
            border_mode_name = {cv2.BORDER_CONSTANT: "BORDER_CONSTANT",
                                cv2.BORDER_REPLICATE: "BORDER_REPLICATE",
                                cv2.BORDER_REFLECT: "BORDER_REFLECT",
                                cv2.BORDER_WRAP: "BORDER_WRAP",
                                cv2.BORDER_REFLECT_101: "BORDER_REFLECT_101"
                                }.get(border_mode, f"Unknown ({border_mode})")

            logger.debug(f"Initialized Letterbox: height={height}, width={width}, "
                         f"pad_val={pad_val}, pad_val_mask={pad_val_mask}, "
                         f"border_mode={border_mode_name}, p={p}")

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('height', 'width', 'pad_val', 'pad_val_mask', 'border_mode', 'debug')

    def get_params(self) -> dict[str, Any]:
        return {}

    def get_params_dependent_on_data(self, params: dict[str, Any],
                                     data: dict[str, Any]) -> dict[str, Any]:
        image: np.ndarray = data["image"]
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

        if self.debug:
            logger.debug(f"Original dimensions: {h}x{w}")
            logger.debug(f"Scale factor: {scale:.4f}")
            logger.debug(f"New dimensions after scaling: {new_h}x{new_w}")
            logger.debug(f"Padding - left: {pad_left}, right: {pad_right}, top: {pad_top}, bottom: {pad_bottom}")

        return {'new_h': new_h, 'new_w': new_w,
                'pad_top': pad_top, 'pad_bottom': pad_bottom,
                'pad_left': pad_left, 'pad_right': pad_right,
                'scale': scale,
                'rows': h, 'cols': w,
                'shape': image.shape}

    def apply(self, img: np.ndarray,
              new_h: int, new_w: int,
              pad_top: int, pad_bottom: int,
              pad_left: int, pad_right: int,
              interpolation: int = cv2.INTER_LINEAR,
              **params: Any) -> np.ndarray:
        if self.debug:
            logger.debug(f"Applying to image with shape {img.shape}")
            logger.debug(f"Using interpolation: {interpolation}")
            logger.debug(f"Image stats before transform - min: {img.min()}, max: {img.max()}, "
                         f"mean: {img.mean():.2f}, std: {img.std():.2f}")

        resized = resize(img, (new_h, new_w), interpolation=interpolation)

        if self.debug:
            logger.debug(f"After resize, before padding: shape={resized.shape}")

        result = pad_with_params(resized, pad_top, pad_bottom, pad_left, pad_right,
                                 border_mode=self.border_mode, value=self.pad_val)

        if self.debug:
            logger.debug(f"Final image shape: {result.shape}")
            logger.debug(f"Image stats after transform - min: {result.min()}, max: {result.max()}, "
                         f"mean: {result.mean():.2f}, std: {result.std():.2f}")
            padded_pixels = pad_top * result.shape[1] + pad_bottom * result.shape[1] + \
                pad_left * new_h + pad_right * new_h
            total_pixels = result.size
            pad_percentage = (padded_pixels / total_pixels) * 100
            logger.debug(f"Added {padded_pixels} padding pixels ({pad_percentage:.2f}% of total)")

        return result

    def apply_to_mask(self,
                      img: np.ndarray,
                      new_h: int, new_w: int,
                      pad_top: int, pad_bottom: int,
                      pad_left: int, pad_right: int,
                      **params: Any) -> np.ndarray:
        if self.debug:
            logger.debug(f"Applying to mask with shape {img.shape}")
            logger.debug(f"Mask stats before transform - min: {img.min()}, max: {img.max()}, "
                         f"unique values: {np.unique(img)}")

        resized = resize(img, (new_h, new_w), interpolation=cv2.INTER_NEAREST)

        if self.debug:
            logger.debug(f"After resize, before padding: shape={resized.shape}")

        result = pad_with_params(resized, pad_top, pad_bottom, pad_left, pad_right,
                                 border_mode=self.border_mode, value=self.pad_val_mask)

        if self.debug:
            logger.debug(f"Final mask shape: {result.shape}")
            logger.debug(f"Mask stats after transform - min: {result.min()}, max: {result.max()}, "
                         f"unique values: {np.unique(result)}")

        return result

    def apply_to_bboxes(self, bboxes: np.ndarray, new_h: int, new_w: int,
                        pad_top: int, pad_bottom: int,
                        pad_left: int, pad_right: int,
                        rows: int, cols: int, **params) -> np.ndarray:
        if self.debug and len(bboxes) > 0:
            logger.debug(f"Applying to {len(bboxes)} bboxes")
            logger.debug(f"Original bboxes (normalized): {bboxes}")

        denorm = denormalize_bboxes(bboxes, (rows, cols))

        if self.debug and len(bboxes) > 0:
            logger.debug(f"Denormalized bboxes: {denorm}")

        denorm[:, [0, 2]] *= new_w / cols
        denorm[:, [1, 3]] *= new_h / rows

        if self.debug and len(bboxes) > 0:
            logger.debug(f"Scaled bboxes: {denorm}")

        denorm[:, [0, 2]] += pad_left
        denorm[:, [1, 3]] += pad_top

        if self.debug and len(bboxes) > 0:
            logger.debug(f"After padding offset: {denorm}")

        result = normalize_bboxes(denorm, (self.height, self.width))

        if self.debug and len(bboxes) > 0:
            logger.debug(f"Final normalized bboxes: {result}")

        return result

    def apply_to_keypoints(self,
                           keypoints: np.ndarray,
                           new_h: int, new_w: int,
                           pad_top: int, pad_bottom: int,
                           pad_left: int, pad_right: int,
                           **params: Any) -> np.ndarray:
        if self.debug and len(keypoints) > 0:
            logger.debug(f"Applying to {len(keypoints)} keypoints")
            logger.debug(f"Original keypoints: {keypoints}")

        keypoints = np.asarray(keypoints)
        rows, cols = params["shape"][:2]
        scaled = keypoints_scale(keypoints, new_w / cols, new_h / rows)

        if self.debug and len(keypoints) > 0:
            logger.debug(f"Scaled keypoints: {scaled}")

        scaled[:, 0] += pad_left
        scaled[:, 1] += pad_top

        if self.debug and len(keypoints) > 0:
            logger.debug(f"Final keypoints after padding offset: {scaled}")

        return scaled

    def __call__(self, force_apply=False, **data):
        if self.debug:
            logger.debug(f"Letterbox.__call__ invoked with force_apply={force_apply}")
            if "image" in data:
                img_shape = data["image"].shape
                logger.debug(f"Input image shape: {img_shape}")

        result = super().__call__(force_apply=force_apply, **data)

        if self.debug and "image" in result:
            logger.debug(f"Output image shape: {result['image'].shape}")

        return result
