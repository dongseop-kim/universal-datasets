import logging
import random
from typing import Any, Literal, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform

logger = logging.getLogger(__name__)
eps = 1e-6


def clamp_bbox_xyxy(bbox: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [max(0, min(x1, 1.0-eps)),
            max(0, min(y1, 1.0-eps)),
            max(0, min(x2, 1.0-eps)),
            max(0, min(y2, 1.0-eps))]


class RandomTranslation(DualTransform):
    """
    Apply random translation (x and/or y) to the image using Albumentations.Affine.

    Args:
        max_dx (float): Max horizontal shift as fraction of image width (±%)
        max_dy (float): Max vertical shift as fraction of image height (±%)
        fill (int): Fill value for empty image pixels
        fill_mask (int): Fill value for empty mask pixels
        interpolation (int): OpenCV interpolation method for image
        debug (bool): If True, prints debug info
        keep_box (bool): If True, fit_output=True to preserve full bbox region
        p (float): Probability of applying the transform
    """

    def __init__(self,
                 max_dx: float = 0.1, max_dy: float = 0.1,
                 fill: int = 0, fill_mask: int = 0,
                 interpolation: int = cv2.INTER_LINEAR,
                 keep_box: bool = False,
                 p: float = 0.5,
                 debug: bool = False):
        super().__init__(p)
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.fill = fill
        self.fill_mask = fill_mask
        self.interpolation = interpolation
        self.keep_box = keep_box
        self.debug = debug

        if self.debug:
            logger.debug(f"Initialized RandomTranslation: "
                         f"max_dx={max_dx}, max_dy={max_dy}, "
                         f"fill={fill}, fill_mask={fill_mask}, "
                         f"interpolation={interpolation}, keep_box={keep_box}, p={p}")

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('max_dx', 'max_dy', 'fill', 'fill_mask', 'interpolation', 'debug', 'keep_box')

    def __call__(self, force_apply=False, **data: Any) -> dict[str, Any]:
        if not self.should_apply(force_apply):
            if self.debug:
                logger.debug("RandomTranslation skipped (should_apply=False)")
            return data

        height, width = data['image'].shape[:2]

        # 선택적 평행이동 방향
        mode = random.choice(['x', 'y', 'xy'])
        tx = random.uniform(-self.max_dx, self.max_dx) if mode in ('x', 'xy') else 0.0
        ty = random.uniform(-self.max_dy, self.max_dy) if mode in ('y', 'xy') else 0.0

        if self.debug:
            logger.debug(f"Applying RandomTranslation:")
            logger.debug(f" - Image shape: (H={height}, W={width})")
            logger.debug(f" - Selected mode: {mode}")
            logger.debug(f" - Sampled translate_percent: dx={tx:.4f}, dy={ty:.4f}")
            logger.debug(f" - Translated pixels: dx={int(tx * width)}, dy={int(ty * height)}")
            logger.debug(f" - fit_output (keep_box): {self.keep_box}")

        # Affine transform (translation only)
        affine = A.Affine(scale=1.0,
                          rotate=0.0,
                          shear=0.0,
                          translate_percent={'x': tx, 'y': ty},
                          interpolation=self.interpolation,
                          mask_interpolation=cv2.INTER_NEAREST,
                          fill=self.fill,
                          fill_mask=self.fill_mask,
                          border_mode=cv2.BORDER_CONSTANT,
                          fit_output=self.keep_box,
                          p=1.0  # 이미 확률 체크 후 호출되므로 무조건 적용
                          )
        result = affine(**data)

        # ⚠️ bbox 클램핑 추가
        if 'bboxes' in result:
            result['bboxes'] = np.array([clamp_bbox_xyxy(bbox) for bbox in result['bboxes']])
            if self.debug:
                logger.debug(f" - Clamped {len(result['bboxes'])} bboxes to image bounds.")

        return result


class RandomZoom(DualTransform):
    """
    Random zoom-in/out using Albumentations.Affine under the hood.
    Keeps image size the same by applying either padding or cropping.

    Args:
        scale (float | tuple[float, float]): Zoom range. If float s, range is (1-s, 1+s).
            - Values < 1.0 → Zoom-out: image is downscaled and padded to original size.
            - Values > 1.0 → Zoom-in: image is upscaled and randomly cropped to original size.
            - If a single float `s` is provided, the scale range is interpreted as (1 - s, 1 + s).
            - If a tuple (min, max) is provided, a value is sampled uniformly from that range.

            Example:
                scale=0.1 → random scale between 0.9 and 1.1
                scale=(0.8, 1.2) → random scale between 0.8 and 1.2

            NOTE:
                - This transform preserves the output size.
                - Padding uses `fill`, cropping is random (top-left offset).

        keep_ratio (bool): Whether to preserve aspect ratio during zoom.
        keep_bbox (bool): Whether to fit output to keep bbox area after zooming out.
        fill (int): Padding value for image.
        fill_mask (int): Padding value for mask.
        interpolation (int): Interpolation for image.
        debug (bool): Enable debug logs.
        p (float): Probability of applying transform.
    """

    def __init__(self,
                 scale: float | Tuple[float, float] = 0.1,
                 keep_ratio: bool = True,
                 keep_bbox: bool = False,
                 fill: int = 0,
                 fill_mask: int = 0,
                 interpolation: int = cv2.INTER_LINEAR,
                 debug: bool = False,
                 p: float = 1.0):
        super().__init__(p=p)

        # scale normalization
        if isinstance(scale, float):
            scale = (1 - scale, 1 + scale)
        scale_dict = {"x": scale, "y": scale}

        self.affine = A.Affine(scale=scale_dict,
                               translate_percent=None,
                               rotate=0.0,
                               shear={"x": (0.0, 0.0), "y": (0.0, 0.0)},
                               interpolation=interpolation,
                               mask_interpolation=cv2.INTER_NEAREST,
                               fit_output=keep_bbox,
                               keep_ratio=keep_ratio,
                               rotate_method="largest_box",
                               balanced_scale=False,
                               fill=fill,
                               fill_mask=fill_mask,
                               border_mode=cv2.BORDER_CONSTANT,
                               p=1.0  # apply manually
                               )

        self.debug = debug

        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Initialized SafeRandomZoom with parameters:")
            logger.debug(f" - scale: {scale_dict}")
            logger.debug(f" - keep_ratio: {keep_ratio}, keep_bbox: {keep_bbox}")
            logger.debug(f" - fill: {fill}, fill_mask: {fill_mask}, interpolation: {interpolation}")

    def __call__(self, force_apply=False, **data: Any) -> dict[str, Any]:
        if not self.should_apply(force_apply):
            if self.debug:
                logger.debug("SafeRandomZoom skipped (should_apply=False)")
            return data

        if self.debug:
            shape = data["image"].shape
            logger.debug(f"Applying SafeRandomZoom to image of shape: {shape}")

        result = self.affine(**data)

        # ✅ bbox 클램핑
        if 'bboxes' in result:
            result['bboxes'] = np.array([clamp_bbox_xyxy(bbox) for bbox in result['bboxes']])
            if self.debug:
                logger.debug(f" - Clamped {len(result['bboxes'])} bboxes to image bounds.")

        return result

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("scale", "keep_ratio", "fill", "fill_mask", "interpolation")
