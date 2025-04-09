import logging
import random
from typing import Any

import albumentations as A
import cv2
from albumentations.core.transforms_interface import DualTransform

logger = logging.getLogger(__name__)


def clamp_bbox_xyxy(bbox: tuple[float, float, float, float],
                    image_shape: tuple[int, int]) -> list[float]:
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]
    return [max(0, min(x1, w)),
            max(0, min(y1, h)),
            max(0, min(x2, w)),
            max(0, min(y2, h)),]


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
            image_shape = result['image'].shape
            result['bboxes'] = [clamp_bbox_xyxy(bbox, image_shape) for bbox in result['bboxes']]
            if self.debug:
                logger.debug(f" - Clamped {len(result['bboxes'])} bboxes to image bounds.")

        return result
