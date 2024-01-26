from typing import Any

import albumentations as A
import albumentations.augmentations.geometric.functional as gf
import albumentations.core.bbox_utils as bbox_utils
import cv2
import numpy as np

# TODO: implement this
# class RandomRatio(A.DualTransform):
#     """
#     Apply random ratio to the input. output image size is different from input image size.
#     """
#     def __init__(self)
