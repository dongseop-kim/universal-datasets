from typing import List

import cv2
import numpy as np
from utils.misc import load_cxr_image


def to_8bit(image: np.ndarray) -> np.ndarray:
    image = image*255.0
    image = image.astype(np.uint8)
    return image


def get_image(path_image: str, train: bool = False) -> np.ndarray:
    width_param = 3.5 + np.random.rand(1) if train else 4.0
    image: np.ndarray = load_cxr_image(filename=path_image, do_windowing=True,
                                       width_param=width_param)  # H W 1 or HW
    image: np.ndarray = to_8bit(image)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    return image  # H x W x C


def get_bboxes(bboxes: List[float]) -> np.ndarray:
    bboxes: np.ndarray = np.array(bboxes).reshape(-1, 4)  # N x 4
    return bboxes


def get_mask(path_mask: str, target_shape: List[int]) -> np.ndarray:
    h, w = target_shape  # H W
    mask: np.ndarray = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    mask: np.ndarray = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = np.squeeze(mask)
    return mask  # H x W


def get_masks(path_masks: List[str], target_shape: List[int]) -> np.ndarray:
    masks: List[np.ndarray] = [get_mask(path_mask, target_shape) for path_mask in path_masks]
    masks: np.ndarray = np.stack(masks, axis=0)  # N x H x W
    return masks  # N x H x W
