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


def get_masks(paths_mask: List[str], target_shape: List[int], labels: List[int]) -> np.ndarray:
    '''
    labels : list of int (우리가 쓰는 class id. e.g. 1: Nodule)
    '''
    # TODO: Update 필요
    h, w = target_shape  # H W
    mask = np.zeros((20, *target_shape), dtype=np.uint8)  # 255 x H x W
    for path_mask, trainid in zip(paths_mask, labels):
        mask_tmp = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        mask_tmp = cv2.resize(mask_tmp, (w, h), interpolation=cv2.INTER_NEAREST)
        mask[trainid] += mask_tmp
    mask: np.ndarray = np.where(mask > 0, 1, 0)
    return mask  # 255 x H x W
