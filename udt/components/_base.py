from torch.utils.data import Dataset
import numpy as np
import cv2

def _check_image(image:np.ndarray):
    # check 3 channels
    if len(image.shape) != 3 and image.shape[2] != 3:
        raise ValueError("image must be 3 channels")
    # check dtype
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")


class BaseComponent(Dataset):
    def __init__(self):
        pass

    def _load_image(self, path_image:str):
        image = cv2.imread(path_image, cv2.IMREAD_UNCHANGED)
        try:
            _check_image(image)
        except ValueError as e:
            raise ValueError("image must be 3 channels and uint8") from e
