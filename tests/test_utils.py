from pathlib import Path

import cv2
import numpy as np

from univdt.utils import image
from univdt.utils.logger import Logger

logger = Logger(__name__, 0)

# np.random.seed(0)  # set random seed for reproducibility
TEST_IMAEG = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)


def test_load_image():
    # Create temporary image file
    test_image_path = "./test_image.png"
    cv2.imwrite(test_image_path, TEST_IMAEG)

    # Test loading image with default parameters
    result_image = image.load_image(test_image_path)
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == (100, 100, 3)  # Default channels is 3

    # Test loading image with specified channels (1 channel)
    result_image = image.load_image(test_image_path, out_channels=1)
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == (100, 100, 1)

    # Clean up: Remove temporary image file
    Path(test_image_path).unlink()

    logger.debug("test_load_image passed!")
