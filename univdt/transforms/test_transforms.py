import cv2
import numpy as np
import pytest

from univdt.transforms.resize import Letterbox, RandomResize


@pytest.fixture
def dummy_data():
    # 원본 이미지 (100x200)
    image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    mask = np.ones((100, 200), dtype=np.uint8)

    # bbox: (x_min, y_min, x_max, y_max) normalized
    bboxes = np.array([[0.1, 0.1, 0.4, 0.4]])

    # keypoint: (x, y, z, angle, scale)
    keypoints = np.array([[20, 40, 0, 0, 1.0]])

    return image, mask, bboxes, keypoints


def test_letterbox_image_shape(dummy_data):
    image, _, _, _ = dummy_data
    transform = Letterbox(height=256, width=256)

    result = transform(image=image)
    transformed = result["image"]

    assert transformed.shape[:2] == (256, 256), "Image was not resized to expected shape"


def test_letterbox_mask(dummy_data):
    _, mask, _, _ = dummy_data
    transform = Letterbox(height=256, width=256)

    result = transform(image=mask)
    transformed = result["image"]

    assert transformed.shape[:2] == (256, 256), "Mask was not resized to expected shape"
    assert transformed.dtype == mask.dtype, "Mask dtype should be preserved"


def test_letterbox_bboxes(dummy_data):
    image, _, bboxes, _ = dummy_data
    transform = Letterbox(height=256, width=256)

    result = transform(image=image, bboxes=bboxes, rows=image.shape[0], cols=image.shape[1])
    new_boxes = result["bboxes"]

    assert new_boxes.shape == bboxes.shape, "BBoxes shape should be preserved"
    assert np.all((0.0 <= new_boxes) & (new_boxes <= 1.0)), "BBoxes must remain normalized"


def test_letterbox_keypoints(dummy_data):
    image, _, _, keypoints = dummy_data
    transform = Letterbox(height=256, width=256)

    result = transform(image=image, keypoints=keypoints, rows=image.shape[0], cols=image.shape[1])
    new_kps = result["keypoints"]

    assert new_kps.shape == keypoints.shape, "Keypoints shape should be preserved"
    assert not np.allclose(new_kps[:, :2], keypoints[:, :2]), "Keypoints should have changed location"


@pytest.fixture
def dummy_image():
    return np.ones((120, 240, 3), dtype=np.uint8) * 255


@pytest.fixture
def dummy_mask():
    return np.ones((120, 240), dtype=np.uint8)


def test_random_resize_output_shape(dummy_image):
    transform = RandomResize(height=256, width=256, p=1.0)
    result = transform(image=dummy_image)
    assert result.shape[:2] == (256, 256), "Image shape must match target resize"


def test_random_resize_mask_shape(dummy_mask):
    transform = RandomResize(height=256, width=256, p=1.0)
    result = transform(image=dummy_mask)
    assert result.shape[:2] == (256, 256), "Mask shape must match target resize"


def test_random_resize_preserve_dtype(dummy_image):
    transform = RandomResize(height=256, width=256, p=1.0)
    result = transform(image=dummy_image)
    assert result.dtype == dummy_image.dtype, "Image dtype must be preserved"


def test_random_resize_variety(dummy_image):
    transform = RandomResize(height=256, width=256, interpolations=[cv2.INTER_LINEAR], p=1.0)
    result1 = transform(image=dummy_image)
    result2 = transform(image=dummy_image)
    # 두 번 모두 Resize만 사용했지만 결과가 항상 같을 필요는 없음 (seed 없이 실행하므로)
    assert result1.shape == result2.shape, "All outputs should match target size"


def test_random_resize_include_letterbox(dummy_image):
    # Letterbox만 남겨서 적용 확인
    transform = RandomResize(height=256, width=256, interpolations=[], p=1.0)
    result = transform(image=dummy_image)
    assert result.shape[:2] == (256, 256), "Letterbox should match target output size"
