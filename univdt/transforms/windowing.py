import numpy as np
import logging

from albumentations.core.transforms_interface import ImageOnlyTransform

# 모듈 수준에서 로거 설정
logger = logging.getLogger(__name__)


def configure_logging(debug=False):
    """전역 로깅 설정을 구성합니다."""
    level = logging.DEBUG if debug else logging.INFO

    # 루트 로거 설정
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def windowing(image: np.ndarray, use_median: bool = False, width_param: float = 4.0) -> np.ndarray:
    """
    Windowing function that clips the values based on the given params.
    Args:
        image (str): the image to do the windowing
        use_median (bool): use median as center if True, mean otherwise
        width_param (float): the width of the value range for windowing.
        brightness (float) : brightness_rate. a value between 0 and 1 and indicates the amount to subtract.

    Returns:
        image that was windowed.
    """
    center = np.median(image) if use_median else image.mean()
    range_width_half = (image.std() * width_param) / 2.0
    low, high = center - range_width_half, center + range_width_half
    return np.clip(image, low, high)


class RandomWindowing(ImageOnlyTransform):
    """
    Apply random windowing
    Args:
        width_param (float): width parameter
        width_range (float): width range. width_param - width_range/2 ~ width_param + width_range/2
                             if width_param = 4.0, width_range = 1.0, then width_param = 3.5 ~ 4.5
        use_median (bool): use median or not
        p (float): probability
        debug (bool): Enable debug logging
    """

    def __init__(self,
                 width_param: float = 4.0,
                 width_range: float = 1.0,
                 use_median: bool = True,
                 p: float = 0.5,
                 debug: bool = False):
        super().__init__(p)
        self.use_median = use_median
        self.width_param = width_param
        self.width_range = width_range
        self.debug = debug

        # 모듈 로거의 레벨만 조정 (필요한 경우)
        if self.debug and logger.level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)

        if self.debug:
            logger.debug(
                f"Initialized RandomWindowing: width_param={width_param}, width_range={width_range}, use_median={use_median}, p={p}")

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if self.debug:
            logger.debug(f"Processing image with shape {img.shape}, dtype {img.dtype}")
            logger.debug(
                f"Image stats before transform - min: {img.min()}, max: {img.max()}, mean: {img.mean():.2f}, std: {img.std():.2f}")

        width_param = self._get_random_width_param()

        if self.debug:
            logger.debug(f"Using width_param: {width_param:.4f}")
            logger.debug(f"Using median as center: {self.use_median}")

        # Calculate statistical values for logging if debug is enabled
        if self.debug:
            center = np.median(img) if self.use_median else img.mean()
            std = img.std()
            range_width_half = (std * width_param) / 2.0
            low, high = center - range_width_half, center + range_width_half
            logger.debug(f"Windowing parameters - center: {center:.2f}, std: {std:.2f}")
            logger.debug(f"Clipping range - low: {low:.2f}, high: {high:.2f}")

        result = windowing(img, use_median=self.use_median, width_param=width_param)

        if self.debug:
            logger.debug(
                f"Image stats after transform - min: {result.min()}, max: {result.max()}, mean: {result.mean():.2f}, std: {result.std():.2f}")
            clipped_pixels = np.sum((img < low) | (img > high))
            total_pixels = img.size
            clip_percentage = (clipped_pixels / total_pixels) * 100
            logger.debug(f"Clipped {clipped_pixels} pixels ({clip_percentage:.2f}% of total)")

        return result.astype(np.uint8)

    def _get_random_width_param(self) -> float:
        # Generate a random width parameter within the specified range.
        random_width = self.width_param + np.random.uniform(-self.width_range/2, self.width_range/2)

        if self.debug:
            logger.debug(
                f"Generated random width_param: {random_width:.4f} (base: {self.width_param}, range: ±{self.width_range/2:.2f})")

        return random_width

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('width_param', 'width_range', 'use_median', 'debug')

    def __call__(self, *args, **kwargs):
        if self.debug:
            logger.debug(f"RandomWindowing.__call__ invoked with p={self.p}")
            if np.random.random() < self.p:
                logger.debug(f"Transform will be applied (random < p)")
            else:
                logger.debug(f"Transform will be skipped (random >= p)")

        return super().__call__(*args, **kwargs)


# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    configure_logging(debug=True)

    # 변환 생성
    transform = RandomWindowing(width_param=4.0, width_range=2.0, use_median=True, p=1.0, debug=True)

    # 테스트 이미지로 변환 적용
    sample_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    result = transform(image=sample_img)
