import random
from typing import Any

import albumentations as A
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


def random_gamma(magnitude: float = 0.2, p=1.0):
    """
    Randomly change the gamma of the input image.

    Args:
        magnitude (float): Maximum gamma variation. The range becomes (100 - m, 100 + m).
                           Ex: magnitude=0.2 → gamma_limit=(80, 120)
        p (float): Probability of applying the transform.
    """
    assert 0.0 < magnitude <= 1.0
    magnitude = min(magnitude, 1.0) * 100
    gamma_limit = (max(1, 100 - magnitude), 100 + magnitude)  # e.g., (80, 120) for magnitude=0.2
    return A.RandomGamma(gamma_limit=gamma_limit, p=p)


class Invert(ImageOnlyTransform):
    """Invert pixel values in the image (255 - pixel value)."""

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Args:
            img: Image to invert.

        Returns:
            Inverted image.
        """
        return 255 - img

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()


def random_inverse(magnitude: float = 0.2, p: float = 1.0) -> Invert:
    """
    Invert pixel values (255 - x) to simulate photographic negative.

    Args:
        magnitude: Not used, kept for API consistency.
        p: Probability of applying the transform.

    Returns:
        Invert transform.
    """
    # magnitude 파라미터는 사용하지 않지만 API 일관성을 위해 유지
    return Invert(p=p)


def random_blur(magnitude: float = 0.2, p: float = 1.0):
    """
    Apply a randomly selected blur transform (box, gaussian, motion, median, glass, advanced),
    with parameters derived from `magnitude`.

    Args:
        magnitude (float): Controls overall blur strength. Value should be in (0.0, 1.0].
                           - Higher magnitude → larger kernel size, stronger blur.
        p (float): Probability of applying the transform.

    Returns:
        Albumentations transform applying one of multiple blur types.
    """
    assert 0.0 < magnitude <= 1.0

    # --- Common blur kernel size: range from 3 to 3 + 18*m (odd only) ---
    max_kernel = int(3 + round(magnitude * 18))
    max_kernel += 1 if max_kernel % 2 == 0 else 0  # must be odd
    blur_limit = (3, max_kernel)

    # --- Gaussian sigma range: 0.1 to 2.0 based on magnitude ---
    sigma_limit = (0.05, 0.1 + 1.9 * magnitude)

    # --- Motion blur angle: full range, direction: balanced (-0.5~0.5), magnitude-scaled bias ---
    angle_range = (0, 360)
    direction_range = (-1.0 * magnitude, 1.0 * magnitude)

    # --- Glass blur ---
    glass_sigma = 0.1 + 1.5 * magnitude
    glass_delta = max(1, int(1 + magnitude * 5))
    glass_iter = max(1, int(1 + magnitude * 4))

    # --- Advanced blur ---
    beta_limit = (0.5, 1.0 + magnitude * 7.0)          # 1=gaussian, <1=box-like, >1=peaked
    noise_limit = (1.0 - 0.25 * magnitude, 1.0 + 0.25 * magnitude)
    sigma_xy = (0.1, 0.2 + 0.8 * magnitude)
    rotate_limit = (-45, 45) if magnitude < 0.5 else (-90, 90)

    return A.OneOf([A.Blur(blur_limit=blur_limit, p=1.0),
                    A.GaussianBlur(blur_limit=blur_limit,
                                   sigma_limit=sigma_limit, p=1.0),
                    A.MotionBlur(blur_limit=blur_limit,
                                 angle_range=angle_range,
                                 direction_range=direction_range,
                                 allow_shifted=True, p=1.0),
                    A.MedianBlur(blur_limit=blur_limit, p=1.0),
                    A.GlassBlur(sigma=glass_sigma,
                                max_delta=glass_delta,
                                iterations=glass_iter,
                                mode="fast", p=1.0),
                    A.AdvancedBlur(blur_limit=blur_limit,
                                   sigma_x_limit=sigma_xy,
                                   sigma_y_limit=sigma_xy,
                                   beta_limit=beta_limit,
                                   noise_limit=noise_limit,
                                   rotate_limit=rotate_limit,
                                   p=1.0)],
                   p=p)


def random_brightness(magnitude: float = 0.2, p: float = 1.0):
    """
    Randomly adjust brightness of the image without changing contrast.

    Args:
        magnitude (float): Controls the brightness shift factor range.
            The range becomes (-m, m), which maps to beta ∈ (-m * max_value, m * max_value)
            For uint8 images (max=255), magnitude=0.2 → brightness shift ∈ [-51, +51]
        p (float): Probability of applying the transform.

    Returns:
        Albumentations transform that applies random brightness adjustment only.
    """
    assert 0.0 < magnitude <= 1.0
    brightness_limit = (-magnitude, magnitude)
    return A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=0,
                                      brightness_by_max=True, ensure_safe_range=True, p=p)


def random_contrast(magnitude: float = 0.2, p: float = 1.0):
    """
    Randomly adjust contrast of the image without changing brightness.

    Args:
        magnitude (float): Controls the contrast scaling factor range.
            The range becomes (-m, m), which maps to alpha ∈ (1 - m, 1 + m)
            Ex: magnitude=0.2 → alpha ∈ [0.8, 1.2]
        p (float): Probability of applying the transform.

    Returns:
        Albumentations transform that applies random contrast adjustment only.
    """
    assert 0.0 < magnitude <= 1.0
    contrast_limit = (-magnitude, magnitude)
    return A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=contrast_limit,
                                      brightness_by_max=True, ensure_safe_range=True, p=p)


def random_noise(magnitude: float = 0.2, p: float = 1.0):
    """
    Randomly apply additive (Gaussian), multiplicative, or Poisson (shot) noise to the input image.

    Args:
        magnitude (float): Maximum strength of noise in range (0.0, 1.0]. Higher value means stronger noise.
        p (float): Probability of applying the transform.

    Returns:
        Albumentations transform that applies one of the noise types.
    """
    assert 0.0 < magnitude <= 1.0, "magnitude must be in (0.0, 1.0]"

    # ===== Gaussian noise =====
    # std_range is relative to max pixel value (e.g., 255)
    # Rule: magnitude = 1.0 → std_range = (0.0, 0.2)
    # This means noise std = up to 20% of image intensity at most.
    std_max = 0.2 * magnitude  # max ~0.2 (e.g. 0.04 when magnitude=0.2)
    gaussian_noise = A.GaussNoise(std_range=(0.0, std_max), mean_range=(0.0, 0.0), p=1.0)

    # ===== Multiplicative noise =====
    # multiplier = image * random_factor
    # Rule: magnitude = 1.0 → multiplier = (0.5, 1.5)
    # i.e., up to 50% scaling in either direction
    m = magnitude * 0.5  # scale factor
    multiplicative_noise = A.MultiplicativeNoise(
        multiplier=(1 - m, 1 + m), per_channel=False, elementwise=True, p=1.0
    )

    # ===== Shot (Poisson) noise =====
    # Poisson noise is scale-dependent. Higher scale → more noise.
    # Rule: magnitude = 1.0 → scale_range = (0.5, 1.5)
    # Reason: scale ~ inverse of photon count (higher = noisier)
    poisson_scale = 1.0 * magnitude  # conservative scaling
    shot_noise = A.ShotNoise(scale_range=(0.1, 0.1 + poisson_scale), p=1.0)

    # ===== Combine =====
    return A.OneOf([gaussian_noise, multiplicative_noise, shot_noise], p=p)


def random_compression(magnitude: float = 0.2, p: float = 1.0):
    """
    Randomly compress the input image using JPEG compression.

    Args:
        magnitude (float): Compression strength in range (0.0, 1.0].
                           Higher value = stronger compression (lower quality).
                           Adjusts both lower and upper bounds of quality.
        p (float): Probability of applying the transform.

    Returns:
        Albumentations transform applying JPEG compression with controlled quality range.
    """
    assert 0.0 < magnitude <= 1.0

    # Compression quality ranges from 1 (worst) to 100 (best)
    # We'll adjust both lower and upper bounds:
    #
    #   - Lower bound: more compression → subtract up to 90
    #   - Upper bound: moderate compression → subtract up to 30
    #
    # Ex:
    #   magnitude = 0.2 → (82, 94)
    #   magnitude = 0.5 → (55, 85)
    #   magnitude = 1.0 → (10, 70)

    min_quality = max(1, int(100 - magnitude * 90))   # up to 90 decrease
    max_quality = max(min_quality, int(100 - magnitude * 30))  # upper still compressed

    return A.ImageCompression(quality_range=(min_quality, max_quality), compression_type='jpeg', p=p)


def random_clahe(magnitude: float = 0.2, p=1.0):
    """
    Randomly apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        magnitude (float): Controls the clip limit range for local contrast enhancement.
                           Formula: clip_limit = (1.0, 1.0 + magnitude * (max_clip - 1.0))
                           E.g., magnitude=0.2 → (1.0, 2.4)
        p (float): Probability of applying the transform.
    """
    assert 0.0 < magnitude <= 1.0
    max_clip = 8.0
    clip_limit_upper = 1.0 + magnitude * (max_clip - 1.0)  # max = 8.0 when magnitude=1.0
    clip_limit = (1.0, clip_limit_upper)
    return A.CLAHE(clip_limit=clip_limit, tile_grid_size=(8, 8), p=p)


def random_hist_equal(magnitude: float = 0.2, p=1.0):
    """
    Randomly apply histogram equalization (OpenCV or PIL mode).
    Magnitude has no effect.

    Args:
        magnitude: Not used, kept for API consistency.
        p (float): Probability of applying the transform.
    """
    t1 = A.Equalize(mode='cv')
    t2 = A.Equalize(mode='pil')
    return A.OneOf([t1, t2], p=p)


AVAILABLE_TRANSFORMS = {'random_blur': random_blur,
                        'random_brightness': random_brightness,
                        'random_clahe': random_clahe,
                        'random_compression': random_compression,
                        'random_contrast': random_contrast,
                        'random_gamma': random_gamma,
                        'random_hist_equal': random_hist_equal,
                        'random_noise': random_noise,
                        'random_inverse': random_inverse
                        }


class RandAugmentPixel(ImageOnlyTransform):
    """
    Apply multiple random pixel-level augmentations with configurable strength.

    This implements a simplified version of RandAugment for pixel-level transforms,
    allowing control over the number of transforms and their magnitude.

    Args:
        transforms: Configuration for transforms to use.
            Can be either:
            - List of transform names from AVAILABLE_TRANSFORMS
            - Dict mapping transform names to their kwargs
        min_n: Minimum number of transforms to apply.
        max_n: Maximum number of transforms to apply.
        magnitude: Global magnitude value to use for all transforms if not specified
                  in transform kwargs. Range (0.0, 1.0].
        replace: Whether to sample transforms with replacement.
        p: Probability of applying the overall transform.

    Example:
        # Using list of transform names with global magnitude
        augmenter = RandAugmentPixel(
            transforms=['random_blur', 'random_contrast'],
            min_n=1, max_n=2, magnitude=0.5
        )

        # Using dict with per-transform configuration
        augmenter = RandAugmentPixel(
            transforms={
                'random_blur': {'magnitude': 0.3},
                'random_contrast': {'magnitude': 0.5, 'p': 0.8}
            },
            min_n=1, max_n=2
        )
    """

    def __init__(self,
                 transforms: list[str] | dict[str, dict[str, Any]],
                 min_n: int = 1, max_n: int = 3,
                 magnitude: float = 0.3,
                 replace: bool = False,
                 p: float = 1.0):
        super().__init__(always_apply=False, p=p)

        self.min_n = min_n
        self.max_n = max_n
        self.replace = replace
        self.magnitude = magnitude

        # 변환 함수 인스턴스화
        self.transforms = []

        # 입력이 변환 이름 리스트인 경우
        if isinstance(transforms, list):
            for name in transforms:
                if name not in AVAILABLE_TRANSFORMS:
                    raise ValueError(f"Unknown transform: {name}")
                self.transforms.append(AVAILABLE_TRANSFORMS[name](magnitude=magnitude, p=1.0))
        # 입력이 변환 이름 -> 설정 매핑인 경우
        elif isinstance(transforms, dict):
            for name, kwargs in transforms.items():
                if name not in AVAILABLE_TRANSFORMS:
                    raise ValueError(f"Unknown transform: {name}")
                # p값이 지정되지 않았으면 기본값 1.0 사용
                if 'p' not in kwargs:
                    kwargs['p'] = 1.0
                # magnitude가 지정되지 않았으면 전역 magnitude 사용
                if 'magnitude' not in kwargs and name != 'random_inverse':
                    kwargs['magnitude'] = magnitude
                self.transforms.append(AVAILABLE_TRANSFORMS[name](**kwargs))
        else:
            raise ValueError("transforms must be a list of names or a dict mapping names to kwargs")

        if not self.transforms:
            raise ValueError("No valid transforms provided")

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply randomly selected transforms to the image."""
        # 적용할 변환 개수 선택
        n = random.randint(self.min_n, min(self.max_n, len(self.transforms)))

        # 변환 선택 방식: 중복 허용 vs 불허
        if self.replace:
            selected = random.choices(self.transforms, k=n)
        else:
            selected = random.sample(self.transforms, k=n)

        # 선택된 변환 적용
        aug = A.Compose(selected)
        augmented = aug(image=img)["image"]
        return augmented

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the parameter names that should be passed to __init__."""
        return ('min_n', 'max_n', 'magnitude', 'transforms', 'replace', 'p')


# Convenience function to create enhanced RandAugmentPixel with default settings
def create_rand_augment(magnitude: float = 0.3, min_n: int = 1, max_n: int = 3) -> RandAugmentPixel:
    """
    Create a RandAugmentPixel instance with all available transforms.

    Args:
        magnitude: Global magnitude for all transforms (0.0, 1.0].
        min_n: Minimum number of transforms to apply.
        max_n: Maximum number of transforms to apply.

    Returns:
        Configured RandAugmentPixel transform.
    """
    transforms = list(AVAILABLE_TRANSFORMS.keys())
    return RandAugmentPixel(transforms=transforms,
                            min_n=min_n,
                            max_n=max_n,
                            magnitude=magnitude,
                            replace=True
                            )
