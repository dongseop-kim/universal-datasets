import random
from typing import Any

import albumentations as A
import albumentations.augmentations.crops.functional as crop_f
import albumentations.augmentations.geometric.functional as geo_f
import albumentations.augmentations.geometric.functional as gf
import albumentations.core.bbox_utils as bbox_utils
from albumentations.core.transforms_interface import DualTransform
import cv2
import numpy as np

DEFAULT_PAD_VAL = 0
DEFAULT_PAD_VAL_MASK = 0


class RandomResize(DualTransform):
    """
    Randomly apply one of several resize methods, including Letterbox.

    Args:
        height (int): Target height for resizing.
        width (int): Target width for resizing.
        scale (float): Scale factor for additional size variation.
        interpolations (List[str]): List of interpolation methods to use.
        include_letterbox (bool): Whether to include Letterbox in resize options.
        letterbox_pad_val (int): Padding value for Letterbox (for image).
        letterbox_pad_val_mask (int): Padding value for Letterbox (for mask).
        always_apply (bool): Whether to always apply the transform.
        p (float): Probability of applying the transform.
    """

    def __init__(self, height: int = 768, width: int = 768, scale: float = 0.1,
                 interpolations: list[str] = ['nearest', 'linear', 'cubic', 'area', 'lanczos'],
                 letterbox_pad_val: int = DEFAULT_PAD_VAL, letterbox_pad_val_mask: int = DEFAULT_PAD_VAL_MASK,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__()
        self.height = height
        self.width = width
        self.scale = scale
        self.interpolations = interpolations
        self.letterbox_pad_val = letterbox_pad_val
        self.letterbox_pad_val_mask = letterbox_pad_val_mask
        self.always_apply = always_apply
        self.p = p
        self.interpolation_methods: dict[str, int] = {'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR,
                                                      'cubic': cv2.INTER_CUBIC, 'area': cv2.INTER_AREA,
                                                      'lanczos': cv2.INTER_LANCZOS4}
        self._create_transforms()

    def _create_transforms(self):
        """Create the list of resize transforms."""
        self.resize_transforms = [A.Resize(height=self.height, width=self.width,
                                           interpolation=self.interpolation_methods[interpolation])
                                  for interpolation in self.interpolations]
        self.resize_transforms.append(Letterbox(height=self.height, width=self.width,
                                                pad_val=self.letterbox_pad_val,
                                                pad_val_mask=self.letterbox_pad_val_mask))
        self.transform = A.OneOf(self.resize_transforms, p=1.0 if self.always_apply else self.p)

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the transform to an image."""
        return self.transform(image=image, **kwargs)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Get the names of the arguments used in __init__."""
        return ('height', 'width', 'scale', 'interpolations',)


class Letterbox(A.DualTransform):
    """
    Resize the input to the specified height and width while maintaining the aspect ratio.
    Padding is added to the image to preserve the aspect ratio.

    Args:
        height (int): Desired height of the output image.
        width (int): Desired width of the output image.
        pad_val (int): Padding value for image if border_mode is cv2.BORDER_CONSTANT.
        pad_val_mask (int): Padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        always_apply (bool): Whether to always apply the transform.
        p (float): Probability of applying the transform.
    """

    def __init__(self, height: int, width: int,
                 pad_val: int = DEFAULT_PAD_VAL,
                 pad_val_mask: int = DEFAULT_PAD_VAL_MASK,
                 always_apply: bool = False,
                 p: float = 1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.pad_val = pad_val
        self.pad_val_mask = pad_val_mask

    @staticmethod
    def get_pad_size(size: int, target_size: int) -> tuple[int, int]:
        """Calculate padding size for a given dimension."""
        if size > target_size:
            return 0, 0
        pad = target_size - size
        pad_a, pad_b = pad // 2, pad - pad // 2
        return pad_a, pad_b

    def update_params(self, params: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Update transform parameters."""
        params = super().update_params(params, **kwargs)
        h, w = params['rows'], params['cols']
        scale = min(self.width / w, self.height / h)
        target_h, target_w = int(h * scale), int(w * scale)

        pad_top, pad_bottom = self.get_pad_size(target_h, self.height)
        pad_left, pad_right = self.get_pad_size(target_w, self.width)

        params.update({'scale': scale, 'target_h': target_h, 'target_w': target_w,
                       'h_pad_top': pad_top, 'h_pad_bottom': pad_bottom,
                       'w_pad_left': pad_left, 'w_pad_right': pad_right})
        return params

    def get_transform_init_args_names(self):
        return ('height', 'width', 'pad_val', 'pad_val_mask')

    def apply(self, img: np.ndarray, interpolation: int = cv2.INTER_LINEAR, **params) -> np.ndarray:
        """Apply the transform to an image."""
        img = gf.resize(img, params['target_h'], params['target_w'], interpolation)
        return self._pad_image(img, params, self.pad_val)

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply the transform to a mask."""

        ('mask:', params)
        img = gf.resize(img, params['target_h'], params['target_w'], cv2.INTER_NEAREST)
        return self._pad_image(img, params, self.pad_val_mask)

    def apply_to_bbox(self, bbox: list[float], **params) -> list[float]:
        """Apply the transform to a bounding box."""
        h, w = params['target_h'], params['target_w']
        x_min, y_min, x_max, y_max = bbox_utils.denormalize_bbox(bbox, h, w)
        x_min, x_max = x_min + params['w_pad_left'], x_max + params['w_pad_left']
        y_min, y_max = y_min + params['h_pad_top'], y_max + params['h_pad_top']
        return bbox_utils.normalize_bbox((x_min, y_min, x_max, y_max),
                                         h + params['h_pad_top'] + params['h_pad_bottom'],
                                         w + params['w_pad_left'] + params['w_pad_right'])

    def apply_to_keypoint(self, keypoint: tuple[float, float, float, float], **params) -> tuple[float, float, float, float]:
        """Apply the transform to a keypoint."""
        scale_x, scale_y = self.width / params['cols'], self.height / params['rows']
        keypoint = gf.keypoint_scale(keypoint, scale_x, scale_y)
        x, y, angle, scale = keypoint
        return x + params['w_pad_left'], y + params['h_pad_top'], angle, scale

    def _pad_image(self, img: np.ndarray, params: dict[str, Any], pad_val: int) -> np.ndarray:
        """Pad the image with the specified value."""
        return gf.pad_with_params(img=img,
                                  h_pad_top=params['h_pad_top'], h_pad_bottom=params['h_pad_bottom'],
                                  w_pad_left=params['w_pad_left'], w_pad_right=params['w_pad_right'],
                                  border_mode=cv2.BORDER_CONSTANT, value=pad_val)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Get the names of the arguments used in __init__."""
        return ('height', 'width', 'pad_val', 'pad_val_mask')


class RandomResizeCrop(DualTransform):
    def __init__(self, height: int, width: int, scale: float = 0.2, interpolation: int = cv2.INTER_LINEAR,
                 border_mode: int = cv2.BORDER_CONSTANT, always_apply: bool = False, p: float = 1.0) -> None:
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.scale_limit: tuple[float, float] = (1.0 - scale, 1.0 + scale)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value: int = 0
        self.mask_value: int = 0

    def get_params(self):
        return {'scale': random.uniform(self.scale_limit[0], self.scale_limit[1]),
                'h_start': random.random(), 'w_start': random.random()}

    def _calculate_padding(self, size: int, target_size: int) -> tuple[int, int]:
        """Calculate padding for the given size to match the target size."""
        if size < target_size:
            pad_top: int = int((target_size - size) / 2.0)
            pad_bottom: int = target_size - size - pad_top
        else:
            pad_top, pad_bottom = 0, 0
        return pad_top, pad_bottom

    def update_params(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        params = super().update_params(params, **kwargs)
        height: int = params['rows']
        width: int = params['cols']
        scale: float = params['scale']
        rows, cols = int(height * scale), int(width * scale)
        h_pad_top, h_pad_bottom = self._calculate_padding(rows, self.height)
        w_pad_left, w_pad_right = self._calculate_padding(cols, self.width)

        params.update({'rows': rows, 'cols': cols,
                       'pad_top': h_pad_top, 'pad_bottom': h_pad_bottom,
                       'pad_left': w_pad_left, 'pad_right': w_pad_right})
        return params

    def _resize_and_crop(self, img: np.ndarray, scale: float, h_start: float, w_start: float,
                         pad_top: int, pad_bottom: int, pad_left: int, pad_right: int, interpolation: int) -> np.ndarray:
        """Resize the image and apply cropping or padding."""
        img = geo_f.scale(img, scale, interpolation)
        if scale >= 1.0:
            return crop_f.random_crop(img, self.height, self.width, h_start, w_start)
        else:
            return geo_f.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right,
                                         border_mode=self.border_mode, value=self.value)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return self._resize_and_crop(img, **params)

    def apply_to_mask(self, img: np.ndarray, **params: Any) -> np.ndarray:
        params['interpolation'] = cv2.INTER_NEAREST  # Always use nearest neighbor for masks
        return self._resize_and_crop(img, **params)

    def apply_to_bbox(self, bbox: tuple[float, float, float, float], scale: float,
                      pad_top: int, pad_bottom: int, pad_left: int, pad_right: int,
                      rows: int, cols: int, **params: Any) -> tuple[float, float, float, float]:
        """Resize and pad bounding boxes."""
        if scale >= 1.0:
            return crop_f.bbox_random_crop(bbox, self.height, self.width, rows=rows, cols=cols, **params)
        else:
            x_min, y_min, x_max, y_max = bbox_utils.denormalize_bbox(bbox, rows, cols)
            bbox = (x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top)
            return bbox_utils.normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    def apply_to_keypoint(self, keypoint: tuple[float, float, float, float], scale: float,
                          pad_top: int, pad_bottom: int, pad_left: int, pad_right: int, **params: Any) -> tuple[float, float, float, float]:
        """Resize and pad keypoints."""
        keypoint = geo_f.keypoint_scale(keypoint, scale, scale)
        if scale >= 1.0:
            return crop_f.keypoint_random_crop(keypoint, self.height, self.width, **params)
        else:
            x, y, angle, scale = keypoint
            return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ('height', 'width', 'scale_limit', 'interpolation',)
