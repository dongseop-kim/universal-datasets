import albumentations as A
import cv2

"""
cv2.BORDER_CONSTANT     : 0
cv2.BORDER_REPLICATE    : 1
cv2.BORDER_REFLECT      : 2
cv2.BORDER_WRAP         : 3
cv2.BORDER_REFLECT_101  : 4
cv2.BORDER_TRANSPARENT  : 5
"""


def random_translate(translate_x: int = 15, translate_y: int = 15, p: float = 0.5):
    """ Randomly translate the image by the given percentage.
    Args:
        translate_x (int): percentage of the image width to translate. (default: 15)
        translate_y (int): percentage of the image height to translate. (default: 15)
        p (float): probability of applying the transform. (default: 0.5)
    """
    x_ratio = translate_x / 100
    y_ratio = translate_y / 100
    transform = A.Affine(translate_percent={'x': (-x_ratio, x_ratio),
                                            'y': (-y_ratio, y_ratio)}, p=p)
    return transform


def random_rotate(rotate: int = 45, p: float = 0.5):
    """ Randomly rotate the image by the given angle.
    Args:
        rotate (int): angle of the rotation in degrees (default: -45 ~ 45)
        p (float): probability of applying the transform. (default: 0.5)
    """
    def create_rotate(border_mode):
        params = {'limit': rotate,
                  'interpolation': cv2.INTER_LINEAR,
                  'p': 1.0,
                  'border_mode': border_mode}
        if border_mode == cv2.BORDER_CONSTANT:
            params['value'] = 0
            params['mask_value'] = 0
        return A.Rotate(**params)

    rotate1 = create_rotate(cv2.BORDER_CONSTANT)
    rotate2 = create_rotate(cv2.BORDER_REFLECT_101)
    return A.OneOf([rotate1, rotate2], p=p)


def random_shear(x: int = 20, y: int = 20, p=0.5):
    """ Randomly shear the image by the given angle.
    Args:
        x (int): angle of the shear in degrees along x-axis
        y (int): angle of the shear in degrees along y-axis
        p (float): probability of applying the transform. (default: 0.5)
    """
    def create_shear(border_mode):
        params = {'shear': {'x': (-x, x), 'y': (-y, y)}, 'mode': border_mode, 'p': 1.0}
        if params['mode'] == cv2.BORDER_CONSTANT:
            params['cval'], params['cval_mask'] = 0, 0
        return A.Affine(**params)

    shear1 = create_shear(cv2.BORDER_CONSTANT)
    shear2 = create_shear(cv2.BORDER_REFLECT_101)
    return A.OneOf([shear1, shear2], p=p)


def random_spatial_augs(n: int = 2,
                        rotate: int = 45,
                        shear_x: int = 20, shear_y: int = 20,
                        translate_x: int = 15, translate_y: int = 15,
                        p=0.5):
    return A.SomeOf([random_rotate(rotate), random_shear(shear_x, shear_y), random_translate(translate_x, translate_y)],
                    n=n, p=p)
