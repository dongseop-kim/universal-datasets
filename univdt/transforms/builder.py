from .pixel import (random_blur, random_brightness, random_clahe,
                    random_compression, random_contrast, random_gamma,
                    random_hist_equal, random_noise, random_windowing)
from .resize import random_resize
from .zoom import random_zoom

AVAILABLE_TRANSFORMS = {'random_blur': random_blur,
                        'random_brightness': random_brightness,
                        'random_clahe': random_clahe,
                        'random_compression': random_compression,
                        'random_contrast': random_contrast,
                        'random_gamma': random_gamma,
                        'random_histequal': random_hist_equal,
                        'random_noise': random_noise,
                        'random_resize': random_resize,
                        'random_windowing': random_windowing,
                        'random_zoom': random_zoom, }
