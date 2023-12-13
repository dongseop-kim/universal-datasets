from udt.components._base import BaseComponent


class CamVid(BaseComponent):
    _INFO = {}
    _TASK = 'segmentation'

    def __init__(self,
                 root_dir: str,
                 split: str,
                 transform=None):
        assert split in ['train', 'val', 'trainval', 'test']
        pass

    def _load_paths(self):
        pass
