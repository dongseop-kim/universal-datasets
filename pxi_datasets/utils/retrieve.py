import abc
import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from annotation.annot_loader import AnnotOutput

import pxi_datasets.utils.utils as du


class TaskType(enum.Enum):
    CLASSIFICATION = 'classification'
    DETECTION = 'detection'
    SEGMENTATION = 'segmentation'
    ALL = 'all'


class LabelType(enum.Enum):
    IMAGE = 1
    BBOXES = 2
    MASK = 3


class BaseRetriever:
    def __init__(self, result_key):
        self._result_key = result_key

    @abc.abstractmethod
    def get_value(self, annots: AnnotOutput, extras: Dict[str, Any]):
        """
        Must implement to retrieve value
        """
        pass

    @property
    def key(self):
        return self._result_key


class ImageRetriever(BaseRetriever):
    def __init__(self):
        super().__init__('image')

    def get_value(self, annots: AnnotOutput, extras: Dict[str, Any]):
        image: np.ndarray = du.get_image(annots.path_image)
        return image  # H x W x C ( 1 or 3)


class BboxRetriever(BaseRetriever):
    def __init__(self):
        super().__init__('bboxes')

    def get_value(self, annots: AnnotOutput, extras: Dict[str, Any]):
        bboxes: np.ndarray = du.get_bboxes(annots.bboxes)
        return bboxes  # M x 4


class MaskRetriever(BaseRetriever):
    def __init__(self):
        super().__init__('mask')

    def get_value(self, annots: AnnotOutput, extras: Dict[str, Any]):
        # TODO: Update 필요
        mask: np.ndarray = du.get_masks(annots.path_masks, extras['image'].shape)
        return mask  # N x H x W


_LABEL_TYPE_TO_RETRIEVER: Dict[LabelType, BaseRetriever] = {LabelType.IMAGE: ImageRetriever(),
                                                            LabelType.BBOXES: BboxRetriever(),
                                                            LabelType.MASK: MaskRetriever()}


_TASK_TO_LABEL_TYPES = {
    TaskType.CLASSIFICATION: [LabelType.IMAGE],
    TaskType.DETECTION: [LabelType.IMAGE, LabelType.BBOXES],
    TaskType.SEGMENTATION: [LabelType.IMAGE, LabelType.MASK],
    TaskType.ALL: [LabelType.IMAGE, LabelType.BBOXES, LabelType.MASK]}


@dataclass
class Data:
    image: np.ndarray
    bboxes: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    lesion_classes: List[int]
    path_image: str
    path_dicom: str
    etc_image: Dict[str, Any]
    etc_objects: List[Dict[str, Any]]


def get_data(task_type: TaskType, annots: AnnotOutput) -> Data:
    lesion_classes: List[int] = annots.lesion_classes
    path_image: str = annots.path_image
    path_dicom: str = annots.path_dicom
    etc_image = annots.etc_image
    etc_objects = annots.etc_objects
    data_dict = {'lesion_classes': lesion_classes,
                 'path_image': path_image,
                 'path_dicom': path_dicom,
                 'etc_image': etc_image,
                 'etc_objects': etc_objects}

    for lt in _TASK_TO_LABEL_TYPES[task_type]:
        retriever = _LABEL_TYPE_TO_RETRIEVER[lt]
        data_dict[retriever.key] = retriever.get_value(annots, data_dict)
    return Data(**data_dict)
