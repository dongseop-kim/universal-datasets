from collections import namedtuple

from ..utils import retrieve as dr
"""
available internal dataset : 
    161128_amc
    180116_snubh
"""

PXIDataset = namedtuple('PXIDataset',
                        ['name', 'abbr', 'id'])

Internelsets = [PXIDataset('161128_amc', 'amc_01', 0),
                PXIDataset('180116_snubh', 'snubh_01', 1),
                ]


class VUNOInternalSet:
    available_split = ['train', 'val', 'trainval', 'test']

    def __init__(self, task: str):
        self.task = dr.TaskType(task)

    def __getitem__(self, index):
        pass

    def __load__(self, index):
        pass
