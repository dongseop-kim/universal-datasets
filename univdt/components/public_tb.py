from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from univdt.components.base import BaseComponent
from univdt.utils.image import load_image


class PublicTuberculosis(BaseComponent):
    """
    Public Tuberculosis dataset, which is a collection of 4 datasets:
    - DADB or TBXPredict
    - Shenzhen 
    - Montgomery
    - TBX11K 

    Args:
        root : root folder for dataset
        path_annotation : path to annotation file
        split : 'train', 'val', 'test' and 'trainval'
        transform : Composed transforms
    """
    _AVAILABLE_SPLITS = ['train', 'val', 'test', 'trainval']

    def __init__(self, root_dir: str, split: str,
                 path_annotation:str, transform=None):
        super().__init__(root_dir, split, transform)
        self._check_split(self._AVAILABLE_SPLITS)

        # TODO: add transform check here or in the base class
        
        self.path_annotation =  Path(self.root_dir) / path_annotation
        self.raw_data = self._load_paths(self.path_annotation)

    def __getitem__(self, index) -> dict[str, Any]:
        data = self._load_data(index)
        image: np.ndarray = data['image'] # Normalized to [0, 255]
        
        
        if self.transform is not None: 
            transformed = self.transform(image=image)
            image = transformed['image']

        # convert image to pytorch tensor and normalize to [0, 1]
        image = self._to_tensor(image)  
        output = {'image': image} | {key: data[key] for key in data if key != 'image'}
        return output

    def _load_paths(self, path_csv:str) -> list[dict[str, Any]]:
        df = pd.read_csv(path_csv)
        df = df[df['split'].isin([self.split])] if self.split != 'trainval' \
            else df[df['split'].isin(['train', 'val'])]
        return [dict(row) for _, row in df.iterrows()]

    def __len__(self) -> int:
        return len(self.raw_data)

class TuberculosisDADB(PublicTuberculosis):
    def __init__(self, root_dir: str, split: str, transform=None):
        super().__init__(root_dir, split, 'tbxpredict.csv', transform)
        self.dataset = 'dadb'
        self.num_classes = 2
        self.void_class = 255
    
    def _load_data(self, index:int) -> dict[str, Any]:
        raw_data:dict[str, Any] = self.raw_data[index]
        path_image:Path = Path(self.root_dir) / raw_data['name']
        assert path_image.exists(), f'Image {path_image} does not exist'
        image:np.ndarray = load_image(path_image, out_channels=1) # normalized to [0, 255]
        label:str = raw_data['label'].lower()
        assert label in ['normal', 'activetb'], f'Invalid label: {label}'

        processed_data = {'image': image, 'label': label, 
                          'dataset':self.dataset, 'path': str(path_image)}
        return processed_data


class TuberculosisTBXPredict(TuberculosisDADB):
    # same as DADB
    pass


class TuberculosisShenzhen(PublicTuberculosis):
    _AVAILABLE_KEYS = ['age', 'gender', 'report']
    def __init__(self, root_dir: str, split: str,  transform=None, 
                 additional_keys: Optional[list[str]] = None):
        super().__init__(root_dir, split, 'shenzhen.csv', transform)
        
        self.dataset = 'shenzhen'
        self.additional_keys = additional_keys if additional_keys is not None \
            else self._AVAILABLE_KEYS
        assert all([key in self._AVAILABLE_KEYS for key in self.additional_keys]), \
            f'Invalid additional keys: {self.additional_keys}'
        
        self.num_classes = 2 # tuberculosis, normal
        self.void_class = 255 # set void class as 255

    def _load_data(self, index:int) -> dict[str, Any]:
        raw_data:dict[str, Any] = self.raw_data[index]
        path_image:Path = Path(self.root_dir) / raw_data['name']
        assert path_image.exists(), f'Image {path_image} does not exist'
        image:np.ndarray = load_image(path_image, out_channels=1) # normalized to [0, 255]
        label:str = raw_data['label'].lower()
        assert label in ['normal', 'activetb'], f'Invalid label: {label}'

        processed_data = {'image': image, 'label': label, 
                          'dataset':self.dataset, 'path': str(path_image)}
        return processed_data | {key: raw_data[key] for key in self.additional_keys}



class TuberculosisMontgomery(PublicTuberculosis):
    _AVAILABLE_KEYS = ['age', 'gender', 'report']
    def __init__(self, root_dir:str, split:str, transform=None,
                 additional_keys: Optional[list[str]] = None):
        super().__init__(root_dir, split, 'montgomery.csv', transform)
        self.dataset = 'montgomery'
        
        self.additional_keys = additional_keys if additional_keys is not None \
            else self._AVAILABLE_KEYS
        assert all([key in self._AVAILABLE_KEYS for key in self.additional_keys]), \
            f'Invalid additional keys: {self.additional_keys}'
        
        self.num_classes = 2
        self.void_class = 255
    
    def _load_data(self, index:int) -> dict[str, Any]:
        raw_data:dict[str, Any] = self.raw_data[index]
        path_image:Path = Path(self.root_dir) / raw_data['name']
        assert path_image.exists(), f'Image {path_image} does not exist'
        image:np.ndarray = load_image(path_image, out_channels=1) # normalized to [0, 255]
        label:str = raw_data['label'].lower()
        assert label in ['normal', 'activetb'], f'Invalid label: {label}'

        processed_data = {'image': image, 'label': label, 
                          'dataset':self.dataset, 'path': str(path_image)}
        return processed_data | {key: raw_data[key] for key in self.additional_keys}



class TuberculosisTBX11K(PublicTuberculosis):
    def __init__(self, root_dir:str, split:str, transform=None):
        super().__init__(root_dir, split, 'tbx11k.csv', transform)
        self.dataset = 'tbx11k'
        self.num_classes = 3 # normal, active, inactive, others
        self.void_class = 255


    def _load_data(self, index:int) -> dict[str, Any]:
        raw_data:dict[str, Any] = self.raw_data[index]
        path_image:Path = Path(self.root_dir) / raw_data['name']
        assert path_image.exists(), f'Image {path_image} does not exist'
        image:np.ndarray = load_image(path_image, out_channels=1) # normalized to [0, 255]
        label:str = raw_data['label'].lower()
        assert label in ['normal', 'activetb', 'inactivetb', 'both', 'others'], \
            f'Invalid label: {label}'

        processed_data = {'image': image, 'label': label, 
                          'dataset':self.dataset, 'path': str(path_image)}
        return processed_data

