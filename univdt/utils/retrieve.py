from pathlib import Path


def load_yaml(filepath: str):
    import yaml
    with open(filepath, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_config(filepath: str):
    from omegaconf import DictConfig
    cfg_file = DictConfig(load_yaml(filepath))
    return cfg_file


def load_pickle(filepath: str):
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_csv(filepath: str, **kwargs):
    import pandas
    return pandas.read_csv(filepath, **kwargs)


def load_json(filepath: str):
    import json
    with open(filepath, 'r') as f:
        return json.load(f)


def load_dicom(filepath):
    import pydicom
    return pydicom.dcmread(filepath)


def load_file(path: str | Path, **kwargs):
    """load file based on extension.
    Args:
        path (str | Path): path to file.

    Returns:
        object: loaded file.
    """
    loaders = {'.yaml': load_yaml,
               '.cfg': load_config,
               '.csv': load_csv,
               '.dcm': load_dicom, 'dicom': load_dicom,
               '.json': load_json,
               '.pkl': load_pickle, '.pickle': load_pickle,
               }
    if isinstance(path, str):
        path = Path(path)
    extension = path.suffix.lower()
    loader = loaders.get(extension)
    if loader:
        return loader(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {path}")
