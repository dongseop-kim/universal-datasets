from pathlib import Path


def load_yaml(filepath: str, **kwargs):
    import yaml
    with open(filepath, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader, **kwargs)


def load_config(filepath: str, **kwargs):
    from omegaconf import DictConfig
    cfg_file = DictConfig(load_yaml(filepath, **kwargs))
    return cfg_file


def load_pickle(filepath: str, **kwargs):
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f, **kwargs)


def load_csv(filepath: str, **kwargs):
    import pandas
    return pandas.read_csv(filepath, **kwargs)


def load_json(filepath: str, **kwargs):
    import json
    with open(filepath, 'r') as f:
        return json.load(f, **kwargs)


def load_dicom(filepath, **kwargs):
    import pydicom
    return pydicom.dcmread(filepath, **kwargs)


def load_file(path: str | Path, **kwargs):
    """load file based on extension.
    Args:
        path (str | Path): path to file.

    Returns:
        object: loaded file.
    """
    if isinstance(path, str):
        path = Path(path)
    extension = path.suffix.lower()
    if extension == '.yaml':
        return load_yaml(path, **kwargs)
    elif extension == '.cfg':
        return load_config(path, **kwargs)
    elif extension == '.csv':
        return load_csv(path, **kwargs)
    elif extension in ['.dcm', 'dicom']:
        return load_dicom(path, **kwargs)
    elif extension == '.json':
        return load_json(path, **kwargs)
    elif extension in ['.pkl', '.pickle']:
        return load_pickle(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {path}")
