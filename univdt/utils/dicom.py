from datetime import datetime
from pathlib import Path
from typing import Any

import pydicom as pyd


def get_age(dicom: pyd.FileDataset) -> int | None:
    age: str = dicom.get('PatientAge', dicom.get((0x0010, 0x1010), None))
    if not age:
        return None

    if age.isnumeric():
        return int(age)

    age = str(age).lower().strip()
    age_value, unit = age[:-1], age[-1]
    if unit in ['y', 'd', 'm'] and age_value.isnumeric():
        age_value = int(age_value)
        if unit == 'y':
            return age_value
        elif unit == 'd':
            return age_value // 365
        elif unit == 'm':
            return age_value // 12
    return None


def get_gender(dicom: pyd.FileDataset) -> str | None:
    def remove_duplicate_chars(input_str):
        from itertools import groupby
        return ''.join(char for char, _ in groupby(input_str))

    gender = dicom.get('PatientSex', dicom.get((0x0010, 0x0040), None))
    if gender is None:
        return None

    gender = remove_duplicate_chars(str(gender).lower().strip())
    if gender == 'f' or gender == 'm':
        return gender
    else:
        return None


def get_cuid(dicom: pyd.FileDataset) -> str | None:
    #  Get class uid from dicom file.
    cuid = dicom.get('SOPClassUID', dicom.get((0x0008, 0x0016), None))
    return str(cuid).strip() if cuid else None


def get_iuid(dicom: pyd.FileDataset) -> str | None:
    # Get instance uid from dicom file.
    iuid = dicom.get('SOPInstanceUID', dicom.get((0x0008, 0x0018), None))
    return str(iuid).strip() if iuid else None


def get_shape(dicom: pyd.FileDataset) -> tuple[int, int] | tuple[None, None]:
    height = int(dicom.get('Rows', dicom.get((0x0028, 0x0010), None)))
    width = int(dicom.get('Columns', dicom.get((0x0028, 0x0011), None)))
    return height, width


def get_study_date(dicom: pyd.FileDataset) -> datetime | None:
    """ Get study date from dicom file.

    Args:
        dicom(pyd.FileDataset): dicom file

    Returns:
        study_date(str): study date of dicom file, if not found, return 0000.00.00
    """
    study_date = dicom.get('StudyDate', dicom.get((0x0008, 0x0020), None))
    if study_date is None:
        return None

    study_date = str(study_date).strip()
    if len(study_date) != 8 or not study_date.isnumeric():
        return None

    try:
        return datetime.strptime(study_date, '%Y%m%d')
    except ValueError:
        return None


def get_view_position(dicom: pyd.FileDataset) -> str | None:
    #  Get view position from dicom file.
    view_position = dicom.get('ViewPosition', dicom.get((0x0018, 0x5101), None))
    if view_position is None:
        return None
    view_position = str(view_position).strip().lower()
    if view_position not in ['ap', 'pa']:
        return None
    return view_position


def get_meta_from_dicom(dicom: str | Path | pyd.FileDataset, key: str) -> Any:
    if isinstance(dicom, (str, Path)):
        dicom = pyd.dcmread(str(dicom), force=True)

    match key:
        case 'age':
            return get_age(dicom)
        case 'gender' | 'sex':
            return get_gender(dicom)
        case 'i_uid' | 'instance_uid' | 'iuid' | 'i-uid' | 'instance-uid':
            return get_iuid(dicom)
        case 'c_uid' | 'class_uid' | 'cuid' | 'c-uid' | 'class-uid':
            return get_cuid(dicom)
        case 'shape':
            return get_shape(dicom)
        case 'height' | 'rows':
            return get_shape(dicom)[0]
        case 'width' | 'columns':
            return get_shape(dicom)[1]
        case 'study_date' | 'studydate' | 'study-date':
            return get_study_date(dicom)
        case 'view_position' | 'viewposition' | 'view-position':
            return get_view_position(dicom)
        case _:
            raise ValueError(f"Unsupported key: {key}")
