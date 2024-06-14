from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pydicom as pyd


def get_age(dicom: pyd.FileDataset) -> int | None:
    """Get age from dicom file.
    Args:
        dicom (pyd.FileDataset): dicom file
    Returns:
        int: age of patient in dicom file. If age is not found or invalid, return None.
    """
    if not (age := dicom.get('PatientAge', dicom.get((0x0010, 0x1010), None))):
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
    """ Get gender from dicom file.

    Args:
        dicom(pyd.FileDataset): dicom file

    Returns:
        gender(str | None): gender of patient. 
    """
    def remove_duplicate_chars(input_str):
        from itertools import groupby
        return ''.join(char for char, _ in groupby(input_str))

    if (gender := dicom.get('PatientSex', dicom.get((0x0010, 0x0040), None))) is None:
        return None

    gender = remove_duplicate_chars(str(gender).lower().strip())
    if gender == 'f' or gender == 'm':
        return gender
    else:
        return None


def get_cuid(dicom: pyd.FileDataset) -> str | None:
    """ Get class uid from dicom file.
    Args:
        dicom(pyd.FileDataset): dicom file
    Returns:
        class_uid(str): SOP Class UID, if not found, return None
    """
    cuid = dicom.get('SOPClassUID', dicom.get((0x0008, 0x0016), None))
    return str(cuid).strip() if cuid else None


def get_iuid(dicom: pyd.FileDataset) -> str | None:
    """ Get instance uid from dicom file.

    Args:
        dicom(pyd.FileDataset): dicom file
    Returns:
        instance_uid(str): SOP Instance UID, if not found, return None
    """
    iuid = dicom.get('SOPInstanceUID', dicom.get((0x0008, 0x0018), None))
    return str(iuid).strip() if iuid else None


def get_shape(dicom: pyd.FileDataset) -> tuple[int, int] | tuple[None, None]:
    """ Get shape from dicom file.

    Args:
        dicom(pyd.FileDataset): dicom file

    Returns:
        shape(tuple[int, int]): (height, width) of dicom file, if not found, return [None, None]
    """
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
    if (study_date := dicom.get('StudyDate', dicom.get((0x0008, 0x0020), None))) is None:
        return None

    study_date = str(study_date).strip()
    if len(study_date) != 8 or not study_date.isnumeric():
        return None

    try:
        return datetime.strptime(study_date, '%Y%m%d')
    except ValueError:
        return None


def get_view_position(dicom: pyd.FileDataset) -> str | None:
    """ Get view position from dicom file.

    Args:
        dicom(pyd.FileDataset): dicom file

    Returns:
        view_position(str): view position of dicom file, if not found, return None
    """
    if (view_position := dicom.get('ViewPosition', dicom.get((0x0018, 0x5101), None))) is None:
        return None
    view_position = str(view_position).strip().lower()
    if view_position not in ['ap', 'pa']:
        return None
    return view_position


def get_pixel_spacing(dicom: pyd.FileDataset) -> tuple[float, float] | tuple[None, None]:
    '''
    Get pixel spacing (x, y) from a DICOM file.

    Determines the relationship between Pixel Spacing (0028,0030) and 
    Imager Pixel Spacing (0018,1164) or Nominal Scanned Pixel Spacing (0018,2010).

    If Pixel Spacing is present:
        - Matches Imager or Nominal Scanned Pixel Spacing: No geometric magnification correction.
        - Differs from Imager or Nominal Scanned Pixel Spacing: Corrected for geometric magnification or calibrated for known object size/depth.

    Returns:
        A tuple of two floats representing the pixel spacing (x, y), or None if not available.
    '''
    if (pixel_spacing := dicom.get('PixelSpacing', dicom.get((0x0028, 0x0030), None))) is None:
        return None, None

    if isinstance(pixel_spacing, Iterable):
        pixel_spacing = list(map(float, pixel_spacing))  # Convert to float
        if len(pixel_spacing) == 1:
            return pixel_spacing[0], pixel_spacing[0]
        elif len(pixel_spacing) == 2:
            # pixel_space_y, pixel_space_x = pixel_spacing
            return pixel_spacing[1], pixel_spacing[0]

    if isinstance(pixel_spacing, float):
        return pixel_spacing, pixel_spacing

    return None, None


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
        case 'pixel_spacing' | 'pixelspacing' | 'pixel-spacing':
            return get_pixel_spacing(dicom)
        case 'pixel_spacing_x' | 'pixelspacing_x' | 'pixel-spacing_x' | 'pixel_spacing-x' | 'pixelspacing-x' | 'pixel-spacing-x':
            return get_pixel_spacing(dicom)[0]
        case 'pixel_spacing_y' | 'pixelspacing_y' | 'pixel-spacing_y' | 'pixel_spacing-y' | 'pixelspacing-y' | 'pixel-spacing-y':
            return get_pixel_spacing(dicom)[1]
        case _:
            raise ValueError(f"Unsupported key: {key}")
