from datetime import datetime
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pydicom as pyd

from univdt.utils import dicom as ud
from univdt.utils import image as ui
from univdt.utils import retrieve as ur
from univdt.utils.logger import Logger

logger = Logger(__name__, 0)

# np.random.seed(0)  # set random seed for reproducibility
TEST_IMAEG = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)


def test_load_image():
    # Create temporary image file
    test_image_path = './test_image.png'
    cv2.imwrite(test_image_path, TEST_IMAEG)

    # Test loading image with default parameters
    result_image = ui.load_image(test_image_path)
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == (100, 100, 3)  # Default channels is 3

    # Test loading image with specified channels (1 channel)
    result_image = ui.load_image(test_image_path, out_channels=1)
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == (100, 100, 1)

    # Clean up: Remove temporary image file
    Path(test_image_path).unlink()

    logger.debug("test_load_image passed!")


def create_dummy_dicom(filename: str) -> str:
    # Create a new DICOM dataset
    ds = pyd.Dataset()

    # Add patient information
    ds.PatientName = 'Test^Patient'
    ds.PatientID = '123456'
    ds.PatientAge = '012Y'  # 12 years
    ds.PatientSex = 'F'  # Female

    # Add study and series information
    ds.StudyInstanceUID = '1.2.3.4.5.6.7.8.9.0'
    ds.SeriesInstanceUID = '1.2.3.4.5.6.7.8.9.1'
    ds.SOPInstanceUID = '1.2.3.4.5.6.7.8.9.2'
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'

    # Add image-related information
    ds.Rows = 512
    ds.Columns = 768
    ds.StudyDate = '20240101'
    ds.ViewPosition = 'AP'

    # Set file meta information values
    meta = pyd.Dataset()
    meta.MediaStorageSOPClassUID = pyd.uid.generate_uid()
    meta.MediaStorageSOPInstanceUID = pyd.uid.generate_uid()
    meta.TransferSyntaxUID = pyd.uid.ImplicitVRLittleEndian

    # Create the FileDataset instance (instance of DICOM dataset)
    file_meta = pyd.dataset.FileMetaDataset(meta)
    ds.file_meta = file_meta

    # Set creation date and time
    dt = datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S.%f')

    # Set the file path
    file_path = os.path.join(tempfile.gettempdir(), filename)

    # Write the DICOM file
    pyd.dcmwrite(file_path, ds)

    return file_path


def test_utils_dicom():
    # Test loading DICOM file
    path_dummy_dicom = create_dummy_dicom('test_dicom.dcm')
    dicom_data = ur.load_dicom(path_dummy_dicom, force=True)
    assert ud.get_meta_from_dicom(dicom_data, 'age') == 12
    assert ud.get_meta_from_dicom(dicom_data, 'gender') == 'f'
    assert ud.get_meta_from_dicom(dicom_data, 'class_uid') == '1.2.840.10008.5.1.4.1.1.2'
    assert ud.get_meta_from_dicom(dicom_data, 'instance_uid') == '1.2.3.4.5.6.7.8.9.2'
    assert ud.get_meta_from_dicom(dicom_data, 'shape') == (512, 768)
    assert ud.get_meta_from_dicom(dicom_data, 'study_date') == datetime(2024, 1, 1)
    assert ud.get_meta_from_dicom(dicom_data, 'view_position') == 'ap'
