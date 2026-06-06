from io import BytesIO

import cv2
import numpy as np
import pydicom

def load_and_clean_image(image_bytes: bytes, filename: str) -> np.ndarray:
    is_dicom = filename.lower().endswith((".dcm", ".dicom"))
    if is_dicom:
        return _decode_dicom_to_numpy(image_bytes)
    return _decode_standard_image_to_numpy(image_bytes)

def _decode_dicom_to_numpy(image_bytes: bytes) -> np.ndarray:
    dicom_file = pydicom.dcmread(BytesIO(image_bytes))
    pixel_array = dicom_file.pixel_array

    if pixel_array.dtype != np.uint8:
        img_out = pixel_array.astype(np.uint16)
    else:
        img_out = pixel_array.copy()

    if len(img_out.shape) == 3:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
    elif len(img_out.shape) == 2:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

    return img_out

def _decode_standard_image_to_numpy(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

    if img is None:
        raise ValueError(
            "Falha ao decodificar a imagem. Formato corrompido ou não suportado."
        )

    return img

def convert_to_clean_png(img_array: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", img_array)

    if not success:
        raise RuntimeError("Falha ao converter a imagem limpa para PNG.")

    return buffer.tobytes()