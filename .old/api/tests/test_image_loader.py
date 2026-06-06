from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from api.pre_processing.image_loader import load_and_clean_image


def test_load_and_clean_image_padrao():
    """
    Testa o carregamento e decodificação de um arquivo de imagem padrão da web (JPEG),
    garantindo que ele é convertido para uma matriz do OpenCV (numpy array).
    """
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    image_bytes = buffer.tobytes()

    resultado = load_and_clean_image(image_bytes, "exame.jpg")

    assert isinstance(resultado, np.ndarray)
    assert resultado.shape == (50, 50, 3)


@patch("api.pre_processing.image_loader.pydicom.dcmread")
def test_load_and_clean_image_dicom(mock_dcmread):
    """
    Testa se o fluxo para arquivos médicos (.dcm) utiliza corretamente a biblioteca
    pydicom para extrair os pixels crus, descartando os metadados do paciente.
    """
    mock_dicom = MagicMock()
    mock_dicom.pixel_array = np.zeros((50, 50), dtype=np.uint8)
    mock_dcmread.return_value = mock_dicom

    fake_bytes = b"fake_dicom_data"

    resultado = load_and_clean_image(fake_bytes, "exame_paciente.dcm")

    assert isinstance(resultado, np.ndarray)
    assert resultado.shape == (50, 50, 3)
