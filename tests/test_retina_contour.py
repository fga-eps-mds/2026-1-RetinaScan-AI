import cv2
import numpy as np
import pytest
from util.pre_processing.retina_contour import retina_contour


def test_retina_contour_sucesso():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)

    _, buffer = cv2.imencode(".jpg", img)
    fake_image_bytes = buffer.tobytes()

    masked_img = retina_contour(fake_image_bytes)

    assert masked_img is not None
    assert masked_img.shape == (100, 100, 3)


def test_retina_contour_falha_com_lixo():
    bytes_invalidos = b"isso_nao_e_uma_imagem_de_verdade"

    with pytest.raises(RuntimeError) as excinfo:
        retina_contour(bytes_invalidos)

    assert "Erro no processamento" in str(excinfo.value)
