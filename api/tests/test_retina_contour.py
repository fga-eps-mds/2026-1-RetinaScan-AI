import cv2
import numpy as np
import pytest
from api.pre_processing.retina_contour import apply_retina_contour


def test_apply_retina_contour_sucesso():
    """
    Testa se a função recorta corretamente uma imagem válida (que contém um formato circular identificável),
    removendo o excesso de fundo preto e mantendo os canais de cor originais.
    """
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)

    cropped_img = apply_retina_contour(img)

    assert cropped_img is not None

    assert cropped_img.shape[0] < 200
    assert cropped_img.shape[1] < 200
    assert cropped_img.shape[2] == 3


def test_apply_retina_contour_imagem_escura():
    """
    Testa se a função levanta o erro apropriado (ValueError) quando recebe uma imagem
    completamente escura ou sem contornos suficientes para identificar o globo ocular.
    """
    img_preta = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError) as excinfo:
        apply_retina_contour(img_preta)

    assert "Nenhum contorno válido" in str(excinfo.value)


def test_apply_retina_contour_16bits():
    """
    Garante que imagens médicas de 16 bits sejam processadas sem erro e sem perder a
    profundidade de pixels no resultado final.
    """
    img_16 = np.zeros((200, 200, 3), dtype=np.uint16)

    cv2.circle(img_16, (100, 100), 50, (65535, 65535, 65535), -1)

    cropped_img = apply_retina_contour(img_16)

    assert cropped_img is not None
    assert cropped_img.dtype == np.uint16
    assert cropped_img.shape[0] < 200
