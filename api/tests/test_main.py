import cv2
import numpy as np
from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_analyze_retina_recusa_extensao_invalida():
    """
    Testa se a rota da API barra o processamento de arquivos não suportados (ex: PDF),
    retornando um status HTTP 400 (Bad Request).
    """
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("relatorio.pdf", b"pdf fake", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Formato não suportado" in response.json()["detail"]


def test_analyze_retina_sucesso_fluxo_completo():
    """
    Testa o fluxo integrado de sucesso da API, simulando o upload de uma imagem
    válida e verificando se todas as etapas de limpeza e corte passam sem erros.
    """
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (255, 255, 255), -1)

    _, buffer = cv2.imencode(".jpg", img)

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("fundo_de_olho_valido.jpg", buffer.tobytes(), "image/jpeg")},
    )

    # Verifica se a API passou por todo o processamento e retornou o status de sucesso
    assert response.status_code == 200
    assert response.json()["status"] == "Pré-processamento concluído"
    assert "Metadados sensíveis removidos" in response.json()["message"]
