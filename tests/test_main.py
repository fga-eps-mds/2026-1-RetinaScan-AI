from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_upload_bloqueia_arquivo_que_nao_e_imagem():
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("curriculo.pdf", b"conteudo falso do pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "O arquivo enviado não é uma imagem válida."


def test_upload_sucesso_sem_mock():
    import cv2
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)
    _, buffer = cv2.imencode(".jpg", img)

    response = client.post(
        "/api/v1/analyze",
        files={"file": ("exame_valido.jpg", buffer.tobytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "Pré-processamento concluído com sucesso."
