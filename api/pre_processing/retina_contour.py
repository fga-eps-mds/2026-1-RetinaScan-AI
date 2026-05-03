import cv2
import numpy as np


def retina_contour(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(
                "Falha ao decodificar a imagem. O arquivo pode ser inválido ou corrompido."
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError(
                "Nenhum contorno encontrado. A imagem pode estar completamente escura."
            )

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 1000:
            raise ValueError(
                "A área identificada é muito pequena para ser uma imagem de exame válida."
            )

        mask = np.zeros_like(gray)

        cv2.drawContours(mask, [largest_contour], -1, (255,), thickness=cv2.FILLED)

        masked_image = cv2.bitwise_and(img, img, mask=mask)

        return masked_image
    except Exception as e:
        raise RuntimeError(f"Erro no processamento da imagem: {str(e)}")
