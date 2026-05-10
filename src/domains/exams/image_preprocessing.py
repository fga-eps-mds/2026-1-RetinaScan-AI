import cv2
import numpy as np

def apply_retina_contour(img: np.ndarray) -> np.ndarray:
    try:
        if img.dtype != np.uint8:
            img_calc = cv2.normalize(
                img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        else:
            img_calc = img

        gray = cv2.cvtColor(img_calc, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        thresh_limpa = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            thresh_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("Nenhum contorno válido encontrado após a limpeza.")

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 1000:
            raise ValueError("Área identificada é muito pequena.")

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, (255,), thickness=cv2.FILLED)

        masked_image = cv2.bitwise_and(img, img, mask=mask)

        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_masked_image = masked_image[y:y + h, x:x + w]

        return cropped_masked_image

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Erro no processamento da imagem: {str(e)}")