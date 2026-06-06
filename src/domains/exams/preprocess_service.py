from .image_loader import load_and_clean_image, convert_to_clean_png
from .image_preprocessing import apply_retina_contour

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".dcm", ".dicom")


def preprocess_retina_image(image_bytes: bytes, filename: str) -> bytes:
    filename_lower = (filename or "").lower()

    if not filename_lower.endswith(VALID_EXTENSIONS):
        raise ValueError(
            f"Formato não suportado. Envie imagens em {', '.join(VALID_EXTENSIONS)}."
        )

    clean_img_array = load_and_clean_image(image_bytes, filename)
    cropped_masked_image = apply_retina_contour(clean_img_array)
    return convert_to_clean_png(cropped_masked_image)