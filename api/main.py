import os
import sys

# from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from api.pre_processing.image_loader import convert_to_clean_png, load_and_clean_image
from api.pre_processing.retina_contour import apply_retina_contour

# from .model import RetinaScanModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

MAX_FILES = 2
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "output_dir"
    / "retfound_mae_RFMiD_binary_finetune"
    / "checkpoint-best.pth"
)

predictor = None


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global predictor
#     predictor = RetinaScanModel(
#         checkpoint_path=CHECKPOINT_PATH,
#         model_name="RETFound_mae",
#         input_size=224,
#         num_classes=2,
#     )
#     yield
#     predictor = None


app = FastAPI(
    title="RetinaScan-AI API",
    version="0.0.1",
    description="API para diagnostico de imagens de retina com IA",
    # lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "max_files": MAX_FILES,
    }


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não carregado. Tente novamente mais tarde.",
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Nenhum arquivo enviado. Por favor, envie pelo menos um arquivo de imagem.",
        )

    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Limite de arquivos excedido. O máximo permitido é {MAX_FILES}.",
        )

    results = []

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de arquivo não permitido: {file.content_type}. Tipos permitidos: {', '.join(ALLOWED_TYPES)}.",
            )

        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"O arquivo {file.filename} está vazio. Por favor, envie um arquivo de imagem válido.",
            )

        pred = predictor.predict_bytes(content)

        results.append(
            {
                "filename": file.filename,
                "content_type": file.content_type,
                **pred,
            }
        )

    return {
        "total_images": len(results),
        "results": results,
    }


@app.post("/api/v1/analyze")
async def analyze_retina(file: UploadFile = File(...)):
    """
    Endpoint para recepção, limpeza e pré-processamento inicial do exame. Garante a anonimização antes
    de qualquer análise posterior.
    """
    valid_extensions = (".png", ".jpg", ".jpeg", ".dcm", ".dicom")

    filename = (file.filename or "").lower()

    if not filename.endswith(valid_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado. Envie imagens em {', '.join(valid_extensions)}.",
        )

    try:
        image_bytes = await file.read()

        clean_img_array = load_and_clean_image(image_bytes, file.filename)

        cropped_masked_image = apply_retina_contour(clean_img_array)

        final_png_bytes = convert_to_clean_png(cropped_masked_image)  # noqa: F841

        # Inserir os próximos passos da IA aqui

        return {
            "status": "Pré-processamento concluído",
            "message": "Metadados sensíveis removidos e imagem isolada com sucesso.",
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=500, detail="Erro interno inesperado no processamento."
        )
