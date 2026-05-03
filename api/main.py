import os
import sys
from pathlib import Path
from typing import List

from pre_processing.retina_contour import retina_contour

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
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="O arquivo enviado não é uma imagem válida."
        )

    try:
        image_bytes = await file.read()
        masked_image = retina_contour(image_bytes)

        # inserir a seguir os proximos passos da análise das imagens pela IA.

        return {"status": "Pré-processamento concluído com sucesso."}

    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro interno no processamento: {str(e)}"
        )
