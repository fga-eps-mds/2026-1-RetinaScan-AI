from contextlib import asynccontextmanager
from typing import List
import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from fastapi import FastAPI, File, HTTPException, UploadFile
from .model import RetinaScanModel

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

MAX_FILES = 2
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CHECKPOINT_PATH = PROJECT_ROOT / "output_dir" / "retfound_dinov2_ODIR_binary_finetune" / "checkpoint-best.pth"

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = RetinaScanModel(
        checkpoint_path=CHECKPOINT_PATH,
        model_name="RETFound_dinov2",
        input_size=224,
        num_classes=2,
        threshold=0.5,
    )
    yield
    predictor = None

app = FastAPI(
    title="RetinaScan-AI API",
    version="0.0.1",
    description="API para diagnostico de imagens de retina com IA",
    lifespan=lifespan,
)

@app.get('/health')
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "max_files": MAX_FILES,
    }

@app.post('/predict')
async def predict(files: List[UploadFile] = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado. Tente novamente mais tarde.")
    
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado. Por favor, envie pelo menos um arquivo de imagem.")
    
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Limite de arquivos excedido. O máximo permitido é {MAX_FILES}.")
    
    results = []

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Tipo de arquivo não permitido: {file.content_type}. Tipos permitidos: {', '.join(ALLOWED_TYPES)}.")
    
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"O arquivo {file.filename} está vazio. Por favor, envie um arquivo de imagem válido.")
        
        pred = predictor.predict_bytes(content)

        results.append({
            "filename": file.filename,
            "content_type": file.content_type,
            **pred,
        })

    return {
        "total_images": len(results),
        "results": results,
    }