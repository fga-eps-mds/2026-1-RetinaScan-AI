from fastapi import APIRouter, Request

router = APIRouter()

@router.get('/health', summary="Verificar a saúde da API", description="Endpoint para verificar se a API está funcionando corretamente")
async def health(request: Request):
    return {
        "status": "ok",
        "message": "API está funcionando corretamente"
    }