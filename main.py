"""
FastAPI app que procesa un zip de WhatsApp export (con multimedia),
extrae texto (chat txt + OCR de imágenes + PDFs), resume y genera
un mensaje humano listo para enviar al cliente.

Optimizado para Render Free Tier (512 MB RAM):
- Usa modelo más pequeño (flan-t5-small por defecto).
- Carga diferida del modelo (lazy loading).
"""

import asyncio
import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from transformers import pipeline, Pipeline
import torch

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configurables ---
MODEL_NAME = os.environ.get("LOCAL_MODEL", "google/flan-t5-small")
PORT = int(os.environ.get("PORT", "8000"))

# ----------------------
@dataclass
class ProcessingResult:
    chat_text: str
    ocr_results: List[str]
    pdf_results: List[str]
    summary: str
    final_message: str
    stats: Dict[str, int]

class ModelManager:
    """Gestor singleton para el modelo de ML"""

    def __init__(self):
        self._generator: Optional[Pipeline] = None
        self._lock = asyncio.Lock()

    async def get_generator(self) -> Optional[Pipeline]:
        if self._generator is not None:
            return self._generator

        async with self._lock:
            if self._generator is not None:
                return self._generator

            try:
                logger.info(f"Cargando modelo local {MODEL_NAME}...")
                device = 0 if torch.cuda.is_available() else -1
                self._generator = pipeline(
                    "text2text-generation",
                    model=MODEL_NAME,
                    tokenizer=MODEL_NAME,
                    device=device,
                    torch_dtype=torch.float32,
                )
                logger.info("Modelo cargado exitosamente")
                return self._generator
            except Exception as e:
                logger.error(f"Error cargando modelo local: {e}")
                self._generator = None
                return None

# Instancia global
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando aplicación (lazy loading activado)...")
    yield
    logger.info("Cerrando aplicación...")

app = FastAPI(
    title="WhatsApp Chat Processor",
    description="Procesa exports de WhatsApp con multimedia y genera resúmenes",
    version="2.1.0",
    lifespan=lifespan,
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- NUEVO: Página principal ----------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>WhatsApp AI Processor</title>
        </head>
        <body style="font-family: sans-serif; margin: 40px;">
            <h2>Subir export de WhatsApp (ZIP)</h2>
            <form action="/process" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".zip">
                <button type="submit">Procesar</button>
            </form>
            <p>También puedes usar la <a href="/docs">documentación interactiva (/docs)</a>.</p>
        </body>
    </html>
    """

# ---------------------- ENDPOINTS EXISTENTES ----------------------
@app.get("/health")
async def health_check():
    """Endpoint de salud"""
    gen = await model_manager.get_generator()
    return {"status": "healthy", "model_loaded": gen is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        workers=1,  # evitar múltiples cargas de modelo
    )
