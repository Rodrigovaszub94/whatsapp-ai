"""
FastAPI app que procesa un zip de WhatsApp export (con multimedia),
extrae texto (chat txt + OCR de imágenes + PDFs), resume y genera
un mensaje humano listo para enviar al cliente.

Requisitos:
- Tesseract en el sistema (apt-get install -y tesseract-ocr tesseract-ocr-spa)
- Python packages: see requirements.txt
- (Opcional) HF_API_TOKEN: si se quiere usar la Hugging Face Inference API
"""

import asyncio
import os
import re
import zipfile
import tempfile
import shutil
import pathlib
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import mimetypes

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps
import pytesseract
import pdfplumber

# Transformers
from transformers import pipeline, Pipeline, AutoTokenizer
import torch
import requests
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configurables ---
MAX_ZIP_SIZE_MB = int(os.environ.get("MAX_ZIP_SIZE_MB", "200"))
OCR_WORKERS = int(os.environ.get("OCR_WORKERS", "4"))
CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "3500"))
SUMMARY_MAX_TOKENS = int(os.environ.get("SUMMARY_MAX_TOKENS", "220"))
FINAL_GEN_MAX_TOKENS = int(os.environ.get("FINAL_GEN_MAX_TOKENS", "400"))
MODEL_NAME = os.environ.get("LOCAL_MODEL", "google/flan-t5-base")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
PORT = int(os.environ.get("PORT", "8000"))

# Tipos de archivos soportados
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
SUPPORTED_DOCUMENT_EXTENSIONS = {'.pdf'}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_DOCUMENT_EXTENSIONS

# ----------------------

@dataclass
class ProcessingResult:
    """Resultado del procesamiento"""
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
            if self._generator is not None:  # Double-check
                return self._generator
                
            try:
                logger.info(f"Cargando modelo local {MODEL_NAME}...")
                device = 0 if torch.cuda.is_available() else -1
                self._generator = pipeline(
                    "text2text-generation", 
                    model=MODEL_NAME, 
                    tokenizer=MODEL_NAME,
                    device=device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                logger.info("Modelo cargado exitosamente")
                return self._generator
            except Exception as e:
                logger.error(f"Error cargando modelo local: {e}")
                self._generator = None
                return None

# Instancia global del gestor de modelos
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    # Startup
    logger.info("Iniciando aplicación...")
    
    # Pre-cargar el modelo de forma asíncrona
    asyncio.create_task(model_manager.get_generator())
    
    yield
    
    # Shutdown
    logger.info("Cerrando aplicación...")

app = FastAPI(
    title="WhatsApp Chat Processor",
    description="Procesa exports de WhatsApp con multimedia y genera resúmenes",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configura según tus necesidades
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileProcessor:
    """Clase para procesar diferentes tipos de archivos"""
    
    @staticmethod
    def safe_extract(zipfile_obj: zipfile.ZipFile, path: str) -> None:
        """Extracción segura de ZIP evitando path traversal"""
        for member in zipfile_obj.namelist():
            member_path = os.path.join(path, member)
            abs_base = os.path.abspath(path)
            abs_target = os.path.abspath(member_path)
            if not abs_target.startswith(abs_base + os.sep) and abs_target != abs_base:
                raise ValueError(f"Miembro de ZIP inseguro detectado: {member}")
        zipfile_obj.extractall(path)
    
    @staticmethod
    def find_chat_txt(root: str) -> Optional[str]:
        """Encuentra el archivo .txt principal del chat"""
        txt_files = list(pathlib.Path(root).rglob("*.txt"))
        if not txt_files:
            return None
        
        # Priorizar archivos que contengan 'WhatsApp' o 'Chat'
        priority_files = [
            p for p in txt_files 
            if any(keyword in p.name.lower() for keyword in ['whatsapp', 'chat'])
        ]
        
        if priority_files:
            return str(max(priority_files, key=lambda x: x.stat().st_size))
        
        # Si no, devolver el más grande
        return str(max(txt_files, key=lambda x: x.stat().st_size))
    
    @staticmethod
    def read_text_file(path: str) -> str:
        """Lee archivo de texto con múltiples encodings"""
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc, errors="replace") as f:
                    content = f.read()
                    if content.strip():  # Verificar que no esté vacío
                        return content
            except Exception:
                continue
        
        # Fallback con lectura binaria
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="replace")
    
    @staticmethod
    def collect_files_by_extension(root: str, extensions: set) -> List[str]:
        """Recolecta archivos por extensiones"""
        files = []
        for ext in extensions:
            files.extend(pathlib.Path(root).rglob(f"*{ext}"))
        return [str(f) for f in files if f.is_file()]
    
    @staticmethod
    def ocr_image(path: str, tesseract_langs: Optional[str] = None) -> str:
        """OCR de una imagen individual"""
        try:
            # Verificar que el archivo existe y es válido
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return f"[OCR {os.path.basename(path)} - Archivo vacío o inexistente]"
            
            with Image.open(path) as img:
                # Optimizaciones para OCR
                img = img.convert("RGB")
                img = ImageOps.autocontrast(img)  # Mejorar contraste
                
                # Redimensionar si es muy grande
                if img.width > 2000 or img.height > 2000:
                    img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
                
                # OCR con configuración optimizada
                config = '--oem 3 --psm 6'  # Configuración para mejor precisión
                
                if tesseract_langs:
                    try:
                        text = pytesseract.image_to_string(img, lang=tesseract_langs, config=config)
                    except Exception:
                        text = pytesseract.image_to_string(img, config=config)
                else:
                    text = pytesseract.image_to_string(img, config=config)
                
                cleaned_text = text.strip()
                if not cleaned_text:
                    return f"[OCR {os.path.basename(path)} - Sin texto detectado]"
                
                return f"[OCR {os.path.basename(path)}]\n{cleaned_text}"
                
        except Exception as e:
            logger.error(f"Error en OCR de {path}: {e}")
            return f"[OCR {os.path.basename(path)} ERROR: {str(e)}]"
    
    @staticmethod
    def ocr_images_batch(paths: List[str], langs: Optional[str] = None) -> List[str]:
        """OCR de múltiples imágenes con ThreadPool"""
        if not paths:
            return []
        
        results = []
        with ThreadPoolExecutor(max_workers=OCR_WORKERS) as executor:
            future_to_path = {
                executor.submit(FileProcessor.ocr_image, path, langs): path 
                for path in paths
            }
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result(timeout=30)  # Timeout por imagen
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error procesando {path}: {e}")
                    results.append(f"[OCR ERROR en {os.path.basename(path)}: {str(e)}]")
        
        return results
    
    @staticmethod
    def extract_text_from_pdf(path: str) -> str:
        """Extrae texto de PDF"""
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return f"[PDF {os.path.basename(path)} - Archivo vacío o inexistente]"
            
            texts = []
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages[:50]):  # Limitar a 50 páginas
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            texts.append(f"--- Página {i+1} ---\n{text.strip()}")
                    except Exception as e:
                        texts.append(f"--- Página {i+1} ERROR: {str(e)} ---")
            
            if not texts:
                return f"[PDF {os.path.basename(path)} - Sin texto extraído]"
            
            return f"[PDF {os.path.basename(path)}]\n" + "\n\n".join(texts)
            
        except Exception as e:
            logger.error(f"Error extrayendo PDF {path}: {e}")
            return f"[PDF {os.path.basename(path)} ERROR: {str(e)}]"

class TextProcessor:
    """Clase para procesamiento de texto y generación"""
    
    @staticmethod
    def clean_chat_text(raw: str) -> str:
        """Limpia el texto del chat de WhatsApp"""
        # Remover marcadores de media
        patterns = [
            r"<Media omitted>|image omitted|\[Media omitted\]",
            r"<audio omitted>|\[audio omitted\]",
            r"<video omitted>|\[video omitted\]",
            r"<document omitted>|\[document omitted\]"
        ]
        
        for pattern in patterns:
            raw = re.sub(pattern, "", raw, flags=re.IGNORECASE)
        
        # Limpiar saltos de línea excesivos
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        # Limpiar espacios en blanco excesivos
        raw = re.sub(r" {2,}", " ", raw)
        
        return raw.strip()
    
    @staticmethod
    def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
        """Divide texto en chunks respetando límites de oraciones"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            
            # Buscar punto de división óptimo
            split_candidates = [
                text.rfind("\n\n", start, end),  # Párrafos
                text.rfind(".\n", start, end),   # Final de oración
                text.rfind(". ", start, end),    # Final de oración con espacio
                text.rfind("\n", start, end),    # Salto de línea
                text.rfind(", ", start, end),    # Coma
            ]
            
            split_pos = next((pos for pos in split_candidates if pos > start), end)
            
            chunk = text[start:split_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            start = split_pos + 1 if split_pos < end else split_pos
        
        return [chunk for chunk in chunks if chunk.strip()]

class LLMProcessor:
    """Clase para interacción con modelos de lenguaje"""
    
    @staticmethod
    async def call_model(prompt: str, max_new_tokens: int = 200) -> str:
        """Llama al modelo local o API remota"""
        # Intentar modelo local primero
        generator = await model_manager.get_generator()
        if generator:
            try:
                # Ejecutar en thread pool para no bloquear
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: generator(
                        prompt, 
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=0.3,
                        num_beams=1
                    )
                )
                
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", str(result[0]))
                return str(result)
                
            except Exception as e:
                logger.error(f"Error en generación local: {e}")
        
        # Fallback a API de Hugging Face
        if HF_API_TOKEN:
            return await LLMProcessor._call_hf_api(prompt, max_new_tokens)
        
        raise RuntimeError("No hay modelo local disponible ni HF_API_TOKEN configurado")
    
    @staticmethod
    async def _call_hf_api(prompt: str, max_new_tokens: int) -> str:
        """Llama a la API de Hugging Face"""
        api_url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": 0.3
            }
        }
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(api_url, headers=headers, json=payload, timeout=120)
            )
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                return data[0]["generated_text"]
            elif isinstance(data, dict) and "error" in data:
                raise Exception(f"Error de API: {data.get('error', 'Unknown')}")
            
            return str(data)
            
        except Exception as e:
            raise RuntimeError(f"Error en API de Hugging Face: {str(e)}")
    
    @staticmethod
    async def summarize_chunks(chunks: List[str]) -> List[str]:
        """Resume múltiples chunks de texto"""
        summaries = []
        
        for i, chunk in enumerate(chunks):
            prompt = (
                "Resumir en español este fragmento de chat de WhatsApp y OCR. "
                "Extraer información clave como nombres, fechas, direcciones, servicios, "
                "precios y detalles importantes. Máximo 8 líneas.\n\n"
                f"Texto:\n{chunk}\n\n"
                "Resumen:"
            )
            
            try:
                summary = await LLMProcessor.call_model(prompt, SUMMARY_MAX_TOKENS)
                summaries.append(summary.strip())
                logger.info(f"Chunk {i+1}/{len(chunks)} resumido")
            except Exception as e:
                logger.error(f"Error resumiendo chunk {i+1}: {e}")
                summaries.append(f"Error resumiendo fragmento {i+1}: {str(e)}")
        
        return summaries
    
    @staticmethod
    async def generate_final_message(aggregated_summary: str) -> str:
        """Genera el mensaje final para el cliente"""
        prompt = (
            "Convierte esta información de un chat de WhatsApp en un mensaje claro "
            "y amigable para enviar al cliente. Usa emojis apropiados y formato legible. "
            "Incluye: nombre del contacto, fecha/hora, ubicación, servicios solicitados, "
            "precios y detalles de pago. Si falta información, indica 'Por confirmar'.\n"
            "NO agregues explicaciones, solo el mensaje final.\n\n"
            f"Información del chat:\n{aggregated_summary}\n\n"
            "Mensaje para el cliente:"
        )
        
        result = await LLMProcessor.call_model(prompt, FINAL_GEN_MAX_TOKENS)
        return result.strip()

async def cleanup_temp_dir(temp_dir: str) -> None:
    """Limpia directorio temporal de forma asíncrona"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: shutil.rmtree(temp_dir, ignore_errors=True))

@app.get("/health")
async def health_check():
    """Endpoint de salud"""
    return {"status": "healthy", "model_loaded": await model_manager.get_generator() is not None}

@app.post("/process", response_model=dict)
async def process_zip(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Archivo ZIP del export de WhatsApp"),
    tesseract_langs: Optional[str] = Query(None, description="Idiomas para Tesseract (ej: 'spa+eng')")
):
    """
    Procesa un ZIP de export de WhatsApp con multimedia.
    
    - **file**: Archivo ZIP del export de WhatsApp
    - **tesseract_langs**: Idiomas para OCR (opcional, por defecto usa configuración de Tesseract)
    
    Retorna resumen procesado y mensaje final para cliente.
    """
    
    # Validaciones iniciales
    if not file.filename or not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un ZIP")
    
    # Leer contenido del archivo
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    
    if size_mb > MAX_ZIP_SIZE_MB:
        raise HTTPException(
            status_code=413, 
            detail=f"ZIP muy grande: {size_mb:.1f} MB (límite: {MAX_ZIP_SIZE_MB} MB)"
        )
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp(prefix="whatsapp_proc_")
    zip_path = os.path.join(temp_dir, "upload.zip")
    
    # Programar limpieza del directorio temporal
    background_tasks.add_task(cleanup_temp_dir, temp_dir)
    
    try:
        # Guardar y extraer ZIP
        with open(zip_path, "wb") as f:
            f.write(contents)
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            FileProcessor.safe_extract(zf, temp_dir)
        
        logger.info(f"ZIP extraído exitosamente: {size_mb:.1f} MB")
        
        # Procesar archivos en paralelo
        async def process_files():
            # Buscar y leer archivo de chat
            txt_path = FileProcessor.find_chat_txt(temp_dir)
            chat_text = ""
            if txt_path:
                chat_text = FileProcessor.read_text_file(txt_path)
                chat_text = TextProcessor.clean_chat_text(chat_text)
            
            # Recolectar archivos multimedia
            image_files = FileProcessor.collect_files_by_extension(temp_dir, SUPPORTED_IMAGE_EXTENSIONS)
            pdf_files = FileProcessor.collect_files_by_extension(temp_dir, SUPPORTED_DOCUMENT_EXTENSIONS)
            
            # Procesar OCR y PDFs en paralelo
            loop = asyncio.get_event_loop()
            
            ocr_task = loop.run_in_executor(
                None, 
                FileProcessor.ocr_images_batch, 
                image_files, 
                tesseract_langs
            )
            
            pdf_tasks = [
                loop.run_in_executor(None, FileProcessor.extract_text_from_pdf, pdf_file)
                for pdf_file in pdf_files
            ]
            
            # Esperar resultados
            ocr_results = await ocr_task
            pdf_results = await asyncio.gather(*pdf_tasks) if pdf_tasks else []
            
            return chat_text, ocr_results, pdf_results, len(image_files), len(pdf_files)
        
        chat_text, ocr_results, pdf_results, num_images, num_pdfs = await process_files()
        
        # Consolidar todo el texto
        all_text_parts = [chat_text] + ocr_results + pdf_results
        all_text = "\n\n".join(part for part in all_text_parts if part.strip())
        
        if not all_text.strip():
            raise HTTPException(status_code=400, detail="No se encontró texto procesable en el ZIP")
        
        # Procesar con LLM
        chunks = TextProcessor.chunk_text(all_text)
        logger.info(f"Texto dividido en {len(chunks)} chunks")
        
        # Resumir chunks
        summaries = await LLMProcessor.summarize_chunks(chunks)
        aggregated_summary = "\n\n".join(summaries)
        
        # Generar mensaje final
        final_message = await LLMProcessor.generate_final_message(aggregated_summary)
        
        # Estadísticas
        stats = {
            "zip_size_mb": round(size_mb, 2),
            "chat_text_chars": len(chat_text),
            "images_processed": num_images,
            "pdfs_processed": num_pdfs,
            "total_chunks": len(chunks),
            "final_message_chars": len(final_message)
        }
        
        logger.info(f"Procesamiento completado: {stats}")
        
        return {
            "success": True,
            "summary": aggregated_summary,
            "final_message": final_message,
            "statistics": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando archivo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=PORT, 
        reload=False,
        workers=1  # Un solo worker para evitar múltiples cargas del modelo
    )
