# Usamos Python 3.12-slim para compatibilidad con Torch CPU
FROM python:3.12-slim

# Evitamos preguntas interactivas en apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8000

# Instalar dependencias del sistema: tesseract y poppler-utils (PDF)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
        build-essential \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Crear carpeta de la app
WORKDIR /app

# Copiar archivo de dependencias
COPY requisitos.txt /app/requirements.txt

# Actualizar pip
RUN pip install --upgrade pip

# Instalar Torch CPU primero para evitar problemas de memoria
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Instalar el resto de dependencias de Python
RUN pip install -r requirements.txt --no-deps

# Copiar todo el proyecto
COPY . /app

# Exponer el puerto para FastAPI
EXPOSE 8000

# Comando por defecto para ejecutar la app con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
