# Usamos una versión ligera de Python 3.13
FROM python:3.13-slim

# Evitamos preguntas interactivas en apt
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema (tesseract)
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev && \
    rm -rf /var/lib/apt/lists/*

# Crear carpeta de la app
WORKDIR /app

# Copiar archivos de dependencias
COPY pyproject.toml poetry.lock* /app/

# Instalar poetry y dependencias Python
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install --no-root

# Copiar todo el proyecto
COPY . /app

# Comando por defecto para ejecutar la app
CMD ["python", "main.py"]
