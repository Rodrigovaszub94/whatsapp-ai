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

# Copiar archivo de dependencias
COPY requisitos.txt /app/requirements.txt

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copiar todo el proyecto
COPY . /app

# Comando por defecto para ejecutar la app
CMD ["python", "main.py"]
