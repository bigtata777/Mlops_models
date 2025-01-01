# Usa una imagen base ligera de Python
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala las herramientas necesarias para construir las dependencias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && apt-get clean

# Copia los archivos de tu proyecto al contenedor
COPY app.py /app/app.py
COPY adaboost_model.pkl /app/adaboost_model.pkl
COPY requirements.txt /app/requirements.txt

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto en el que la aplicación escuchará
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

