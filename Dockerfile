# Imagen base
FROM python:3.12-slim

# Crear directorio de trabajo
WORKDIR /app

# Copiar todos los archivos al contenedor
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto usado por Streamlit
EXPOSE 8501

# Comando para ejecutar la app
CMD ["streamlit", "run", "emotionapp.py"]
