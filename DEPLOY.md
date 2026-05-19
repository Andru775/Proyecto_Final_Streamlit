# Despliegue de la aplicación Streamlit

Este documento describe pasos sencillos para desplegar la aplicación en **Streamlit Cloud** y de forma alternativa en **Heroku**, además de instrucciones para ejecutar localmente.

## Requisitos previos
- Cuenta de GitHub (para Streamlit Cloud) o cuenta de Heroku (para Heroku).
- `requirements.txt` presente (ya incluido).
- `xgb_model.json` en la raíz del repositorio (ya incluido).

## Despliegue en Streamlit Cloud (recomendado)

1. Subir este repositorio a GitHub.
2. Ir a https://streamlit.io/cloud y entrar con GitHub.
3. Crear una nueva app y seleccionar el repositorio y la rama correspondiente.
4. Como `Entry point` poner `app.py`.
5. Streamlit Cloud detectará `requirements.txt` e instalará dependencias automáticamente.
6. Desplegar y abrir la URL proporcionada.

## Despliegue en Heroku (alternativa)

1. Instalar Heroku CLI y loguearse: `heroku login`.
2. Crear app Heroku: `heroku create nombre-de-tu-app`.
3. Añadir buildpacks que permiten usar Python (Heroku lo detecta por `runtime.txt`).
4. Empujar el repositorio a Heroku (git push heroku main).
5. Heroku usará el `Procfile` para ejecutar el comando: `web: streamlit run app.py --server.port $PORT --server.enableCORS false`.

Nota: Heroku puede requerir configuraciones adicionales para dependencias nativas (XGBoost). Si encuentra errores, pruebe con un contenedor Docker.

## Ejecutar localmente (prueba rápida)

1. Crear y activar un entorno virtual (recomendado):

```bash
python -m venv .venv
.
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecutar la app:

```bash
streamlit run app.py
```

4. Abrir `http://localhost:8501` en el navegador.

## Notas
- Si la app falla al cargar `xgb_model.json`, confirme que el archivo existe en la raíz y que es compatible con la versión de `xgboost` instalada.
- El repositorio ahora está configurado para compatibilidad con el entorno actual de Streamlit Cloud (Python 3.14, `streamlit==1.57.0`, `pandas==3.0.3`, `Pillow==12.2.0`).
- Para desplegar en Streamlit Cloud, asegúrese de que el repositorio sea público o que Streamlit Cloud tenga acceso al repositorio privado.

## Fallback automático
La app incluye un predictor de respaldo (`fallback_model.py`) que se usa automáticamente si `xgboost` no está disponible o si el modelo `xgb_model.json` no puede cargarse en el entorno de despliegue. Esto garantiza que la interfaz Streamlit funcionará y proporcionará predicciones aproximadas aunque XGBoost no se instale correctamente en el servidor.

Si deseas forzar el uso del modelo XGBoost en Streamlit Cloud, intenta instalar `xgboost` en el entorno de despliegue o usa un contenedor Docker que incluya las ruedas necesarias.
