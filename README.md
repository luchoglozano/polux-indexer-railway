# POLUX Indexer (Railway Version)

Este proyecto genera una base vectorial desde archivos PDF usando LangChain, Chroma y OpenAI.

## Uso

1. Coloca tu archivo PDF en la carpeta `docs/` y edita `load_documents.py` para apuntar a él.
2. Crea un archivo `.env` basado en `.env.example` con tu clave de OpenAI.
3. Instala dependencias:
   ```
   pip install -r requirements.txt
   ```
4. Ejecuta el script:
   ```
   python load_documents.py
   ```

Esto generará una carpeta `chroma_db/` con la base vectorial persistente.
# trigger deploy
