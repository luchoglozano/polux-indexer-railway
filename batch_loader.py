import os
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Asegúrate de tener la clave API configurada como variable de entorno
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Ruta a la carpeta de documentos PDF
docs_path = Path("docs")
pdf_files = list(docs_path.glob("*.pdf"))

if not pdf_files:
    print("❌ No se encontraron archivos PDF en la carpeta 'docs/'.")
    exit()

print(f"🔍 Encontrados {len(pdf_files)} archivos PDF. Iniciando carga...")

all_docs = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

for pdf_file in pdf_files:
    print(f"📄 Procesando: {pdf_file.name}")
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()
    split_docs = text_splitter.split_documents(documents)
    all_docs.extend(split_docs)

# Crear y guardar la base vectorial
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=all_docs,
    embedding=embedding,
    persist_directory="./chroma_db"
)
vectordb.persist()

print("✅ Base vectorial generada y persistida correctamente con todos los documentos PDF.")