import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Ruta base a documentos
docs_path = "docs"
persist_directory = "chroma_db"

def load_and_split_documents():
    all_docs = []
    for file in os.listdir(docs_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(docs_path, file)
            loader = PyPDFLoader(full_path)
            pages = loader.load()
            all_docs.extend(pages)
    return all_docs

def main():
    print("🧠 Cargando documentos PDF...")
    documents = load_and_split_documents()

    print(f"📄 Documentos encontrados: {len(documents)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    print(f"✂️ Fragmentos generados: {len(texts)}")
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

    vectordb.persist()
    print("✅ Base vectorial guardada en:", persist_directory)

if __name__ == "__main__":
    main()
