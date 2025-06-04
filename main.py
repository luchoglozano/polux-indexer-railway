from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Inicializar la app
app = FastAPI()

# Healthcheck para Railway
@app.get("/")
def healthcheck():
    return {"status": "Polux is running"}

# Obtener la clave de OpenAI desde el entorno
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is missing in environment variables.")

# Cargar base vectorial
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key)
)

# Crear cadena de QA con GPT-4
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4", temperature=0),
    retriever=vectordb.as_retriever()
)

# Modelo para recibir preguntas
class Question(BaseModel):
    query: str

# Endpoint de interacción
@app.post("/ask")
def ask(question: Question):
    response = qa.run(question.query)
    return {"response": response}

