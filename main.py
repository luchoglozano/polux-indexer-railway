from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import os

app = FastAPI()

# Healthcheck básico
@app.get("/")
def read_root():
    return {"status": "ok"}

# Configuración de vectores
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
)

# Configurar el modelo de QA
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    retriever=vectordb.as_retriever()
)

# Esquema de la pregunta
class Question(BaseModel):
    query: str

# Endpoint principal
@app.post("/ask")
def ask(question: Question):
    response = qa.run(question.query)
    return {"response": response}
