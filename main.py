from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

app = FastAPI()

# Endpoint para el healthcheck
@app.get("/")
def read_root():
    return {"status": "ok"}

# Inicializar vectorstore desde la base persistida
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
)

# Crear el QA chain con GPT-4
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    retriever=vectordb.as_retriever()
)

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask(question: Question):
    response = qa.run(question.query)
    return {"response": response}
