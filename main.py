from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

app = FastAPI()

# Endpoint base para healthcheck
@app.get("/")
def read_root():
    return {"status": "ok"}

# Cargar vectorstore desde disco
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

# Crear cadena de QA
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
