from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

app = FastAPI()

# Inicializar vectorstore desde la base persistida
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

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