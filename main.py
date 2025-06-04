from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI()

# Carga del vectorstore
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0),
    retriever=vectordb.as_retriever()
)

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask(question: Question):
    result = qa.run(question.query)
    return {"response": result}
