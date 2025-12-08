from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import agent
from agents import Runner

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(body: Query):
    question = body.question
    result = await Runner.run(agent, input=question)
    return {"answer": result.final_output}

@app.get("/")
def home():
    return {"status": "RAG chatbot API running"}
