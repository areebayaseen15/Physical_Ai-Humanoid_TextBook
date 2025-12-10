# FastAPI application entry point
# This file will be populated with a basic FastAPI app setup.

from fastapi import FastAPI

app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook API",
    description="Backend API for managing textbook content, RAG chatbot, personalization, and user authentication.",
    version="0.0.1",
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Physical AI & Humanoid Robotics Textbook API!"}

