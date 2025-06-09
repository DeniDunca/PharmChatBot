from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .asi_mini import chain
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Message(BaseModel):
    content: str
@app.post("/send-message")
async def send_message(message: Message):
    # Use the chain to get a response based on the question
    ai_response = chain.invoke({"question": message.content})
    return {"message": ai_response}

@app.get("/")
async def root():
    return {"message": "Welcome to the chat API!"}