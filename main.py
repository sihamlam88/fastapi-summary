from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import os

app = FastAPI()

# Hugging Face model endpoint
HF_API_URL = "https://api-inference.huggingface.co/models/philschmid/bart-large-cnn-samsum"
HF_API_TOKEN = os.getenv("HF_TOKEN")  # You’ll set this on Render

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

class Turn(BaseModel):
    speaker: str
    text: str

class Transcript(BaseModel):
    transcript: List[Turn]

@app.post("/summarize")
async def summarize_call(data: Transcript):
    dialogue = "\n".join([f"{turn.speaker}: {turn.text}" for turn in data.transcript])
    payload = {"inputs": dialogue}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    result = response.json()
    
    # Handle model warmup
    if isinstance(result, dict) and result.get("error"):
        return {"summary": "Model is warming up or unavailable. Please try again shortly."}

    return {"summary": result[0]["summary_text"]}
