from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import os

app = FastAPI()

# Hugging Face Inference API settings
HF_API_URL = "https://api-inference.huggingface.co/models/philschmid/bart-large-cnn-samsum"
HF_API_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

# Input models
class Turn(BaseModel):
    speaker: str
    text: str

class Transcript(BaseModel):
    transcript: List[Turn]

# Summarization endpoint with full debugging output
@app.post("/summarize")
async def summarize_call(data: Transcript):
    dialogue = "\n".join([f"{turn.speaker}: {turn.text}" for turn in data.transcript])
    payload = {"inputs": dialogue}

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    try:
        result = response.json()
    except Exception:
        return {
            "error": "Failed to parse Hugging Face JSON response.",
            "raw_response": response.text
        }

    # Return the full response from Hugging Face for debugging
    return {"huggingface_response": result}
