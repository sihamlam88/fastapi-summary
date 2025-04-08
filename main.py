from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import pipeline

app = FastAPI()

summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

class Turn(BaseModel):
    speaker: str
    text: str

class Transcript(BaseModel):
    transcript: List[Turn]

@app.post("/summarize")
async def summarize_call(data: Transcript):
    dialogue = "\n".join([f"{turn.speaker}: {turn.text}" for turn in data.transcript])
    result = summarizer(dialogue, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
    return {"summary": result}
