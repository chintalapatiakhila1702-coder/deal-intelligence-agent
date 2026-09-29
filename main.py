import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from hindsight_client import Hindsight
import requests

load_dotenv()

app = FastAPI()

HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

hindsight_client = Hindsight(
    base_url="https://api.hindsight.vectorize.io",
    api_key=HINDSIGHT_API_KEY
)

BANK_ID = "deal-intelligence-bank"

class CallInput(BaseModel):
    client_name: str
    transcript: str

@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/process-call")
def process_call(data: CallInput):
    try:
        content_text = f"Client: {data.client_name}. Call Details: {data.transcript}"
        hindsight_client.retain(bank_id=BANK_ID, content=content_text)
        return {"status": "success", "message": "Call stored successfully in Hindsight memory!"}
    except Exception as e:
        print(f"Hindsight error: {e}")
        raise HTTPException(status_code=500, detail="Failed to store memory in Hindsight")

@app.post("/briefing")
def get_briefing(client_name: str):
    try:
        query_text = f"Objections, preferences, and history for {client_name}"
        memories = hindsight_client.recall(bank_id=BANK_ID, query=query_text)
    except Exception as e:
        print(f"Recall error: {e}")
        memories = []

    memory_context = str(memories)

    groq_url = "https://api.groq.com/openai/v1/chat/completions"
    groq_headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    You are an expert sales strategist. Based on the following historical memories recalled from Hindsight:
    {memory_context}
    
    Generate a precise, tactical pre-call briefing for {client_name}, highlighting past objections and suggested strategies.
    """
    
    groq_payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    groq_res = requests.post(groq_url, json=groq_payload, headers=groq_headers)
    if groq_res.status_code != 200:
        print(f"GROQ ERROR: {groq_res.text}")
        return {"strategic_brief": f"Groq Error: {groq_res.text}"}
        
    result_text = groq_res.json()["choices"][0]["message"]["content"]
    return {"client": client_name, "memories_used": memories, "strategic_brief": result_text}