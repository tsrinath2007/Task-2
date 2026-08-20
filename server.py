import os
import sys
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import Pipeline
from harness import RAGPipeline

app = FastAPI(title="Voice-Enabled RAG Pipeline Dashboard")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Pipeline on startup
pipeline = None
try:
    pipeline = RAGPipeline()
except Exception as e:
    print(f"Warning during pipeline initialization: {e}")

# Helper to calculate percentiles
def get_percentile(data, q):
    if not data:
        return 0.0
    return float(np.percentile(data, q))

@app.post("/api/query")
async def query_endpoint(
    query_text: Optional[str] = Form(None),
    strategy: str = Form("sentence-aware"),
    off_topic_threshold: float = Form(0.35),
    file: Optional[UploadFile] = File(None)
):
    if not pipeline or pipeline.embeddings is None:
        raise HTTPException(status_code=500, detail="Vector index not loaded. Please run ingest.py first.")

    audio_bytes = None
    if file:
        audio_bytes = await file.read()

    try:
        response = pipeline.run_pipeline(
            query_text=query_text,
            audio_bytes=audio_bytes,
            strategy=strategy,
            off_topic_threshold=off_topic_threshold
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.get("/api/analytics")
async def get_analytics():
    log_file = "latency_logs.json"
    if not os.path.exists(log_file):
        return {
            "total_runs": 0,
            "P50": {}, "P70": {}, "P100": {},
            "raw_logs": []
        }

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")

    if not logs:
        return {
            "total_runs": 0,
            "P50": {}, "P70": {}, "P100": {},
            "raw_logs": []
        }

    # Extract timings
    stt_times = []
    embed_times = []
    retrieve_times = []
    llm_times = []
    guardrails_times = []
    total_times = []
    rag_path_times = []  # Retrieval + Generation path (excludes STT)

    for entry in logs:
        t = entry.get("timings", {})
        stt = t.get("stt_ms", 0.0)
        emb = t.get("embed_ms", 0.0)
        ret = t.get("retrieve_ms", 0.0)
        llm = t.get("llm_generate_ms", 0.0)
        grd = t.get("guardrails_ms", 0.0)
        tot = t.get("total_ms", 0.0)

        if stt > 0:
            stt_times.append(stt)
            
        embed_times.append(emb)
        retrieve_times.append(ret)
        llm_times.append(llm)
        guardrails_times.append(grd)
        total_times.append(tot)
        
        # Core RAG latency: everything excluding STT (embed + retrieve + llm + guardrails)
        rag_path_times.append(emb + ret + llm + grd)

    # Calculations
    percentiles = [50, 70, 100]
    p50_dict = {}
    p70_dict = {}
    p100_dict = {}

    for p in percentiles:
        target_dict = p50_dict if p == 50 else (p70_dict if p == 70 else p100_dict)
        target_dict["stt"] = get_percentile(stt_times, p)
        target_dict["embed"] = get_percentile(embed_times, p)
        target_dict["retrieve"] = get_percentile(retrieve_times, p)
        target_dict["llm_generate"] = get_percentile(llm_times, p)
        target_dict["guardrails"] = get_percentile(guardrails_times, p)
        target_dict["total"] = get_percentile(total_times, p)
        target_dict["rag_path"] = get_percentile(rag_path_times, p)

    return {
        "total_runs": len(logs),
        "P50": p50_dict,
        "P70": p70_dict,
        "P100": p100_dict,
        "raw_logs": logs[-10:]  # Return last 10 entries for log table display
    }

@app.post("/api/reset_analytics")
async def reset_analytics():
    log_file = "latency_logs.json"
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            return {"status": "success", "message": "Logs reset successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reset logs: {str(e)}")
    return {"status": "success", "message": "Logs were already empty."}

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
