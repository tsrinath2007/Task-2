import os
import sys
import time
import json
import requests
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset

# Avoid Windows console encoding issues for Indic characters
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# Config
TARGET_LANG_CONFIG = os.getenv("TARGET_LANGUAGE", "hin_Devn")
NUM_TEST_QUERIES = 25
API_URL = "http://127.0.0.1:8000/api/query"

# Map target language configuration to its specific parquet file
LANG_TO_FILE_MAP = {
    "hin_Devn": "validation/hinval.parquet",
    "tam_Taml": "validation/tamval.parquet",
    "tel_Telu": "validation/telval.parquet",
    "ben_Beng": "validation/benval.parquet",
    "guj_Gujr": "validation/gujval.parquet",
    "mar_Devn": "validation/marval.parquet",
    "kan_Knda": "validation/kanval.parquet",
    "mal_Mlym": "validation/malval.parquet",
    "urd_Arab": "validation/urdval.parquet",
    "pan_Guru": "validation/panval.parquet",
    "asm_Beng": "validation/asmval.parquet",
    "nep_Devn": "validation/nepval.parquet",
    "ory_Orya": "validation/orival.parquet",
    "san_Devn": "validation/sanval.parquet"
}

parquet_file = LANG_TO_FILE_MAP.get(TARGET_LANG_CONFIG, "validation/hinval.parquet")

def run_direct_pipeline(queries):
    """Fallback runner if the FastAPI server is not running."""
    print("FastAPI server is offline. Running queries directly via RAGPipeline class...")
    from harness import RAGPipeline
    
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return []

    results = []
    for idx, q in enumerate(queries):
        print(f"[{idx+1}/{len(queries)}] Querying: '{q}'")
        try:
            response = pipeline.run_pipeline(
                query_text=q,
                strategy="sentence-aware",
                off_topic_threshold=0.35
            )
            results.append({
                "query": q,
                "status": response.status,
                "timings": response.latency_breakdown.model_dump(),
                "success": True
            })
        except Exception as e:
            print(f"Query error: {e}")
            results.append({"query": q, "success": False})
            
    return results

def run_server_pipeline(queries):
    """Queries the local running FastAPI endpoint to measure server overhead."""
    print(f"FastAPI server detected. Running queries via API endpoint: {API_URL}...")
    
    results = []
    for idx, q in enumerate(queries):
        print(f"[{idx+1}/{len(queries)}] Querying API: '{q}'")
        try:
            payload = {
                "query_text": q,
                "strategy": "sentence-aware",
                "off_topic_threshold": 0.35
            }
            response = requests.post(API_URL, data=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results.append({
                "query": q,
                "status": data.get("status"),
                "timings": data.get("latency_breakdown", {}),
                "success": True
            })
        except Exception as e:
            print(f"API request failed: {e}")
            results.append({"query": q, "success": False})
            
    return results

def calculate_percentiles(results):
    valid_runs = [r for r in results if r.get("success", False)]
    if not valid_runs:
        print("No successful runs to calculate analytics.")
        return

    stt_vals = []
    embed_vals = []
    retrieve_vals = []
    llm_vals = []
    guard_vals = []
    rag_vals = []
    total_vals = []

    for r in valid_runs:
        t = r["timings"]
        stt_vals.append(t.get("stt_ms", 0.0))
        emb = t.get("embed_ms", 0.0)
        ret = t.get("retrieve_ms", 0.0)
        llm = t.get("llm_generate_ms", 0.0)
        grd = t.get("guardrails_ms", 0.0)
        tot = t.get("total_ms", 0.0)
        
        embed_vals.append(emb)
        retrieve_vals.append(ret)
        llm_vals.append(llm)
        guard_vals.append(grd)
        total_vals.append(tot)
        
        # RAG Path: embed + retrieve + llm + guard
        rag_vals.append(emb + ret + llm + grd)

    percentiles = [50, 70, 100]
    
    print("\n" + "="*50)
    print("              LATENCY BENCHMARK REPORT")
    print("="*50)
    print(f"Total Successful Queries Run: {len(valid_runs)}")
    print(f"Target Language: {TARGET_LANG_CONFIG}")
    print("-"*50)
    print(f"{'Step':<18} | {'P50 (ms)':<10} | {'P70 (ms)':<10} | {'P100 (ms)':<10}")
    print("-"*50)
    
    steps = [
        ("STT Transcription", stt_vals),
        ("Query Embedding", embed_vals),
        ("Vector Search", retrieve_vals),
        ("Groq Answer Gen", llm_vals),
        ("Guardrails (LLM/Zero)", guard_vals),
        ("Core RAG Path", rag_vals),
        ("Total Pipeline", total_vals)
    ]
    
    for name, vals in steps:
        vals_filtered = [v for v in vals if v > 0] if "STT" in name else vals
        if not vals_filtered:
            p50, p70, p100 = 0.0, 0.0, 0.0
        else:
            p50 = np.percentile(vals_filtered, 50)
            p70 = np.percentile(vals_filtered, 70)
            p100 = np.percentile(vals_filtered, 100)
            
        print(f"{name:<18} | {p50:<10.1f} | {p70:<10.1f} | {p100:<10.1f}")
        
    print("-"*50)
    p50_rag = np.percentile(rag_vals, 50)
    p100_rag = np.percentile(rag_vals, 100)
    print(f"RAG PATH LATENCY TARGET (<200ms):")
    print(f"  - Median (P50) RAG Path: {p50_rag:.1f} ms -> {'PASS' if p50_rag < 200 else 'FAIL'}")
    print(f"  - Worst-case (P100) RAG Path: {p100_rag:.1f} ms -> {'PASS' if p100_rag < 200 else 'FAIL'}")
    print("="*50)

def main():
    print(f"Loading cached dataset parquet file for language config: {TARGET_LANG_CONFIG} ({parquet_file})...")
    try:
        # Load only the cached file
        ds = load_dataset(
            "ai4bharat/MSMARCO-XI", 
            data_files={"validation": parquet_file}, 
            split="validation"
        )
    except Exception as e:
        print(f"Failed to connect to HuggingFace or read cache: {e}")
        sys.exit(1)
        
    test_queries = []
    total_records = len(ds)
    
    # Take the first NUM_TEST_QUERIES queries
    subset_indices = range(min(NUM_TEST_QUERIES, total_records))
    for idx in subset_indices:
        row = ds[idx]
        query = row.get("query")
        if query:
            test_queries.append(query.strip())
                
    if not test_queries:
        print(f"No queries found in {parquet_file}.")
        sys.exit(1)
        
    print(f"Gathered {len(test_queries)} test queries.")
    
    # Check if server is running
    server_online = False
    try:
        res = requests.get("http://127.0.0.1:8000/api/analytics", timeout=2)
        if res.status_code == 200:
            server_online = True
    except Exception:
        pass
        
    if server_online:
        results = run_server_pipeline(test_queries)
    else:
        results = run_direct_pipeline(test_queries)
        
    calculate_percentiles(results)

if __name__ == "__main__":
    main()
