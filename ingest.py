import os
import sys
import json
import time
import re
import numpy as np

# Apply Hugging Face Hub download optimizations BEFORE importing datasets
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"  # 3 minutes timeout
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Faster multi-threaded downloads

# Set HF Token if provided in env
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables
load_dotenv()

# Constants
TARGET_LANG_CONFIG = os.getenv("TARGET_LANGUAGE", "hin_Devn")  # Hindi by default
NUM_SAMPLES = 50  # Number of query items to ingest (each has 10 passages)
VECTOR_INDEX_FILE = "vector_index.npz"

# Map target language configuration to its specific parquet file in the repo
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

print(f"Target language configured: {TARGET_LANG_CONFIG}")
print(f"Target file to download: {parquet_file}")
print(f"Number of queries to process: {NUM_SAMPLES}")

# Helper to split text into sentences (handles Hindi/English boundaries)
def split_sentences(text):
    # Split by standard sentence terminators (. ! ?) and Hindi danda (।)
    sentences = re.split(r'([।\.!\?\n])', text)
    chunks = []
    # Reassemble punctuation with sentence text
    for i in range(0, len(sentences) - 1, 2):
        s = sentences[i].strip()
        p = sentences[i+1]
        if s:
            chunks.append(s + p)
    # Append the last sentence if there's any remainder
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        chunks.append(sentences[-1].strip())
    return [c for c in chunks if c]

# 1. Chunking Strategy: Fixed-Size with Overlap
def chunk_fixed_size(text, size=400, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += (size - overlap)
        if start >= len(text) - overlap:
            break
    return chunks

# 2. Chunking Strategy: Sentence-Boundary Aware
def chunk_sentence_aware(text, max_size=500):
    sentences = split_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > max_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# 3. Chunking Strategy: Structure-Aware (Passage boundaries & original query metadata)
def chunk_structure_aware(passage_text, query_text, english_query, passage_index, is_selected):
    clean_text = passage_text.strip()
    if len(clean_text) > 800:
        clean_text = clean_text[:800]
    return [clean_text]

# Embedding generation (OpenAI text-embedding-3-small or fallback)
def get_embeddings(texts, model="text-embedding-3-small"):
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("WARNING: OPENAI_API_KEY not found in env.")
        print("Using local mock embeddings (random vectors) for scaffolding/testing.")
        print("Please configure OPENAI_API_KEY in .env for real embeddings.")
        dim = 1536
        rng = np.random.default_rng(seed=42)
        return [rng.normal(0, 0.1, dim).tolist() for _ in texts]

    import requests
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }
    
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch = [t.replace("\n", " ") for t in batch]
        payload = {
            "input": batch,
            "model": model
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            res_data = response.json()
            embeddings = [item["embedding"] for item in res_data["data"]]
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"Error fetching embeddings from OpenAI: {e}")
            print("Falling back to mock embeddings for this batch.")
            dim = 1536
            rng = np.random.default_rng(seed=42)
            all_embeddings.extend([rng.normal(0, 0.1, dim).tolist() for _ in batch])
            
    return all_embeddings

def main():
    print(f"Loading dataset parquet file: {parquet_file}...")
    try:
        # Load ONLY the target file, avoiding other languages
        ds = load_dataset(
            "ai4bharat/MSMARCO-XI", 
            data_files={"validation": parquet_file}, 
            split="validation"
        )
        print("Successfully loaded configuration file.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    print("Beginning ingestion processing...")
    chunks_data = []
    
    # Process up to NUM_SAMPLES queries
    query_count = 0
    total_records = len(ds)
    
    # Select the first N records
    subset_indices = range(min(NUM_SAMPLES, total_records))
    
    for idx in subset_indices:
        row = ds[idx]
        query_count += 1
        query_id = row.get("query_id")
        eng_query = row.get("Eng_Query", "")
        target_query = row.get("query", "")
        
        passages = row.get("passages", {})
        translated_passages = passages.get("Translated_passages", [])
        english_passages = passages.get("English_passages", [])
        is_selected = passages.get("is_selected", [])
        
        for p_idx, (trans_p, eng_p, selected) in enumerate(zip(translated_passages, english_passages, is_selected)):
            # Strategy 1: Fixed Overlap
            fixed_chunks = chunk_fixed_size(trans_p)
            for chunk_txt in fixed_chunks:
                chunks_data.append({
                    "text": chunk_txt,
                    "strategy": "fixed-overlap",
                    "query_id": query_id,
                    "passage_index": p_idx,
                    "is_selected": int(selected),
                    "target_lang": TARGET_LANG_CONFIG,
                    "eng_query": eng_query,
                    "target_query": target_query
                })
                
            # Strategy 2: Sentence Aware
            sent_chunks = chunk_sentence_aware(trans_p)
            for chunk_txt in sent_chunks:
                chunks_data.append({
                    "text": chunk_txt,
                    "strategy": "sentence-aware",
                    "query_id": query_id,
                    "passage_index": p_idx,
                    "is_selected": int(selected),
                    "target_lang": TARGET_LANG_CONFIG,
                    "eng_query": eng_query,
                    "target_query": target_query
                })
                
            # Strategy 3: Structure Aware
            struct_chunks = chunk_structure_aware(trans_p, target_query, eng_query, p_idx, selected)
            for chunk_txt in struct_chunks:
                chunks_data.append({
                    "text": chunk_txt,
                    "strategy": "structure-aware",
                    "query_id": query_id,
                    "passage_index": p_idx,
                    "is_selected": int(selected),
                    "target_lang": TARGET_LANG_CONFIG,
                    "eng_query": eng_query,
                    "target_query": target_query
                })
                
        if query_count % 10 == 0:
            print(f"Processed {query_count}/{NUM_SAMPLES} query records...")
            
    print(f"\nIngested {query_count} queries.")
    print(f"Total chunks created: {len(chunks_data)}")
    
    strategies = [c["strategy"] for c in chunks_data]
    from collections import Counter
    print("Chunks count by strategy:", dict(Counter(strategies)))
    
    # Generate embeddings in batches
    print("\nGenerating embeddings for all chunks...")
    chunk_texts = [c["text"] for c in chunks_data]
    t0 = time.time()
    embeddings = get_embeddings(chunk_texts)
    print(f"Generated embeddings in {time.time() - t0:.2f} seconds.")
    
    # Save index as numpy file
    print(f"\nSaving vector index to {VECTOR_INDEX_FILE}...")
    np_embeddings = np.array(embeddings, dtype=np.float32)
    metadata_json = json.dumps(chunks_data, ensure_ascii=False)
    
    np.savez(VECTOR_INDEX_FILE, embeddings=np_embeddings, metadata=metadata_json)
    print("Ingestion complete! Index saved successfully.")

if __name__ == "__main__":
    main()
