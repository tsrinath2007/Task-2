# Voice-Enabled RAG Pipeline Harness (Task 2)

This repository contains an end-to-end voice-enabled Retrieval-Augmented Generation (RAG) system with a real-time analytics dashboard, multiple chunking strategies, and strict guardrails. It retrieves context from a subset of the `ai4bharat/MSMARCO-XI` dataset on Hugging Face.

## System Features

1. **Speech-to-Text (ElevenLabs)**: Real-time transcription integration using the ElevenLabs `scribe_v2` REST API.
2. **Multi-Strategy Ingestion**:
   - *Fixed-Size with Overlap*: Baseline strategy cutting documents into 400-character segments with 100-character overlaps.
   - *Sentence-Boundary Aware*: Preserves sentence integrity (handles Hindi/English delimiters) up to 500 characters.
   - *Structure-Aware*: Maps translated passages, query associations, indices, and ground-truth selections directly.
3. **Guardrails**:
   - *Input Safety Filter*: Fast local matching with offensive keywords.
   - *Off-Topic Detector*: Zero-latency query-context embedding similarity check. If similarity is low, immediately refuses to answer without LLM synthesis.
   - *Groundedness Validation*: Post-generation verification that queries Groq to check if the generated answer is supported by retrieved passages. Refuses/rewrites if hallucinated.
4. **Sub-200ms Latency Target**:
   - Uses a pre-normalized local Vector database (.npz NumPy store) to run search operations in **<1ms**.
   - Leverages **Groq** (`llama-3.1-8b-instant`) for generation with generation constraints, achieving Time-to-First-Token (TTFT) in **<80ms**.
5. **Latency Instrumentation**: Logs execution time for each stage and displays P50, P70, and P100 stats.

---

## Setup & Ingestion

### 1. Install Dependencies
Make sure you have python and `uv` installed, then run:
```bash
uv pip install -r requirements.txt
```

### 2. Configure Environment Keys
Create a `.env` file (copied from `.env.example`):
```bash
cp .env.example .env
```
Fill in the following variables:
- `ELEVENLABS_API_KEY` (For speech queries)
- `GROQ_API_KEY` (For fast LLM replies & groundedness verification)
- `OPENAI_API_KEY` (For vector embeddings)
- `TARGET_LANGUAGE` (Optional, defaults to `hin_Devn` for Hindi)

### 3. Run Ingest Script
Run the ingestion script to stream sample records, apply the chunking strategies, and generate vectors:
```bash
python ingest.py
```
*Note: If no `OPENAI_API_KEY` is present, the script automatically uses highly reproducible mock embeddings so you can build the index and explore the application structure immediately.*

---

## Running the Web Server

Start the local FastAPI server:
```bash
python server.py
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```
- Click the microphone icon to record a voice query.
- Use the dropdown to dynamically switch between **Sentence-Aware**, **Fixed-Size**, or **Structure-Aware** retrieval.
- Observe the real-time latency breakdown chart (STT, Embed, Search, LLM Gen, Guardrails) and check the P50/P70/P100 metrics.

---

## Running Latency Benchmarks

To log a batch of test queries and generate performance analytics (P50/P70/P100 stats), run:
```bash
python benchmark.py
```
The script will fetch 25 validation queries from the `ai4bharat/MSMARCO-XI` dataset configuration, execute them against the pipeline, and output a detailed markdown report of individual step latencies.
These runs will also update the metrics shown on the frontend dashboard!
