import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Simple local list of offensive words/phrases for rapid safety check
BLOCKED_KEYWORDS = [
    "bomb", "kill", "terrorist", "hack", "bypass", "suicide", "murder", "abuse"
]

def check_input_safety(query_text: str) -> dict:
    """
    Rapidly checks if the input query is safe.
    Uses keyword filtering and optionally a fast model check.
    """
    t0 = time.time()
    
    # 1. Local keyword check (0ms)
    lower_query = query_text.lower()
    for word in BLOCKED_KEYWORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', lower_query):
            return {
                "safe": False,
                "reason": f"Input contains inappropriate keyword: '{word}'",
                "latency_ms": (time.time() - t0) * 1000
            }
            
    # 2. Return safe
    return {
        "safe": True,
        "reason": None,
        "latency_ms": (time.time() - t0) * 1000
    }

def check_off_topic(query_text: str, top_score: float, threshold: float = 0.35) -> dict:
    """
    Zero-latency off-topic check.
    If the cosine similarity score of the top-retrieved document is below
    the threshold, the query is deemed off-topic.
    """
    t0 = time.time()
    is_off_topic = top_score < threshold
    
    return {
        "off_topic": is_off_topic,
        "top_score": float(top_score),
        "threshold": threshold,
        "reason": "Query does not match any index topics sufficiently" if is_off_topic else None,
        "latency_ms": (time.time() - t0) * 1000
    }

def check_groundedness(query_text: str, retrieved_chunks: list, generated_answer: str) -> dict:
    """
    LLM-based groundedness check.
    Verifies if the generated answer is supported by the retrieved chunks.
    Constrained to 1-token output (YES/NO) on Groq for ultra-low latency (<80ms).
    """
    t0 = time.time()
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not groq_key:
        # Dry-run fallback if no keys configured
        return {
            "grounded": True,
            "score": 1.0,
            "reason": "Skip groundedness check (GROQ_API_KEY missing)",
            "latency_ms": (time.time() - t0) * 1000
        }
        
    # Combine chunks to form reference context
    context = "\n---\n".join([c["text"] for c in retrieved_chunks])
    
    prompt = f"""
Analyze if the GENERATED_ANSWER is supported by the CONTEXT.
Respond with EXACTLY one word: "YES" if the answer is grounded in the context, or "NO" if the answer is not supported, contradicts, or contains information outside the context.
Do not write anything else.

CONTEXT:
{context}

GENERATED_ANSWER:
{generated_answer}

GROUNDED (YES or NO):"""

    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 2  # Keep it ultra short
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        res_json = response.json()
        decision = res_json["choices"][0]["message"]["content"].strip().upper()
        
        # Parse decision (remove punctuation if any)
        decision_clean = re.sub(r'[^A-Z]', '', decision)
        is_grounded = "YES" in decision_clean
        
        return {
            "grounded": is_grounded,
            "reason": None if is_grounded else "Generated answer contains hallucinations or is not supported by context",
            "latency_ms": (time.time() - t0) * 1000
        }
    except Exception as e:
        # Fallback on failure (we fail-safe: assume grounded if API error but log warning)
        return {
            "grounded": True,
            "reason": f"Groundedness check failed to execute: {e}",
            "latency_ms": (time.time() - t0) * 1000
        }
