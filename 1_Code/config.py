# -*- coding: utf-8 -*-
"""Central configuration — single source of truth for paths, models, and thresholds.
Import this in any notebook:  from config import *  (or import config)."""
from pathlib import Path

PROJECT_DIR   = Path("saudi_labor_law_voice_agent_project")
RAW_DIR       = PROJECT_DIR / "01_raw"
CLEAN_DIR     = PROJECT_DIR / "02_clean"
FINAL_DIR     = PROJECT_DIR / "03_final"
CHUNKS_DIR    = PROJECT_DIR / "04_chunks"
REPORTS_DIR   = PROJECT_DIR / "05_reports"
VECTOR_DIR    = PROJECT_DIR / "06_chroma_db"
STAGE03_DIR   = PROJECT_DIR / "08_stage03_rag_results"
STAGE04_DIR   = PROJECT_DIR / "09_stage04_llm_generation_results"
STAGE05_DIR   = PROJECT_DIR / "10_stage05_rigorous_evaluation"
VOICE_DIR     = PROJECT_DIR / "11_voice_layer"
ABLATION_DIR  = PROJECT_DIR / "12_baselines_ablations"
LATENCY_DIR   = PROJECT_DIR / "13_latency_analysis"

# Local model paths
MODELS = {
    "e5_large":  r"C:/models/multilingual-e5-large",
    "bge_m3":    r"C:/models/bge-m3",
    "reranker":  r"C:/models/bge-reranker-v2-m3",
    "allam_7b":  r"C:/models/ALLaM-7B-Instruct-preview",
    "qwen_7b":   r"C:/models/Qwen2.5-7B-Instruct",
    "whisper":   "openai/whisper-large-v3",
    "tts_arabic":"facebook/mms-tts-ara",
}

# Best retrieval configuration (from Stage 03)
BEST_CONFIG = {
    "chunking_strategy": "structural",
    "embedding_model":   "e5_large",
    "retriever":         "hybrid_reranker",
    "alpha":             0.80,
    "top_k":             5,
    "candidate_k":       50,
    "chroma_collection": "rag_structural_e5_large",
}

# Reliability gate (recalibrated on the full eval set incl. out-of-scope)
RELIABILITY_THRESHOLD = 0.65

# Reproducibility
RANDOM_SEED = 42

# Voice / app
ELEVEN_VOICE_ID = "zVtnf0bbXNM1auWoX2kH"
