AI-Powered Voice Agent for Saudi Labor Law Inquiries
=====================================================
Master's project — conversational Arabic voice agent based on Retrieval-Augmented Generation (RAG).

FOLDER STRUCTURE
----------------
1_Code/                 The nine pipeline notebooks (Stage_01 .. Stage_09), config.py, and app/ (the web interface).
2_Data_and_Results/     The project data folder (knowledge base, evaluation sets, and all result files).
3_Documentation/        Pipeline documentation, the consolidated metrics, and the framework diagram.
4_Thesis/               The thesis document (Chapters 1-5).
requirements.txt        Python packages used.

PIPELINE (run in order, kernel = datacollection)
------------------------------------------------
Stage_01  Data collection (web scraping)
Stage_02  Data preparation (cleaning, leakage-safe split, chunking, knowledge base)
Stage_03  RAG retrieval evaluation (ChromaDB + retriever benchmark)
Stage_04  Evaluation-set expansion (24 -> 218 legal questions)
Stage_05  LLM generation (ALLaM vs Qwen)
Stage_06  Rigorous evaluation (bootstrap CIs, McNemar, LLM-as-judge)
Stage_07  Voice layer (Whisper ASR + Arabic TTS, WER)
Stage_08  Baselines and ablations (No-RAG vs RAG)
Stage_09  Latency analysis and optimization

WEB APPLICATION
---------------
Run:  python 1_Code/app/voice_agent_app.py
Then open http://127.0.0.1:7860
Features: text/voice input, voice/text output, selectable RAG type and LLM, conversational memory, safe refusal.

REQUIRED MODELS (NOT included here — download separately, they are public)
--------------------------------------------------------------------------
Place these under C:/models/ (or update config.py):
  - intfloat/multilingual-e5-large        (embedding)
  - BAAI/bge-m3                           (embedding)
  - BAAI/bge-reranker-v2-m3               (reranker)
  - humain-ai/ALLaM-7B-Instruct-preview   (LLM)
  - Qwen/Qwen2.5-7B-Instruct              (LLM)
  - openai/whisper-large-v3               (ASR)
  - facebook/mms-tts-ara                  (TTS)

NOTE
----
- The ElevenLabs API key is intentionally NOT included for security.
- See 3_Documentation/Project_Pipeline_Documentation.docx for a full description of each stage and its outputs.
