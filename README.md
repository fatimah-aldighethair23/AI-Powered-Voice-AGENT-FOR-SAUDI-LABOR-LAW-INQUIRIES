# AI-Powered Voice Agent for Saudi Labor Law Inquiries 🇸🇦⚖️

[span_6](start_span)This project develops an advanced AI-based voice agent designed to answer inquiries related to **Saudi Labor Law** using natural Arabic speech[span_6](end_span). [span_7](start_span)[span_8](start_span)It aims to replace traditional IVR systems with a more intuitive, real-time conversational experience[span_7](end_span)[span_8](end_span).

## 🎯 Research Objectives
* **[span_9](start_span)Natural Interaction:** Allow users to speak naturally without fixed menus[span_9](end_span).
* **[span_10](start_span)[span_11](start_span)Accuracy:** Utilize **RAG (Retrieval-Augmented Generation)** to ensure legal precision[span_10](end_span)[span_11](end_span).
* **[span_12](start_span)Privacy:** Use fully open-source components deployed locally for data control[span_12](end_span).
* **[span_13](start_span)Efficiency:** Reduce waiting times and provide 24/7 accessibility[span_13](end_span).

## 🏗️ System Architecture (Proposed)
[span_14](start_span)The system follows a modular pipeline[span_14](end_span):
1. **[span_15](start_span)Streaming ASR:** Speech-to-Text conversion (comparing Whisper, GPT-4o, etc.)[span_15](end_span).
2. **[span_16](start_span)RAG Engine:** Retrieving relevant law articles from a Vector Database (FAISS/Chroma)[span_16](end_span).
3. **[span_17](start_span)Arabic LLM:** Generating accurate responses (LLaMA/Qwen)[span_17](end_span).
4. **[span_18](start_span)Streaming TTS:** Converting text back to natural Arabic speech[span_18](end_span).

## 📚 Key Technologies & References
* **[span_19](start_span)Framework:** Python, RAG, Vector DB[span_19](end_span).
* **[span_20](start_span)[span_21](start_span)Speech Models:** CTC-TTS, SpeechGPT, LLMVoX[span_20](end_span)[span_21](end_span).
* **[span_22](start_span)[span_23](start_span)Benchmarks:** ALARB (Arabic Legal Argument Reasoning Benchmark)[span_22](end_span)[span_23](end_span).

---
**[span_24](start_span)Supervisor:** Dr. Hager Saleh[span_24](end_span)  
**[span_25](start_span)Team:** Fatimah Aldigheyhair & Hussam Almokayed[span_25](end_span)
