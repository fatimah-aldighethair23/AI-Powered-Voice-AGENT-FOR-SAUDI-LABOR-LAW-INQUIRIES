# هيكل الورقة العلمية المقترحة

## العنوان المقترح

وكيل صوتي عربي مدعوم بالاسترجاع المعزز بالتوليد للإجابة عن أسئلة نظام العمل السعودي

عنوان إنجليزي بديل:

An Arabic Retrieval-Augmented Voice Agent for Saudi Labor Law: Framework, Evaluation, and Latency Analysis

## Abstract

This paper presents an Arabic retrieval-augmented voice agent for answering questions related to Saudi Labor Law and HRSD services. The proposed framework combines Arabic legal data preprocessing, leakage-safe evaluation design, ChromaDB-based vector retrieval, hybrid retrieval with reranking, grounded answer generation, and a speech interface. Experiments show that the best retrieval configuration, based on structural chunking, e5_large embeddings, and hybrid reranking, achieved Recall@5 of 0.9935 and MRR of 0.9508. Generation experiments comparing ALLaM-7B-Instruct-preview and Qwen2.5-7B-Instruct showed answer accuracies of 0.7333 and 0.6889, respectively. The voice layer achieved a mean WER of 0.2095, while end-to-end latency analysis showed a current response time of 4.07 seconds and an optimized estimate of 3.03 seconds. The results indicate that retrieval-augmented generation can improve safety and grounding in Arabic legal question answering, although further work is required to improve FAQ coverage, real-world speech robustness, and large-scale validation.

## 1. Introduction

- أهمية الوصول إلى المعلومات القانونية بالعربية.
- تحديات الأسئلة القانونية: الدقة، الاستشهاد، الهلوسة.
- لماذا RAG مناسب للمجال القانوني.
- لماذا إضافة الصوت مهمة لتجربة المستخدم.
- مساهمات الورقة.

## 2. Related Work

- Question answering in legal domain.
- Retrieval-Augmented Generation.
- Arabic NLP and Arabic legal NLP.
- Voice agents and ASR/TTS for Arabic.

## 3. Proposed Framework

- Data sources.
- Preprocessing.
- Leakage-safe split.
- Chunking strategies.
- Embedding and ChromaDB indexing.
- Hybrid retrieval and reranking.
- LLM answer generation.
- Voice layer.

## 4. Experimental Setup

- Dataset size.
- Evaluation questions.
- Models used.
- Metrics:
  - Recall@k.
  - MRR.
  - nDCG@5.
  - Accuracy.
  - Citation hit rate.
  - WER.
  - Latency.

## 5. Results and Discussion

- Retrieval results table.
- LLM generation comparison.
- LLM-as-Judge summary.
- Voice layer result.
- Latency result.
- Baseline and ablation discussion.

## 6. Limitations

- Small generation evaluation sample.
- FAQ performance needs improvement.
- Synthetic or limited voice testing.
- Need expert legal validation.

## 7. Conclusion

- Summary of achieved results.
- Value of RAG for Arabic legal voice assistants.
- Future work.

## ملاحظات مهمة قبل الإرسال

1. يجب تعديل النص بأسلوب الباحثة لتقليل نسبة الاعتماد على الذكاء الاصطناعي.
2. يجب إدخال المراجع الأكاديمية الحقيقية حسب قالب المجلة.
3. يجب التأكد من متطلبات Midocean Journal من حيث عدد الكلمات، تنسيق الجداول، ونظام التوثيق.
4. يجب استخدام الأرقام النهائية بعد حل تعارضات Stage 03.
