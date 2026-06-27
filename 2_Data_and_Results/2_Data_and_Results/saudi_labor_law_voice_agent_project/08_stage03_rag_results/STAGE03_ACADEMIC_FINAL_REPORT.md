# التقرير الأكاديمي النهائي للمرحلة الثالثة

## الهدف والمنهجية

تهدف المرحلة الثالثة إلى بناء وتقييم طبقة الاسترجاع في نظام RAG الخاص بوكيل صوتي لقانون العمل السعودي. اعتمدت التجربة على قاعدة متجهات مستمرة باستخدام ChromaDB، وقارنت بين الاسترجاع الدلالي، والاسترجاع اللفظي BM25، والاسترجاع الهجين، والاسترجاع الهجين مع إعادة الترتيب.

## الإعداد الأفضل المعتمد

- أفضل استراتيجية تقسيم: `structural`
- أفضل نموذج تضمين: `bge_m3`
- أفضل مسترجع: `hybrid_reranker`
- معامل الدمج: `0.8`
- عدد أسئلة التقييم: `235`
- عدد إعدادات التجربة: `22`

## النتائج الرئيسة

حقق الإعداد النهائي `Recall@5 = 0.9745` و`MRR = 0.8472` و`nDCG@5 = 0.8773` مع متوسط زمن استرجاع `0.407` ثانية. تشير هذه النتائج إلى قدرة عالية على إرجاع المرجع القانوني أو المعرفي الصحيح ضمن أعلى خمس نتائج، مع جودة ترتيب قوية تجعل المخرجات مناسبة للانتقال إلى مرحلة توليد الإجابات.

## التحليل حسب نوع السؤال

| eval_type | questions | recall_at_1 | recall_at_3 | recall_at_5 | mrr | ndcg_at_5 | avg_latency_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| faq_paraphrase_retrieval | 131 | 0.9771 | 0.9924 | 1.0 | 0.9866 | 0.99 | 0.4166 |
| legal_article_retrieval | 104 | 0.7115 | 0.8654 | 0.8654 | 0.7804 | 0.8023 | 0.3948 |
| overall_source_aware | 235 | 0.8596 | 0.9362 | 0.9404 | 0.8954 | 0.9069 | 0.407 |

## التحليل التشخيصي

يستخدم التحليل التشخيصي لعزل أثر مطابقة نطاق المصدر والتحقق من السقف العملي للأداء، ولا يستبدل التقييم النهائي الكامل.

| eval_type | questions | diagnostic_recall_at_1 | diagnostic_recall_at_3 | diagnostic_recall_at_5 | diagnostic_mrr | diagnostic_ndcg_at_5 | avg_latency_sec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| faq_paraphrase_retrieval | 131 | 0.9771 | 0.9924 | 1.0 | 0.9866 | 0.99 | 6.021 |
| legal_article_retrieval | 104 | 0.875 | 0.9167 | 0.9167 | 0.8889 | 0.8958 | 5.8044 |
| overall_diagnostic | 235 | 0.9613 | 0.9806 | 0.9871 | 0.9715 | 0.9754 | 5.9252 |

## ضمان الجودة

| check | status | evidence |
| --- | --- | --- |
| required_file::stage03_final_summary_table.csv | PASS | C:\Users\PC\Desktop\data collection code\saudi_labor_law_voice_agent_project\08_stage03_rag_results\stage03_final_summary_table.csv |
| required_file::stage03_final_retrieval_report.json | PASS | C:\Users\PC\Desktop\data collection code\saudi_labor_law_voice_agent_project\08_stage03_rag_results\stage03_final_retrieval_report.json |
| required_file::chroma_collections_manifest.csv | PASS | C:\Users\PC\Desktop\data collection code\saudi_labor_law_voice_agent_project\08_stage03_rag_results\chroma_collections_manifest.csv |
| required_file::retrieval_evaluation_summary.csv | PASS | C:\Users\PC\Desktop\data collection code\saudi_labor_law_voice_agent_project\08_stage03_rag_results\retrieval_evaluation_summary.csv |
| required_file::source_aware_retrieval_summary_by_type.csv | PASS | C:\Users\PC\Desktop\data collection code\saudi_labor_law_voice_agent_project\08_stage03_rag_results\source_aware_retrieval_summary_by_type.csv |
| required_file::source_aware_diagnostic_summary_by_type.csv | PASS | C:\Users\PC\Desktop\data collection code\saudi_labor_law_voice_agent_project\08_stage03_rag_results\source_aware_diagnostic_summary_by_type.csv |
| required_file::stage03_rag_retriever_context_examples.xlsx | PASS | C:\Users\PC\Desktop\data collection code\saudi_labor_law_voice_agent_project\08_stage03_rag_results\stage03_rag_retriever_context_examples.xlsx |
| best_config_is_structural_bge_m3_hybrid_reranker | PASS | {"rank": 1, "chunking_strategy": "structural", "embedding_model": "bge_m3", "retriever": "hybrid_reranker", "alpha": 0.8, "questions": 235, "recall_at_1": 0.7617, "recall_at_3": 0.9277, "recall_at_5": 0.9745, "mrr": 0.8472, "ndcg_at_5": 0.8773, "avg_latency_sec": 0.407, "selection_basis": "Final Stage 03 summary after ChromaDB retrieval, hybrid retrieval, and reranker evaluation."} |
| headline_recall_at_5_ge_0_97 | PASS | Recall@5=0.9745 |
| headline_mrr_ge_0_84 | PASS | MRR=0.8472 |
| chromadb_collections_complete | PASS | [{"collection_name":"rag_structural_e5_large","records_expected":722,"records_in_chroma":722,"status":"OK"},{"collection_name":"rag_structural_bge_m3","records_expected":722,"records_in_chroma":722,"status":"OK"},{"collection_name":"rag_fixed_size_e5_large","records_expected":719,"records_in_chroma":719,"status":"OK"},{"collection_name":"rag_fixed_size_bge_m3","records_expected":719,"records_in_chroma":719,"status":"OK"}] |

## الخلاصة

تؤكد النتائج أن الجمع بين التقسيم البنيوي، ونموذج `bge_m3`، والاسترجاع الهجين مع إعادة الترتيب هو الخيار الأكثر ملاءمة للمرحلة التالية. هذا الاختيار يوازن بين الدقة، جودة الترتيب، وزمن الاستجابة، ويقدم أساسًا موثوقًا لتغذية نموذج اللغة بسياق قانوني مضبوط وقابل للتتبع.
