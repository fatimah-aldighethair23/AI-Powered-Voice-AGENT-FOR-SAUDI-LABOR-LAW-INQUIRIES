# Project Review Findings

Review date: 2026-06-21

## Executive Summary

The pipeline is mostly complete and the core outputs are present. The notebooks are largely executed and no stored execution errors were found. The strongest result is the Stage 03 retrieval run, with structural chunking, e5_large embeddings, hybrid reranking, Recall@5 = 0.9935, MRR = 0.9508, and average retrieval latency = 0.3361 seconds.

The main improvement need is consistency and presentation readiness. Several summary files disagree on headline counts or selected configurations, and the PowerPoint currently contains only one dense framework slide.

## High Priority Findings

1. Stage 03 validation file is stale or inconsistent.
   - `08_stage03_rag_results/stage03_final_summary_table.csv` reports best model `e5_large`, 155 evaluation questions, Recall@5 = 0.9935, MRR = 0.9508.
   - `08_stage03_rag_results/stage03_final_retrieval_report.json` agrees with that same `e5_large` configuration.
   - `08_stage03_rag_results/stage03_academic_validation_checks.csv` still validates `bge_m3`, 235 questions, Recall@5 = 0.9745, MRR = 0.8472, and Chroma counts around 719-722 records.
   - Recommended fix: regenerate the validation checks from the final Stage 03 report and make the check names model-agnostic.

2. Reliability threshold is inconsistent.
   - `stage03_final_summary_table.csv` reports suggested threshold = 0.65.
   - `stage03_final_retrieval_report.json` and `stage03_best_retriever_ready_for_llm_config.json` report suggested threshold = 0.7492.
   - Recommended fix: choose one threshold, document how it was selected, and propagate it to all downstream stages.

3. FAQ split counts disagree across reports.
   - `05_reports/dataset_summary.csv` reports FAQ indexing = 357 and held-out FAQ = 112.
   - `05_reports/data_quality_report.csv` reports FAQ indexing = 278 and evaluation after leakage checks = 122.
   - Recommended fix: define canonical split files and make all summaries read from those files only.

4. Generation evaluation uses a small sample and aggressive refusal behavior.
   - Stage 05 evaluates 45 questions per model.
   - ALLaM accuracy = 0.7333, Qwen accuracy = 0.6889, but refusal rates are high: 0.4667 and 0.5333.
   - FAQ-only accuracy is weak in rigorous evaluation: ALLaM = 0.4286, Qwen = 0.2143.
   - Recommended fix: separate legal, FAQ, small-talk, and out-of-scope evaluation policies; tune retrieval threshold separately by eval type.

5. No-RAG vs RAG result needs careful explanation.
   - `12_baselines_ablations/norag_vs_rag_summary.csv` shows No-RAG correct citation rate = 0.292 and RAG = 0.208, while RAG has much lower wrong citation rate = 0.042 versus 0.25.
   - Recommended framing: RAG improved citation safety, not citation hit rate, on this small 24-question subset.

## Medium Priority Findings

1. The stored notebooks are mostly clean, but a few are only partially executed:
   - Stage 01: 11/12 code cells executed.
   - Stage 04: 5/6 code cells executed.
   - Stage 05: 27/28 code cells executed.
   - Recommended fix: rerun all notebooks from a clean kernel and save outputs.

2. Stage 06 markdown says LLM-as-Judge is "قيد الإضافة", but output files exist.
   - Recommended fix: update the markdown to reflect the completed LLM judge stage.

3. Stage 05 has an output line containing "Error-analysis rows", which is not an execution error but can look alarming.
   - Recommended fix: rename the print label to "Failure-analysis rows" or "Incorrect/refusal rows".

4. Latency analysis is internally consistent.
   - Current pipeline = 4.07 sec.
   - Optimized pipeline = 3.03 sec.
   - Streaming perceived time-to-first-audio = 0.95 sec.
   - Recommended fix: add confidence intervals or repeated-run averages if this will be defended as an empirical result.

5. Voice layer WER is reasonable but needs a benchmark.
   - Mean WER = 0.2095 over 40 samples.
   - Recommended fix: report WER by dialect/noise/question type and compare large-v3 vs small if latency optimization claims use small ASR.

## Presentation Readiness

The PowerPoint file contains one slide with a dense framework diagram. It is useful as a system overview, but it is not enough for a complete academic/project presentation.

Recommended deck structure:

1. Problem and scope.
2. Data sources and cleaning.
3. Leakage-safe evaluation design.
4. RAG architecture and ChromaDB indexing.
5. Retrieval experiments and final configuration.
6. LLM generation results.
7. Rigorous evaluation and statistical uncertainty.
8. Voice layer and latency.
9. Baselines and ablations.
10. Limitations and future improvements.

## Best Immediate Improvements

1. Fix Stage 03 consistency issues first.
2. Create one canonical final summary table that all notebook conclusions and slides use.
3. Rerun notebooks from clean kernels and save all cells executed.
4. Expand Stage 05/06 evaluation samples, especially FAQ and out-of-scope cases.
5. Convert the single-slide PowerPoint into a 10-slide results deck.
6. Add a final reproducibility manifest with input files, output files, run date, model names, thresholds, and selected metrics.
