from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE03_DIR = PROJECT_DIR / "08_stage03_rag_results"
REPORTS_DIR = PROJECT_DIR / "05_reports"
SOURCE_NOTEBOOK = PROJECT_DIR.parent / "3-8_RAG_Retrieval_Evaluation_ChromaDB_READY_REVIEWED_MASTER.ipynb"
FINAL_NOTEBOOK = PROJECT_DIR / "3-8_RAG_Retrieval_Evaluation_ChromaDB_STAGE03_FINAL_ACADEMIC.ipynb"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_float(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 4)
    return value


def row_to_clean_dict(row: pd.Series) -> dict:
    return {key: clean_float(value) for key, value in row.to_dict().items()}


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"

    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].map(lambda value: "" if pd.isna(value) else str(value))

    headers = [str(col) for col in display_df.columns]
    rows = display_df.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        escaped = [cell.replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")


def build_final_config(final_summary: pd.DataFrame) -> dict:
    row = final_summary.iloc[0]
    return {
        "rank": 1,
        "chunking_strategy": row["best_chunking_strategy"],
        "embedding_model": row["best_embedding_model"],
        "retriever": row["best_retriever"],
        "alpha": clean_float(row["best_alpha"]),
        "questions": int(row["evaluation_questions"]),
        "recall_at_1": clean_float(row["best_recall_at_1"]),
        "recall_at_3": clean_float(row["best_recall_at_3"]),
        "recall_at_5": clean_float(row["best_recall_at_5"]),
        "mrr": clean_float(row["best_mrr"]),
        "ndcg_at_5": clean_float(row["best_ndcg_at_5"]),
        "avg_latency_sec": clean_float(row["best_avg_latency_sec"]),
        "selection_basis": (
            "Final Stage 03 summary after ChromaDB retrieval, hybrid retrieval, "
            "and reranker evaluation."
        ),
    }


def build_validation_checks(final_summary: pd.DataFrame, final_config: dict) -> pd.DataFrame:
    checks = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
        )

    required_files = [
        STAGE03_DIR / "stage03_final_summary_table.csv",
        STAGE03_DIR / "stage03_final_retrieval_report.json",
        STAGE03_DIR / "chroma_collections_manifest.csv",
        STAGE03_DIR / "retrieval_evaluation_summary.csv",
        STAGE03_DIR / "source_aware_retrieval_summary_by_type.csv",
        STAGE03_DIR / "source_aware_diagnostic_summary_by_type.csv",
        STAGE03_DIR / "stage03_rag_retriever_context_examples.xlsx",
    ]
    for path in required_files:
        add(f"required_file::{path.name}", path.exists(), str(path))

    add(
        "best_config_is_structural_bge_m3_hybrid_reranker",
        final_config["chunking_strategy"] == "structural"
        and final_config["embedding_model"] == "bge_m3"
        and final_config["retriever"] == "hybrid_reranker",
        json.dumps(final_config, ensure_ascii=False),
    )

    add(
        "headline_recall_at_5_ge_0_97",
        float(final_config["recall_at_5"]) >= 0.97,
        f"Recall@5={final_config['recall_at_5']}",
    )

    add(
        "headline_mrr_ge_0_84",
        float(final_config["mrr"]) >= 0.84,
        f"MRR={final_config['mrr']}",
    )

    manifest_path = STAGE03_DIR / "chroma_collections_manifest.csv"
    if manifest_path.exists():
        manifest = read_csv(manifest_path)
        ok = bool((manifest["status"].astype(str).str.upper() == "OK").all())
        evidence = manifest[["collection_name", "records_expected", "records_in_chroma", "status"]].to_json(
            orient="records", force_ascii=False
        )
        add("chromadb_collections_complete", ok, evidence)

    return pd.DataFrame(checks)


def update_configs(final_config: dict, final_summary: pd.DataFrame) -> None:
    best_config = dict(final_config)
    write_json(STAGE03_DIR / "best_retrieval_config.json", best_config)

    ready_for_llm = {
        "best_chunking_strategy": final_config["chunking_strategy"],
        "best_embedding_model": final_config["embedding_model"],
        "best_retriever": final_config["retriever"],
        "best_alpha": final_config["alpha"],
        "top_k": 5,
        "suggested_reliability_threshold": clean_float(
            final_summary.iloc[0].get("suggested_reliability_threshold", 0.4983)
        ),
        "headline_metrics": {
            "recall_at_1": final_config["recall_at_1"],
            "recall_at_3": final_config["recall_at_3"],
            "recall_at_5": final_config["recall_at_5"],
            "mrr": final_config["mrr"],
            "ndcg_at_5": final_config["ndcg_at_5"],
            "avg_latency_sec": final_config["avg_latency_sec"],
        },
        "notes": (
            "Final Stage 03 retriever selected for Stage 04 LLM generation. "
            "Use this retriever to prepare grounded context only; answer generation belongs to Stage 04."
        ),
    }
    write_json(STAGE03_DIR / "stage03_best_retriever_ready_for_llm_config.json", ready_for_llm)


def write_academic_outputs(
    final_summary: pd.DataFrame,
    final_config: dict,
    source_aware: pd.DataFrame,
    diagnostic: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    academic_summary = pd.DataFrame(
        [
            {
                "stage": "Stage 03 - RAG Retrieval Evaluation",
                "evaluation_questions": final_config["questions"],
                "experiment_configurations": int(final_summary.iloc[0]["experiment_configurations"]),
                "best_configuration": (
                    f"{final_config['chunking_strategy']} + {final_config['embedding_model']} + "
                    f"{final_config['retriever']} (alpha={final_config['alpha']})"
                ),
                "recall_at_1": final_config["recall_at_1"],
                "recall_at_3": final_config["recall_at_3"],
                "recall_at_5": final_config["recall_at_5"],
                "mrr": final_config["mrr"],
                "ndcg_at_5": final_config["ndcg_at_5"],
                "avg_latency_sec": final_config["avg_latency_sec"],
                "interpretation": (
                    "The selected retriever demonstrates high top-5 retrieval coverage with strong ranking "
                    "quality, making it suitable for grounding the downstream legal voice agent."
                ),
            }
        ]
    )
    academic_summary.to_csv(STAGE03_DIR / "stage03_academic_final_summary.csv", index=False, encoding="utf-8-sig")
    academic_summary.to_excel(STAGE03_DIR / "stage03_academic_final_summary.xlsx", index=False)
    validation.to_csv(STAGE03_DIR / "stage03_academic_validation_checks.csv", index=False, encoding="utf-8-sig")
    validation.to_excel(STAGE03_DIR / "stage03_academic_validation_checks.xlsx", index=False)

    source_table = dataframe_to_markdown(source_aware)
    diagnostic_table = dataframe_to_markdown(diagnostic)
    validation_table = dataframe_to_markdown(validation)
    report = f"""# التقرير الأكاديمي النهائي للمرحلة الثالثة

## الهدف والمنهجية

تهدف المرحلة الثالثة إلى بناء وتقييم طبقة الاسترجاع في نظام RAG الخاص بوكيل صوتي لقانون العمل السعودي. اعتمدت التجربة على قاعدة متجهات مستمرة باستخدام ChromaDB، وقارنت بين الاسترجاع الدلالي، والاسترجاع اللفظي BM25، والاسترجاع الهجين، والاسترجاع الهجين مع إعادة الترتيب.

## الإعداد الأفضل المعتمد

- أفضل استراتيجية تقسيم: `{final_config['chunking_strategy']}`
- أفضل نموذج تضمين: `{final_config['embedding_model']}`
- أفضل مسترجع: `{final_config['retriever']}`
- معامل الدمج: `{final_config['alpha']}`
- عدد أسئلة التقييم: `{final_config['questions']}`
- عدد إعدادات التجربة: `{int(final_summary.iloc[0]['experiment_configurations'])}`

## النتائج الرئيسة

حقق الإعداد النهائي `Recall@5 = {final_config['recall_at_5']}` و`MRR = {final_config['mrr']}` و`nDCG@5 = {final_config['ndcg_at_5']}` مع متوسط زمن استرجاع `{final_config['avg_latency_sec']}` ثانية. تشير هذه النتائج إلى قدرة عالية على إرجاع المرجع القانوني أو المعرفي الصحيح ضمن أعلى خمس نتائج، مع جودة ترتيب قوية تجعل المخرجات مناسبة للانتقال إلى مرحلة توليد الإجابات.

## التحليل حسب نوع السؤال

{source_table}

## التحليل التشخيصي

يستخدم التحليل التشخيصي لعزل أثر مطابقة نطاق المصدر والتحقق من السقف العملي للأداء، ولا يستبدل التقييم النهائي الكامل.

{diagnostic_table}

## ضمان الجودة

{validation_table}

## الخلاصة

تؤكد النتائج أن الجمع بين التقسيم البنيوي، ونموذج `bge_m3`، والاسترجاع الهجين مع إعادة الترتيب هو الخيار الأكثر ملاءمة للمرحلة التالية. هذا الاختيار يوازن بين الدقة، جودة الترتيب، وزمن الاستجابة، ويقدم أساسًا موثوقًا لتغذية نموذج اللغة بسياق قانوني مضبوط وقابل للتتبع.
"""
    (STAGE03_DIR / "STAGE03_ACADEMIC_FINAL_REPORT.md").write_text(report, encoding="utf-8")


def append_notebook_audit(final_config: dict, validation: pd.DataFrame) -> None:
    if not SOURCE_NOTEBOOK.exists():
        return

    shutil.copy2(SOURCE_NOTEBOOK, FINAL_NOTEBOOK)
    notebook = json.loads(FINAL_NOTEBOOK.read_text(encoding="utf-8"))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    validation_preview = dataframe_to_markdown(validation)

    notebook["cells"].append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ملحق الإقفال الأكاديمي للمرحلة الثالثة\\n",
                "\\n",
                f"تم توليد هذا الملحق داخل مجلد المشروع بتاريخ `{timestamp}` بعد توحيد ملفات الإعداد والتقرير النهائي.\\n",
                "\\n",
                "### الإعداد النهائي المعتمد\\n",
                f"- `{final_config['chunking_strategy']} + {final_config['embedding_model']} + {final_config['retriever']} (alpha={final_config['alpha']})`\\n",
                f"- `Recall@5 = {final_config['recall_at_5']}`\\n",
                f"- `MRR = {final_config['mrr']}`\\n",
                f"- `nDCG@5 = {final_config['ndcg_at_5']}`\\n",
                "\\n",
                "### فحوصات الإقفال\\n",
                validation_preview,
                "\\n",
            ],
        }
    )
    FINAL_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def update_output_manifest(new_files: list[Path]) -> None:
    manifest_path = STAGE03_DIR / "stage03_output_manifest.csv"
    if manifest_path.exists():
        manifest = read_csv(manifest_path)
    else:
        manifest = pd.DataFrame(columns=["file_name", "path", "exists", "size_bytes"])

    rows = []
    for path in new_files:
        rows.append(
            {
                "file_name": path.name,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    combined = pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["file_name", "path"], keep="last")
    combined.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    combined.to_excel(STAGE03_DIR / "stage03_output_manifest.xlsx", index=False)


def main() -> None:
    require_file(STAGE03_DIR / "stage03_final_summary_table.csv")
    require_file(STAGE03_DIR / "source_aware_retrieval_summary_by_type.csv")
    require_file(STAGE03_DIR / "source_aware_diagnostic_summary_by_type.csv")

    final_summary = read_csv(STAGE03_DIR / "stage03_final_summary_table.csv")
    source_aware = read_csv(STAGE03_DIR / "source_aware_retrieval_summary_by_type.csv")
    diagnostic = read_csv(STAGE03_DIR / "source_aware_diagnostic_summary_by_type.csv")
    final_config = build_final_config(final_summary)

    update_configs(final_config, final_summary)
    validation = build_validation_checks(final_summary, final_config)
    write_academic_outputs(final_summary, final_config, source_aware, diagnostic, validation)
    append_notebook_audit(final_config, validation)

    new_files = [
        STAGE03_DIR / "best_retrieval_config.json",
        STAGE03_DIR / "stage03_best_retriever_ready_for_llm_config.json",
        STAGE03_DIR / "stage03_academic_final_summary.csv",
        STAGE03_DIR / "stage03_academic_final_summary.xlsx",
        STAGE03_DIR / "stage03_academic_validation_checks.csv",
        STAGE03_DIR / "stage03_academic_validation_checks.xlsx",
        STAGE03_DIR / "STAGE03_ACADEMIC_FINAL_REPORT.md",
        FINAL_NOTEBOOK,
    ]
    update_output_manifest(new_files)

    failed = validation[validation["status"] != "PASS"]
    print("Final Stage 03 configuration:")
    print(json.dumps(final_config, ensure_ascii=False, indent=2))
    print(f"Validation checks: {len(validation) - len(failed)}/{len(validation)} passed")
    if len(failed):
        print(failed.to_string(index=False))
        raise SystemExit(1)
    print(f"Academic report: {STAGE03_DIR / 'STAGE03_ACADEMIC_FINAL_REPORT.md'}")
    print(f"Final notebook: {FINAL_NOTEBOOK}")


if __name__ == "__main__":
    main()
