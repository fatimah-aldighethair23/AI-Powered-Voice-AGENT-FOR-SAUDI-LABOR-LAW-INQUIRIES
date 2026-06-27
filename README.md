# 🇸🇦 Saudi Labor Law RAG — Knowledge Base & Retrieval Pipeline

مشروع لبناء قاعدة معرفة (Knowledge Base) ونظام استرجاع (Retrieval) باللغة العربية مبني على **نظام العمل السعودي** وأسئلته الشائعة، كخطوة تأسيسية لوكيل صوتي/نصي ذكي (RAG) يجيب على استفسارات نظام العمل.

> **Arabic RAG (Retrieval-Augmented Generation) pipeline for the Saudi Labor Law** — built for a future Arabic AI voice/text agent that answers labor-law questions grounded in official sources.

---

## 🧩 نظرة عامة على المراحل (Pipeline Overview)

المشروع مقسّم إلى ٣ نوتبوكات متسلسلة، كل واحد يعتمد على مخرجات الذي قبله:

| # | Notebook | الوصف |
|---|----------|--------|
| 1 | [`01_Data_Collection_Web_Scraping.ipynb`](notebooks/01_Data_Collection_Web_Scraping.ipynb) | سحب (Scraping) مواد نظام العمل السعودي والأسئلة الشائعة من موقع **وزارة الموارد البشرية والتنمية الاجتماعية (HRSD)**، مع تحويل ترقيم المواد العربي (مثل "الحادية والثلاثون") إلى أرقام صحيحة، وحفظ البيانات الخام. |
| 2 | [`02_Data_Preparation_and_Knowledge_Base.ipynb`](notebooks/02_Data_Preparation_and_Knowledge_Base.ipynb) | تنظيف النصوص العربية (إزالة التطويل، توحيد الترقيم، معالجة المواد "الملغاة" و"مكررة")، إزالة التكرار، وبناء ملف قاعدة المعرفة النظيف الجاهز للمرحلة التالية. |
| 3 | [`03_RAG_Retrieval_Evaluation_ChromaDB.ipynb`](notebooks/03_RAG_Retrieval_Evaluation_ChromaDB.ipynb) | بناء فهرس متجهي (Vector Index) باستخدام **ChromaDB**، وتقييم استراتيجيات الاسترجاع المختلفة (Dense / BM25 / Hybrid / Reranking) باستخدام نماذج `sentence-transformers` على مجموعة أسئلة قانونية مرجعية. |

تشغّل النوتبوكات بالترتيب (1 → 2 → 3)، حيث أن نوتبوك السحب يُشغَّل مرة واحدة فقط عند الحاجة لتحديث البيانات من الموقع الرسمي.

---

## 📂 هيكل المشروع (Repository Structure)

```
saudi-labor-law-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_Data_Collection_Web_Scraping.ipynb
│   ├── 02_Data_Preparation_and_Knowledge_Base.ipynb
│   └── 03_RAG_Retrieval_Evaluation_ChromaDB.ipynb
└── data/
    └── combined_saudi_labor_law_06-10_23.csv
```

---

## 🗂️ ملف البيانات (Dataset)

**`data/combined_saudi_labor_law_06-10_23.csv`** — قاعدة معرفة جاهزة لـ RAG تحتوي على **597 سجلًا** موزعة على **24 تصنيفًا** (الأجور والخصومات، عقد العمل، ساعات العمل، إصابات العمل والسلامة، إنهاء العقد، مكافأة نهاية الخدمة، الإجازات، ...إلخ).

من أهم الأعمدة:

- `question` / `answer` — سؤال وجواب جاهزان للاستخدام في RAG.
- `category`, `intent` — تصنيف الموضوع والنية.
- `article_number`, `article_reference_original` — رقم ومرجع المادة النظامية.
- `source`, `source_authority`, `source_url` — المصدر الرسمي (وزارة الموارد البشرية والتنمية الاجتماعية / هيئة الخبراء بمجلس الوزراء).
- `verification_status`, `academic_reliability`, `quality_grade` — مؤشرات موثوقية وجاهزية السجل للاستخدام في نظام RAG حقيقي.
- `searchable_text`, `citation_text`, `ml_text` — نسخ نصية محسّنة للبحث، للاستشهاد، وللتدريب/التضمين.

> ⚠️ هذا الملف بيانات عامة مأخوذة من مصادر رسمية منشورة، ولا يُعد بديلاً عن استشارة قانونية رسمية.

---

## ⚙️ التشغيل (Getting Started)

```bash
git clone https://github.com/<your-username>/saudi-labor-law-rag.git
cd saudi-labor-law-rag

python -m venv venv
source venv/bin/activate   # على ويندوز: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

ثم افتح النوتبوكات داخل مجلد `notebooks/` بالترتيب الرقمي.

---

## 🛠️ التقنيات المستخدمة (Tech Stack)

- **Web Scraping:** `requests`, `beautifulsoup4`, `lxml`
- **Data Processing:** `pandas`, `numpy`, `scikit-learn`
- **Arabic Text Handling:** `arabic-reshaper`, `python-bidi`
- **Vector DB:** `ChromaDB`
- **Embeddings / Reranking:** `sentence-transformers`, `torch`

---

## 🗺️ خطوات قادمة (Roadmap)

- [ ] دمج طبقة توليد إجابات (Generation) فوق نتائج الاسترجاع لإكمال نظام RAG كامل.
- [ ] بناء واجهة وكيل صوتي (Voice Agent) بالعربية.
- [ ] إضافة تقييم دوري للبيانات عند تحديث نظام العمل رسميًا.

---

## ⚖️ إخلاء مسؤولية (Disclaimer)

هذا المشروع لأغراض بحثية/تعليمية، ويعتمد على بيانات عامة من مصادر حكومية رسمية. المحتوى لا يشكل استشارة قانونية، ويُنصح بالرجوع للنص الرسمي لنظام العمل السعودي وللجهات المختصة عند الحاجة لقرار قانوني فعلي.

## 📄 الترخيص (License)

لم يُحدَّد بعد — أضف ملف `LICENSE` (مثل MIT) حسب رغبتك قبل نشر المشروع كمفتوح المصدر.
