# -*- coding: utf-8 -*-
"""
AI-Powered Voice Agent for Saudi Labor Law Inquiries — Web Interface
====================================================================
واجهة ويب: سؤال نصّي أو صوتي -> اختيار نوع RAG واختيار نموذج LLM -> إجابة نصّية وصوتية.

Run:  python app/voice_agent_app.py
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent / "saudi_labor_law_voice_agent_project"
CHUNKS = PROJECT_DIR / "04_chunks" / "rag_chunks_structural_legal_experiment.csv"
RELIABILITY_THRESHOLD = 0.65          # recalibrated reliability gate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading corpus ...")
df = pd.read_csv(CHUNKS, encoding="utf-8-sig")
CORPUS = df["chunk_text"].fillna("").astype(str).tolist()
META = df.to_dict("records")

def _norm(t):
    t = re.sub(r"[ًٌٍَُِّْـ]", "", str(t)); t = re.sub(r"[إأآا]", "ا", t)
    t = re.sub(r"ى", "ي", t); t = re.sub(r"ة", "ه", t)
    return re.sub(r"\s+", " ", t).strip()

# ---------------------------------------------------------------- retrieval models
print("Loading embedding / reranker / BM25 ...")
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

EMB = {
    "e5-large":  {"model": SentenceTransformer(r"C:/models/multilingual-e5-large", device=DEVICE),
                  "qpfx": "query: ", "ppfx": "passage: "},
    "bge-m3":    {"model": SentenceTransformer(r"C:/models/bge-m3", device=DEVICE),
                  "qpfx": "", "ppfx": ""},
}
RERANKER = CrossEncoder(r"C:/models/bge-reranker-v2-m3", device=DEVICE)

# pre-encode corpus once per embedding model
CORPUS_EMB = {}
for name, cfg in EMB.items():
    CORPUS_EMB[name] = cfg["model"].encode([cfg["ppfx"] + c for c in CORPUS],
                                            normalize_embeddings=True, batch_size=64,
                                            show_progress_bar=False)
BM25 = BM25Okapi([_norm(c).split() for c in CORPUS])

def _mm(x):
    x = np.asarray(x, float); rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

def retrieve(question, rag_type="Hybrid + Reranker", emb_choice="e5-large", top_k=3, alpha=0.65):
    cfg = EMB[emb_choice]
    qv = cfg["model"].encode(cfg["qpfx"] + question, normalize_embeddings=True)
    dense = CORPUS_EMB[emb_choice] @ qv                      # cosine sim (normalized)
    if rag_type == "Dense":
        scores = dense
    else:
        bm = BM25.get_scores(_norm(question).split())
        scores = alpha * _mm(dense) + (1 - alpha) * _mm(bm)  # hybrid
    order = np.argsort(scores)[::-1][:30]
    if rag_type == "Hybrid + Reranker":
        rr = RERANKER.predict([[question, CORPUS[i]] for i in order])
        order = order[np.argsort(rr)[::-1]]
    top = order[:top_k]
    # reliability score = reranker relevance of the #1 chunk (consistent across rag types)
    rel = float(torch.sigmoid(torch.tensor(RERANKER.predict([[question, CORPUS[top[0]]]])[0])).item())
    ctx = [{"text": CORPUS[i], "article": META[i].get("article_number_int", META[i].get("article_number")),
            "source": META[i].get("source_type")} for i in top]
    return ctx, rel

# ---------------------------------------------------------------- LLMs
print("Loading LLMs (ALLaM + Qwen, 4-bit) ...")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
_qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
LLMS = {}
for key, path in [("ALLaM-7B", r"C:/models/ALLaM-7B-Instruct-preview"),
                  ("Qwen2.5-7B", r"C:/models/Qwen2.5-7B-Instruct")]:
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForCausalLM.from_pretrained(path, device_map="auto", quantization_config=_qc)
    mdl.eval()
    LLMS[key] = (tok, mdl)

def build_messages(question, ctx, reliable):
    if not reliable or not ctx:
        return [{"role": "system", "content": "أنت مساعد قانوني عربي لنظام العمل السعودي."},
                {"role": "user", "content": f"السؤال: {question}\n\nلا يوجد سياق موثوق كافٍ. "
                 "قل بوضوح: المعلومات المتاحة غير كافية للإجابة بدقة، واطلب إعادة الصياغة."}]
    blocks = []
    for c in ctx:
        if pd.notna(c.get("article")):
            head = f"المادة رقم {int(c['article'])} من نظام العمل السعودي"
        else:
            head = "معلومة من قاعدة المعرفة"
        blocks.append(f"{head}:\n{c['text'][:650]}")
    ctx_text = "\n\n".join(blocks)
    sys = ("أنت مساعد قانوني عربي متخصص في نظام العمل السعودي. أجب اعتماداً فقط على المواد المرفقة أدناه. "
           "اختر المادة الأكثر صلة بالسؤال. "
           "إذا وصف المستخدم موقفاً أو حالة وسأل عن حقّه أو حكمه (مثل: «طُردت في فترة التجربة، هل يحق ذلك؟»)، "
           "فطبّق المادة على حالته وابدأ بحكم مباشر وواضح (نعم/لا ولماذا)، ثم استشهد برقم المادة. "
           "لا تكتفِ بنقل نص المادة حرفياً، بل اشرح ما يعنيه لموقفه بإيجاز. "
           "اذكر رقم المادة صراحةً بصيغة: «وفقاً للمادة رقم (الرقم) من نظام العمل السعودي». "
           "ممنوع منعاً باتاً ذكر كلمة «السياق» أو عبارات مثل «حسب السياق رقم». "
           "لا تخترع رقم مادة أو حكماً غير موجود في المواد المرفقة. "
           "إذا كانت المواد المرفقة لا تجيب عن السؤال فقل: المعلومات المتاحة غير كافية للإجابة بدقة. "
           "اجعل الإجابة مختصرة وواضحة ومناسبة للنطق الصوتي.")
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"المواد المرفقة:\n{ctx_text}\n\nالسؤال: {question}\n\nالإجابة:"},
    ]

def generate(question, ctx, reliable, llm_choice="ALLaM-7B", max_new=180):
    tok, mdl = LLMS[llm_choice]
    msgs = build_messages(question, ctx, reliable)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=3200).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=max_new, do_sample=False,
                           repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

def stream_messages(msgs, llm_choice="ALLaM-7B", max_new=130):
    """Yield the answer progressively (token streaming) to minimise perceived latency."""
    tok, mdl = LLMS[llm_choice]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=3200).to(mdl.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(inp, max_new_tokens=max_new, do_sample=False,
                  repetition_penalty=1.05, pad_token_id=tok.eos_token_id, streamer=streamer)
    Thread(target=mdl.generate, kwargs=kwargs).start()
    partial = ""
    for piece in streamer:
        partial += piece
        yield partial.strip()

# ---------------------------------------------------------------- intent gate (small-talk vs domain)
SMALLTALK_PATTERNS = [
    "السلام عليكم", "سلام عليكم", "هلا", "هلو", "مرحبا", "اهلا", "صباح", "مساء",
    "كيف حالك", "كيفك", "كيف الحال", "شلونك", "شخبارك", "اخبارك", "عساك بخير",
    "من انت", "مين انت", "وش اسمك", "ما اسمك", "عرف نفسك", "عرفني بنفسك", "من تكون",
    "وش تسوي", "وش تقدر", "ايش تقدر", "كيف تساعدني", "وش تساعدني", "ماذا تفعل",
    "ماذا تستطيع", "وش وظيفتك", "شكرا", "مشكور", "يعطيك العافيه", "تسلم", "الله يعافيك",
]

def classify_intent(question):
    qn = _norm(question)
    if any(_norm(p) in qn for p in SMALLTALK_PATTERNS):
        return "small_talk"
    return "domain"

def generate_smalltalk(question, llm_choice="ALLaM-7B", max_new=120):
    tok, mdl = LLMS[llm_choice]
    msgs = [
        {"role": "system", "content":
         "أنت مساعد صوتي ودود ومحترف، متخصص في نظام العمل السعودي وخدمات وزارة الموارد "
         "البشرية والتنمية الاجتماعية. ردّ باختصار وبأسلوب لبق. إذا سُئلت من أنت، عرّف نفسك "
         "كمساعد ذكي للإجابة عن أسئلة نظام العمل السعودي. لا تذكر مواد قانونية إلا إذا طُرح سؤال قانوني صريح."},
        {"role": "user", "content": question},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=max_new, do_sample=False,
                           repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

# ---------------------------------------------------------------- voice (ASR + TTS)
print("Loading voice layer (Whisper ASR + Microsoft neural Arabic TTS) ...")
from transformers import pipeline
import asyncio, edge_tts, tempfile, os, uuid
ASR = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
               device=0 if DEVICE == "cuda" else -1, torch_dtype=torch.float16)

# Microsoft neural voices (edge-tts) — always available fallback
TTS_VOICES = {"رجل (Hamed)": "ar-SA-HamedNeural", "امرأة (Zariyah)": "ar-SA-ZariyahNeural"}

# ElevenLabs (best quality) — activated when an API key is provided
ELEVEN_VOICE_ID = "zVtnf0bbXNM1auWoX2kH"
_keyfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elevenlabs_key.txt")
ELEVEN_KEY = open(_keyfile, encoding="utf-8").read().strip() if os.path.exists(_keyfile) else ""
ELEVEN_KEY = ELEVEN_KEY or os.environ.get("ELEVENLABS_API_KEY", "")
_eleven = None
if ELEVEN_KEY:
    try:
        from elevenlabs.client import ElevenLabs
        _eleven = ElevenLabs(api_key=ELEVEN_KEY)
        print(f"ElevenLabs TTS enabled (voice {ELEVEN_VOICE_ID}).")
    except Exception as e:
        print("ElevenLabs init failed:", e); _eleven = None
print("TTS engines:", "ElevenLabs + edge-tts" if _eleven else "edge-tts only (add app/elevenlabs_key.txt to enable ElevenLabs)")

def transcribe(audio_path):
    if not audio_path:
        return ""
    return ASR(audio_path, generate_kwargs={"language": "arabic"})["text"].strip()

def _speak_elevenlabs(text):
    out_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
    audio = _eleven.text_to_speech.convert(
        voice_id=ELEVEN_VOICE_ID, model_id="eleven_multilingual_v2",
        text=text[:1500], output_format="mp3_44100_128")
    with open(out_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    return out_path

def _speak_edge(text, voice_label):
    voice = TTS_VOICES.get(voice_label, "ar-SA-HamedNeural")
    out_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(edge_tts.Communicate(text[:600], voice).save(out_path))
    finally:
        loop.close()
    return out_path

def speak(text, voice_label="ElevenLabs (Nora)"):
    if str(voice_label).startswith("ElevenLabs") and _eleven is not None:
        try:
            return _speak_elevenlabs(text)
        except Exception as e:
            print("ElevenLabs TTS error -> fallback to edge-tts:", e)
    return _speak_edge(text, voice_label)

# ---------------------------------------------------------------- pipeline + UI
SMALLTALK_SYS = ("أنت مساعد صوتي ودود متخصص في نظام العمل السعودي وخدمات وزارة الموارد البشرية. "
                 "ردّ بجملة واحدة قصيرة جداً ومباشرة دون إطالة أو مقدمات. "
                 "عند الشكر: قل «العفو» ثم اسأل باختصار إن كان لديه استفسار آخر يخص نظام العمل تساعده فيه. "
                 "عند السؤال عن هويتك: عرّف نفسك بجملة واحدة. "
                 "لا تذكر مواد قانونية إلا إذا طُرح سؤال قانوني صريح.")
SAFE_REFUSAL = ("عذراً، هذا السؤال يبدو خارج نطاق تخصصي. أنا مساعد متخصص في نظام العمل السعودي "
                "وخدمات وزارة الموارد البشرية والتنمية الاجتماعية فقط. "
                "هل لديك سؤال يتعلق بنظام العمل أو حقوق العامل وصاحب العمل؟")

def _complete(msgs, llm_choice="ALLaM-7B", max_new=64):
    tok, mdl = LLMS[llm_choice]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt", truncation=True, max_length=1600).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

def _msg_text(m):
    """Extract plain text from a chat message whose content may be a str or a list (Gradio 6)."""
    c = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
    if isinstance(c, (list, tuple)):
        parts = []
        for x in c:
            if isinstance(x, str): parts.append(x)
            elif isinstance(x, dict): parts.append(str(x.get("text", "")))
        return " ".join(p for p in parts if p).strip()
    return str(c or "").strip()

def _msg_role(m):
    return m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")

def rewrite_query(history, question, llm_choice="ALLaM-7B"):
    """Conversational memory: rewrite a follow-up into a standalone question using recent dialogue."""
    turns = [m for m in (history or []) if _msg_text(m)]
    if not turns:
        return question
    recent = turns[-4:]
    convo = "\n".join((("المستخدم: " if _msg_role(m) == "user" else "المساعد: ") + _msg_text(m)) for m in recent)
    msgs = [
        {"role": "system", "content":
         "أعد صياغة سؤال المستخدم الأخير ليكون سؤالاً مستقلاً مكتمل المعنى بالعربية، وحُلّ أي إشارة "
         "للسياق السابق (مثل: ها، هذا، تمديدها، وماذا عن). إن كان السؤال مستقلاً أصلاً فأعِده كما هو. "
         "أخرج السؤال فقط، سطراً واحداً، بدون أي شرح."},
        {"role": "user", "content": f"المحادثة:\n{convo}\n\nالسؤال الأخير: {question}\n\nالسؤال المستقل:"},
    ]
    out = _complete(msgs, llm_choice, max_new=64).split("\n")[0].strip().strip('"').strip()
    return out if 4 <= len(out) <= 200 else question

def answer(text_q, audio_q, llm_choice, rag_type, emb_choice, voice_label, history):
    # Streaming + multi-turn memory. Normalize incoming chat to clean {role, content:str} dicts.
    history = [{"role": _msg_role(m), "content": _msg_text(m)} for m in (history or []) if _msg_text(m)]
    question = (text_q or "").strip()
    if audio_q:
        question = transcribe(audio_q)
    if not question:
        yield history, None, "اكتب سؤالاً أو سجّل صوتاً.", ""
        return

    # 1) small-talk: friendly reply, no retrieval / memory
    if classify_intent(question) == "small_talk":
        history = history + [{"role": "user", "content": question},
                             {"role": "assistant", "content": ""}]
        info = f"النوع: محادثة عامة (small-talk) | النموذج: {llm_choice}"
        msgs = [{"role": "system", "content": SMALLTALK_SYS}, {"role": "user", "content": question}]
        full = ""
        for partial in stream_messages(msgs, llm_choice, max_new=70):
            full = partial; history[-1]["content"] = full
            yield history, None, info, "_(محادثة عامة)_"
        yield history, speak(full, voice_label), info, "_(محادثة عامة)_"
        return

    # 2) domain: resolve follow-up via memory, then retrieve -> gate -> grounded answer
    standalone = rewrite_query(history, question, llm_choice)
    ctx, rel = retrieve(standalone, rag_type, emb_choice)
    reliable = rel >= RELIABILITY_THRESHOLD
    note = "" if standalone == question else f"\nبعد دمج سياق المحادثة: {standalone}"
    info = (f"السؤال: {question}{note}\n"
            f"النموذج: {llm_choice} | RAG: {rag_type} | Embedding: {emb_choice}\n"
            f"الموثوقية: {rel:.2f} (العتبة {RELIABILITY_THRESHOLD}) -> "
            f"{'موثوق ✓' if reliable else 'غير موثوق — رفض آمن'}")
    history = history + [{"role": "user", "content": question},
                         {"role": "assistant", "content": ""}]
    if not reliable:
        history[-1]["content"] = SAFE_REFUSAL
        yield history, None, info, "_(لم يُستخدم سياق — استرجاع غير موثوق)_"
        yield history, speak(SAFE_REFUSAL, voice_label), info, "_(لم يُستخدم سياق)_"
        return

    ctx_md = "\n\n".join(
        f"**سياق {i} — مادة {c['article']} ({c['source']})**\n\n{c['text'][:400]}…"
        for i, c in enumerate(ctx, 1))
    msgs = build_messages(standalone, ctx, True)
    full = ""
    for partial in stream_messages(msgs, llm_choice, max_new=130):
        full = partial; history[-1]["content"] = full
        yield history, None, info, ctx_md
    yield history, speak(full, voice_label), info, ctx_md

import gradio as gr

with gr.Blocks(title="مساعد نظام العمل السعودي الصوتي", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ المساعد الصوتي لنظام العمل السعودي\n"
                "### AI-Powered Voice Agent for Saudi Labor Law — محادثة نصّية وصوتية بذاكرة")
    with gr.Row():
        with gr.Column(scale=1):
            text_q = gr.Textbox(label="سؤالك (نصّاً)", placeholder="مثال: كم مدة فترة التجربة؟", rtl=True, lines=2)
            audio_q = gr.Audio(label="أو اسأل صوتاً 🎤", sources=["microphone"], type="filepath")
            llm_choice = gr.Dropdown(["ALLaM-7B", "Qwen2.5-7B"], value="ALLaM-7B", label="نموذج اللغة (LLM)")
            rag_type = gr.Dropdown(["Dense", "Hybrid", "Hybrid + Reranker"], value="Hybrid + Reranker", label="نوع الاسترجاع (RAG)")
            emb_choice = gr.Dropdown(["e5-large", "bge-m3"], value="e5-large", label="نموذج التضمين (Embedding)")
            voice_label = gr.Dropdown(["ElevenLabs (Nora)", "رجل (Hamed)", "امرأة (Zariyah)"], value="ElevenLabs (Nora)", label="صوت الإجابة 🔊")
            with gr.Row():
                btn = gr.Button("اسأل 🔎", variant="primary")
                clear = gr.Button("محادثة جديدة 🗑️")
        with gr.Column(scale=2):
            chat = gr.Chatbot(label="المحادثة (بذاكرة متعددة الأدوار)", height=400)
            audio_out = gr.Audio(label="الإجابة صوتاً 🔊", autoplay=False)
            info = gr.Textbox(label="معلومات التشغيل", lines=3)
            ctx_md = gr.Markdown(label="السياق المسترجع")
    btn.click(answer, [text_q, audio_q, llm_choice, rag_type, emb_choice, voice_label, chat],
              [chat, audio_out, info, ctx_md]).then(lambda: (None, None), None, [text_q, audio_q])
    clear.click(lambda: ([], None, "", ""), None, [chat, audio_out, info, ctx_md])
    gr.Examples([["كم مدة فترة التجربة في نظام العمل السعودي؟"],
                 ["وهل يمكن تمديدها؟"],
                 ["ما حقوق العامل عند انتهاء الخدمة؟"]], inputs=text_q)

if __name__ == "__main__":
    print("Launching web interface (with conversational memory) ...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=False)
