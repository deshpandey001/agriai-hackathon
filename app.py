# app.py
# KissanConnect — Streamlit UI with WhatsApp start-message integration (via Twilio)
import os
import time
import csv
import traceback
import base64
from pathlib import Path
from typing import List, Dict
from io import BytesIO

import streamlit as st
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer  # For embeddings (optional)
from streamlit_mic_recorder import mic_recorder  # 🎤 Voice input

import requests
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import google.generativeai as genai  # Gemini (optional)
import whisper   # Whisper for STT

# -------- CONFIG ----------
APP_NAME = "KissanConnect"

KB_DIR = Path("kb")
PROGRAMS_DIR = KB_DIR / "programs"
OUTPUT_DIR = Path("output")
UPLOADS_DIR = OUTPUT_DIR / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

ESCALATION_FILE = OUTPUT_DIR / "escalations.csv"
FEEDBACK_FILE = OUTPUT_DIR / "feedback.csv"
CHAT_HISTORY_FILE = OUTPUT_DIR / "chat_history.csv"

# Cache keys for embeddings
EMBEDDINGS_CACHE_KEY = "kb_embeddings"
KB_DOCS_KEY = "kb_docs"
EMBEDDING_MODEL_KEY = "embedding_model"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Read secrets from environment (server-side)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
TWILIO_SID = os.environ.get("TWILIO_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
TWILIO_FROM = os.environ.get("TWILIO_FROM", "")  # Preferably a WhatsApp-enabled Twilio sender, e.g. "whatsapp:+1415..." or "+1415..."

# default language mapping
LANG_MAP = {
    "English": "english",
    "Hindi": "hindi",
}

# Load Whisper once (cached)
@st.cache_resource
def load_whisper_model():
    # choose model according to your resources: tiny/base/small/medium/large
    try:
        return whisper.load_model("base")
    except Exception:
        return None

# ---------------- Utilities (KB + embeddings) ----------------
def load_kb_texts(kb_dir: Path) -> List[Dict]:
    docs = []
    if not kb_dir.exists():
        return docs
    for p in sorted(kb_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        meta = {"id": p.stem, "text": text, "source": str(p)}
        docs.append(meta)
    return docs

@st.cache_resource
def load_embedding_model(model_name: str):
    try:
        model = SentenceTransformer(model_name)
        st.session_state[EMBEDDING_MODEL_KEY] = model
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model '{model_name}': {e}")
        st.session_state[EMBEDDING_MODEL_KEY] = None
        return None

def embed_texts(texts: List[str]) -> np.ndarray:
    model = st.session_state.get(EMBEDDING_MODEL_KEY)
    if model:
        try:
            embeddings = model.encode(
                texts, convert_to_tensor=False, normalize_embeddings=True
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            st.warning(f"Embedding model failed. Falling back to TF-IDF. ({e})")
            print("Embed error:", traceback.format_exc())

    # TF-IDF fallback
    vectorizer = TfidfVectorizer(max_features=1024, stop_words="english")
    vectorizer.fit(texts)
    st.session_state["tfidf_vectorizer"] = vectorizer
    X = vectorizer.transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    st.session_state["tfidf_dim"] = X.shape[1]
    return X

def build_kb_embeddings_if_not_cached():
    docs = load_kb_texts(KB_DIR)
    st.session_state[KB_DOCS_KEY] = docs
    if not docs:
        st.session_state[EMBEDDINGS_CACHE_KEY] = None
        return
    if EMBEDDINGS_CACHE_KEY not in st.session_state:
        st.session_state[EMBEDDINGS_CACHE_KEY] = None
    if st.session_state[EMBEDDINGS_CACHE_KEY] is not None:
        return
    texts = [d["text"] for d in docs]
    emb_mat = embed_texts(texts)
    st.session_state[EMBEDDINGS_CACHE_KEY] = emb_mat

def retrieve_top_k(query: str, k: int = 3):
    if (
        EMBEDDINGS_CACHE_KEY not in st.session_state
        or st.session_state[EMBEDDINGS_CACHE_KEY] is None
    ):
        return []
    q_emb = None
    model = st.session_state.get(EMBEDDING_MODEL_KEY)
    if model:
        try:
            q_emb = model.encode(
                [query], convert_to_tensor=False, normalize_embeddings=True
            )[0].astype(np.float32)
        except Exception as e:
            st.warning(f"Query embedding failed, using TF-IDF fallback. ({e})")

    if q_emb is None:
        vectorizer = st.session_state.get("tfidf_vectorizer", None)
        if vectorizer is None:
            return []
        q_vec = vectorizer.transform([query]).toarray().astype(np.float32)
        kb_emb = st.session_state[EMBEDDINGS_CACHE_KEY]
        if q_vec.shape[1] != kb_emb.shape[1]:
            if q_vec.shape[1] < kb_emb.shape[1]:
                pad = np.zeros((1, kb_emb.shape[1] - q_vec.shape[1]), dtype=np.float32)
                q_emb = np.hstack([q_vec, pad])[0]
            else:
                q_emb = q_vec[0][: kb_emb.shape[1]]
        else:
            q_emb = q_vec[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    kb_emb = st.session_state[EMBEDDINGS_CACHE_KEY]
    scores = (kb_emb @ q_emb).reshape(-1)
    idxs = np.argsort(-scores)[:k]
    docs = st.session_state.get(KB_DOCS_KEY, [])
    results = []
    for i in idxs:
        if i < len(docs):
            results.append({"doc": docs[i], "score": float(scores[i])})
    return results

# ---------------- Gemini / multimodal helper ----------------
def generate_answer_with_gemini(prompt_text: str, gemini_key: str, image_data_uri: str = None) -> str:
    """
    Generate answer using Gemini (if configured). Falls back to explanatory string if key missing.
    """
    if not gemini_key:
        return "Gemini API key missing on server (set GEMINI_API_KEY environment variable)."

    final_prompt = prompt_text
    if image_data_uri:
        final_prompt += (
            "\n\n[Attached image follows as a base64 data URI — analyze the image for disease, pests, "
            "nutrient deficiency, damage, or other visible issues. Answer in the requested language and cite sources.]\n\n"
        )
        final_prompt += image_data_uri

    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.5-pro")
        try:
            resp = model.generate_content(final_prompt)
            text = getattr(resp, "text", None)
            if text is None:
                if isinstance(resp, dict) and "candidates" in resp and resp["candidates"]:
                    return resp["candidates"][0].get("content", str(resp))
                text = str(resp)
            return text
        except Exception:
            resp2 = genai.generate_text(model="gemini-2.5-pro", text=final_prompt)
            if isinstance(resp2, dict) and "candidates" in resp2:
                return resp2["candidates"][0].get("content", str(resp2))
            return str(resp2)
    except Exception as e:
        return f"Gemini generation failed: {e}\n(Consider checking GEMINI_API_KEY or SDK compatibility.)"

# ---------------- Weather interpretation ----------------
def fetch_weather_raw(location: str, openweather_api_key: str):
    if not location or not openweather_api_key:
        return None
    try:
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": location, "appid": openweather_api_key, "units": "metric"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("fetch_weather_raw error:", e)
        return None

def interpret_weather_conditions(weather_json: dict) -> Dict:
    if not weather_json:
        return {}
    main = weather_json.get("main", {})
    wind = weather_json.get("wind", {})
    weather = weather_json.get("weather", [{}])[0]

    temp = main.get("temp")
    feels_like = main.get("feels_like")
    humidity = main.get("humidity")
    wind_speed = wind.get("speed", 0.0)
    weather_main = weather.get("main", "")
    weather_desc = weather.get("description", "")

    rain_1h = weather_json.get("rain", {}).get("1h", 0.0)
    snow_1h = weather_json.get("snow", {}).get("1h", 0.0)

    triggers = []
    if temp is not None and temp >= 35.0:
        triggers.append("heat")
    if temp is not None and temp <= 2.0:
        triggers.append("frost")
    if rain_1h and rain_1h >= 10.0:
        triggers.append("heavy_rain")
    if snow_1h and snow_1h >= 5.0:
        triggers.append("heavy_snow")
    if wind_speed and wind_speed >= 10.0:
        triggers.append("high_wind")
    if weather_main and weather_main.lower() in ["thunderstorm", "tornado", "squall"]:
        triggers.append("severe_weather")
    if weather_desc and "storm" in weather_desc.lower():
        triggers.append("severe_weather")

    return {
        "temp": temp,
        "feels_like": feels_like,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "weather_main": weather_main,
        "weather_desc": weather_desc,
        "triggers": sorted(set(triggers)),
    }

def build_weather_message(location: str, interp: Dict, lang: str = "English") -> str:
    if not interp:
        if lang == "Hindi":
            return f"{location} के लिए मौसम डेटा उपलब्ध नहीं है।"
        return f"Weather data for {location} is unavailable."

    if lang == "Hindi":
        parts = []
        if interp.get("triggers"):
            parts.append("अलर्ट: " + ", ".join(interp["triggers"]).replace("_", " "))
        parts.append(f"स्थान: {location}")
        if interp.get("temp") is not None:
            parts.append(f"तापमान: {interp['temp']}°C (अनुभव: {interp.get('feels_like')})")
        if interp.get("humidity") is not None:
            parts.append(f"आर्द्रता: {interp['humidity']}%")
        if interp.get("rain_1h"):
            parts.append(f"बारिश (1h): {interp['rain_1h']} mm")
        if interp.get("wind_speed"):
            parts.append(f"हवा: {interp['wind_speed']} m/s")
        parts.append(f"विवरण: {interp.get('weather_desc','')}")
        parts.append("कृषि सुझाव: गंभीर समस्या होने पर नजदीकी कृषिभवन से संपर्क करें।")
        return " | ".join(parts)
    else:
        parts = []
        if interp.get("triggers"):
            parts.append(f"ALERT: {', '.join(interp['triggers']).replace('_',' ')}")
        parts.append(f"Location: {location}")
        if interp.get("temp") is not None:
            parts.append(f"Temp: {interp['temp']}°C (Feels {interp.get('feels_like')})")
        if interp.get("humidity") is not None:
            parts.append(f"Humidity: {interp['humidity']}%")
        if interp.get("rain_1h"):
            parts.append(f"Rain (1h): {interp['rain_1h']} mm")
        if interp.get("wind_speed"):
            parts.append(f"Wind: {interp['wind_speed']} m/s")
        parts.append(f"Description: {interp.get('weather_desc','')}")
        parts.append("Advice: If serious issues, contact local Krishibhavan.")
        return " | ".join(parts)

# ---------------- Twilio WhatsApp helper ----------------
from twilio.rest import Client

def send_whatsapp_via_twilio(twilio_sid, twilio_token, from_whatsapp, to_number, body_text):
    """
    from_whatsapp: 'whatsapp:+14155238886' (Twilio sandbox sender or your Twilio WA number)
    to_number: '+91xxxxxxxxxx' (E.164) -- library will send as 'whatsapp:+91...'
    """
    if not (twilio_sid and twilio_token and from_whatsapp and to_number):
        return False, "Missing Twilio credentials or numbers."

    try:
        client = Client(twilio_sid, twilio_token)
        message = client.messages.create(
            body=body_text,
            from_=from_whatsapp,
            to=f"whatsapp:{to_number}"
        )
        return True, {"sid": message.sid, "status": message.status}
    except Exception as e:
        return False, str(e)


# ---------------- small helpers ----------------
def ensure_csv_has_header(path: Path, header: List[str]):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

def load_program_guides_for_lang(lang: str) -> List[Dict]:
    guides = []
    if not PROGRAMS_DIR.exists():
        return guides
    for p in sorted(PROGRAMS_DIR.glob("*.txt")):
        name = p.name
        if lang == "Hindi":
            if not name.endswith(".hi.txt"):
                continue
            base_id = p.stem
            if base_id.endswith(".hi"):
                base_id = base_id[: -3]
            text = p.read_text(encoding="utf-8")
            guides.append({"id": base_id, "text": text, "source": str(p)})
        else:
            if name.endswith(".hi.txt"):
                continue
            base_id = p.stem
            text = p.read_text(encoding="utf-8")
            guides.append({"id": base_id, "text": text, "source": str(p)})
    return guides

def classify_programs(guides: List[Dict]) -> Dict[str, List[Dict]]:
    mapping = {
        "Loans": [],
        "Insurance": [],
        "Mechanization": [],
        "Policies & Subsidies": [],
        "Other": [],
    }
    for g in guides:
        txt = (g["id"] + " " + g["text"]).lower()
        if any(k in txt for k in ["loan", "kcc", "credit", "interest", "finance", "kisan"]):
            mapping["Loans"].append(g)
        elif any(k in txt for k in ["insurance", "pmfby", "fasal", "bima", "premium", "claim"]):
            mapping["Insurance"].append(g)
        elif any(k in txt for k in ["mechaniz", "smam", "machine", "tractor", "custom hiring", "chc"]):
            mapping["Mechanization"].append(g)
        elif any(k in txt for k in ["policy", "subsid", "pm-kisan", "pmkisan", "scheme", "support", "subvention", "grant", "subsidy"]):
            mapping["Policies & Subsidies"].append(g)
        else:
            mapping["Other"].append(g)
    return mapping

def prepare_context_text(guides: List[Dict], max_chars_per_file: int = 3500) -> str:
    parts = []
    for g in guides:
        text = g["text"]
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + "\n\n[TRUNCATED]"
        parts.append(f"[{g['id']}]\n" + text)
    return "\n\n".join(parts)

# ---------------- UI -----------------
st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")

# CSS (kept similar)
st.markdown(
    """
    <style>
    .app-header { display:flex; align-items:center; gap:12px; }
    .app-title { font-size:28px; font-weight:700; color:#0b6b3a; }
    .app-sub { color:#2b6e49; margin-top: -6px; }
    .card { background: #ffffff; border-radius:12px; padding:16px; box-shadow: 0 6px 20px rgba(10,20,12,0.06); }
    .big-btn { background:#0b6b3a; color:white; padding:10px 18px; border-radius:8px; font-weight:600; }
    .muted { color:#6b7280; }
    .small { font-size:13px; color:#6b7280; }
    pre { white-space: pre-wrap; }
    </style>
    """,
    unsafe_allow_html=True,
)

# header
st.markdown(
    f"""
    <div class="app-header">
      <div style="font-size:34px">🌱</div>
      <div>
        <div class="app-title">{APP_NAME}</div>
        <div class="app-sub small">Digital Krishi Officer — Voice/Text/Image advisory</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: farmer/context + language selector
st.sidebar.header("Farmer context")
name = st.sidebar.text_input("Name", value="")
phone = st.sidebar.text_input("Phone (E.164)", value="")  # expects +91...
location = st.sidebar.text_input("Location (city or district)", value="")
crop = st.sidebar.selectbox("Crop", options=["banana", "rice", "coconut", "tapioca", "vegetable", "other"])

st.sidebar.markdown("---")
lang = st.sidebar.radio("Language / भाषा", options=["English", "Hindi"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("🔐 API keys are stored on the server (environment variables).")

# Show API key presence status (no raw secrets)
col_api1, col_api2, col_api3 = st.columns(3)
with col_api1:
    if GEMINI_API_KEY:
        st.success("Gemini: configured")
    else:
        st.error("Gemini: NOT configured")
with col_api2:
    if OPENWEATHER_API_KEY:
        st.success("OpenWeather: configured")
    else:
        st.info("OpenWeather: not configured")
with col_api3:
    if TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM:
        st.success("Twilio: configured")
    else:
        st.info("Twilio: not configured")

st.write("")

# Build KB embeddings (non-blocking if model missing)
if "kb_built" not in st.session_state:
    try:
        load_embedding_model(EMBEDDING_MODEL_NAME)
    except Exception:
        pass
    build_kb_embeddings_if_not_cached()
    st.session_state["kb_built"] = True

# Top-level tabs
tab_advisory, tab_chatbot, tab_programs, tab_calendar, tab_fertcalc, tab_admin = st.tabs(
    ["Advisory", "Chatbot", "Programs", "Crop Calendar", "Fertilizer Calculator", "Admin / Weather"]
)

# ---------------- Advisory tab ----------------
with tab_advisory:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Ask (text / voice / image)")
    st.markdown("Type your question. You can also record voice or upload a leaf image for analysis.")

    if "pending_user_query" not in st.session_state:
        st.session_state["pending_user_query"] = ""
    audio = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹ Stop",
        just_once=True,
        use_container_width=True,
        key="recorder2",
    )
    if audio and "bytes" in audio:
        whisper_model = load_whisper_model()
        if whisper_model:
            tmp_wav = "temp_audio.wav"
            with open(tmp_wav, "wb") as f:
                f.write(audio["bytes"])
            result = whisper_model.transcribe(tmp_wav)
            st.session_state["pending_user_query"] = result.get("text", "")
            st.success(f"🎤 Transcribed: {st.session_state['pending_user_query']}")
        else:
            st.info("Whisper model not available - install whisper in the venv if you want voice transcription.")

    q_text = st.text_area("Your question (editable)", height=120, value=st.session_state.get("pending_user_query", ""))

    st.markdown("**Upload image (optional)** — clear, focused close-up of affected leaf/plant works best.")
    img_file = st.file_uploader("Upload plant/leaf image (optional)", type=["jpg", "jpeg", "png"])
    pil_img = None
    image_data_uri = None
    if img_file:
        pil_img = Image.open(img_file).convert("RGB")
        st.image(pil_img, caption="Uploaded image", use_container_width=True)
        try:
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=75)
            b = buffer.getvalue()
            if len(b) > 200 * 1024:
                w, h = pil_img.size
                new_w = 800
                new_h = int(new_w * (h / w))
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                buffer = BytesIO()
                pil_img.save(buffer, format="JPEG", quality=70)
                b = buffer.getvalue()
            image_data_uri = "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")
            fname = f"{int(time.time())}_{os.urandom(4).hex()}.jpg"
            with open(UPLOADS_DIR / fname, "wb") as out_f:
                out_f.write(b)
        except Exception as e:
            st.error(f"Failed to encode image: {e}")
            image_data_uri = None

    get_advice_btn = st.button("Get Advice", key="get_advice_btn")

    if get_advice_btn:
        if not q_text and not image_data_uri:
            st.warning("Please type your question, record voice, or upload an image.")
        else:
            with st.spinner("Retrieving KB passages & generating answer..."):
                retrieved = retrieve_top_k(q_text, k=3) if q_text else []
                passages_text = "\n\n".join([f"[{r['doc']['id']}] {r['doc']['text'][:500]}..." for r in retrieved])

                system_prompt = (
                    "You are KrishiAdviser — a concise, practical agricultural assistant. "
                    "Use the Sources below for factual guidance. If an image is attached, analyze it carefully:\n"
                    "- Identify visible symptoms (disease, pest, nutrient deficiency, mechanical damage).\n"
                    "- Give a short diagnosis (with uncertainty if low confidence).\n"
                    "- Provide up to 3 actionable steps (safe, local, low-cost first).\n"
                    "- If unsure about pesticide dosage or safety-critical instructions, instruct to escalate to the local Krishibhavan.\n"
                    "Cite any sources from the 'Sources' list inside square brackets.\n"
                )

                if lang == "Hindi":
                    system_prompt += "Respond in Hindi. Be concise and use local terms if relevant.\n"
                else:
                    system_prompt += "Respond in English.\n"

                context = f"Location: {location}\nCrop: {crop}\nFarmer name: {name}\n"

                prompt_parts = [
                    "SYSTEM:\n" + system_prompt,
                    "CONTEXT:\n" + context,
                ]
                if q_text:
                    prompt_parts.append("USER QUERY:\n" + q_text)
                if passages_text:
                    prompt_parts.append("SOURCES:\n" + passages_text)
                prompt_text = "\n\n".join(prompt_parts)

                answer = generate_answer_with_gemini(prompt_text, GEMINI_API_KEY, image_data_uri)

            advice_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "name": name,
                "phone": phone,
                "location": location,
                "crop": crop,
                "query": q_text,
                "answer": answer,
                "language": lang,
                "image_saved": bool(image_data_uri),
            }
            if "advice_history" not in st.session_state:
                st.session_state["advice_history"] = []
            st.session_state["advice_history"].insert(0, advice_entry)

            st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
            st.subheader("Answer")
            st.markdown(answer)
            if retrieved:
                st.markdown("**Retrieved sources:**")
                for r in retrieved:
                    st.markdown(f"- `{r['doc']['id']}` (score: {r['score']:.3f}) — {r['doc']['source']}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            if st.checkbox("Escalate this query (save for Krishibhavan)"):
                payload = {
                    "timestamp": advice_entry["timestamp"],
                    "name": name,
                    "phone": phone,
                    "location": location,
                    "crop": crop,
                    "query": q_text,
                    "image_saved": bool(image_data_uri),
                }
                header = list(payload.keys())
                ensure_csv_has_header(ESCALATION_FILE, header)
                with open(ESCALATION_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writerow(payload)
                st.success("Escalation saved.")

            st.markdown("**Feedback**")
            fb = st.radio("Was this answer helpful? / क्या यह उत्तर मददगार था?", ["Select", "Yes", "Partial", "No"], index=0)
            if fb != "Select" and st.button("Submit feedback"):
                fb_record = {
                    "timestamp": advice_entry["timestamp"],
                    "name": name,
                    "phone": phone,
                    "location": location,
                    "crop": crop,
                    "query": q_text,
                    "answer_excerpt": answer[:500].replace('\n',' '),
                    "feedback": fb,
                    "language": lang,
                }
                header = list(fb_record.keys())
                ensure_csv_has_header(FEEDBACK_FILE, header)
                with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writerow(fb_record)
                st.success("Thanks — feedback saved.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Chatbot tab ----------------
with tab_chatbot:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Chatbot — continue conversation based on previous advice")
    st.markdown("Select an item from the advice history to use as context or start a fresh chat.")

    history = st.session_state.get("advice_history", [])
    choices = [f"[{i}] {h['timestamp']} — {h.get('query','(no query)')[:60]}" for i,h in enumerate(history)]
    choices.insert(0, "Start new chat")
    selected = st.selectbox("Choose context", options=choices, index=0)

    base_context = None
    if selected != "Start new chat":
        idx = int(selected.split(']')[0].strip('['))
        base_context = history[idx]
        st.markdown("**Context excerpt:**")
        st.markdown(base_context.get("answer",""))

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    with st.form(key="chat_form"):
        user_msg = st.text_input("Your message")
        submit_chat = st.form_submit_button("Send")

    if submit_chat and user_msg:
        system_prompt = "You are KrishiAdviser Chatbot. Keep answers short and practical. "
        if base_context:
            system_prompt += "Use the following previous advice as context for the user:\n"
            system_prompt += base_context.get("answer","") + "\n"
        if lang == "Hindi":
            system_prompt += "Respond in Hindi.\n"
        else:
            system_prompt += "Respond in English.\n"

        prompt = "SYSTEM:\n" + system_prompt + "\nUSER:\n" + user_msg
        resp = generate_answer_with_gemini(prompt, GEMINI_API_KEY)

        st.session_state["chat_messages"].append({"role":"user","text":user_msg, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        st.session_state["chat_messages"].append({"role":"assistant","text":resp, "time": time.strftime("%Y-%m-%d %H:%M:%S")})

        chat_row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "context_timestamp": base_context["timestamp"] if base_context else "",
            "user": name,
            "phone": phone,
            "location": location,
            "crop": crop,
            "language": lang,
            "user_msg": user_msg,
            "assistant_msg": resp[:2000],
        }
        header = list(chat_row.keys())
        ensure_csv_has_header(CHAT_HISTORY_FILE, header)
        with open(CHAT_HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(chat_row)

    for m in st.session_state.get("chat_messages", [])[-40:]:
        if m["role"] == "user":
            st.markdown(f"**You ({m['time']}):** {m['text']}")
        else:
            st.markdown(f"**Assistant ({m['time']}):** {m['text']}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Programs tab (language-aware KB files) ----------------
with tab_programs:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Government Programs / Loans / Insurance / Mechanization Guides")
    st.markdown("Select a category and program documents in the chosen language; the AI will answer questions using those guides as context.")

    guides = load_program_guides_for_lang(lang)
    if not guides:
        st.info(
            "No program guides found for the selected language.\n\n"
            "- For English: place files as `kb/programs/<name>.txt` (e.g., kcc.txt, pmfby.txt)\n"
            "- For Hindi: place files as `kb/programs/<name>.hi.txt` (e.g., kcc.hi.txt, pmfby.hi.txt)\n"
        )
    else:
        cat_map = classify_programs(guides)
        categories = ["Loans", "Insurance", "Mechanization", "Policies & Subsidies", "Other"]
        selected_category = st.selectbox("Choose category", options=categories, index=0)

        available_guides = cat_map.get(selected_category, [])
        if not available_guides:
            st.info(f"No guides found in category '{selected_category}' for language {lang}.")
        else:
            guide_labels = [f"{g['id']}" for g in available_guides]
            selected_files = st.multiselect("Choose program document(s) to use as context", options=guide_labels, default=guide_labels[:1])
            selected_guides = [g for g in available_guides if g['id'] in selected_files]

            with st.expander("Preview selected documents", expanded=False):
                for g in selected_guides:
                    st.markdown(f"**{g['id']}** — {g['source']}")
                    st.text(g['text'][:1500] + ("..." if len(g['text']) > 1500 else ""))

            st.markdown("---")
            st.markdown("Ask a question (the selected program docs will be provided as sources to the model).")
            user_question = st.text_area("Your question about the selected programs", height=140, placeholder="E.g., What are the subsidy rates and how to apply for mechanization in my state?")

            col1, col2 = st.columns([1,1])
            with col1:
                ask_btn = st.button("Ask AI about selected programs")
            with col2:
                summarize_btn = st.button("Get short summary of selected programs")

            if ask_btn or summarize_btn:
                if not selected_guides:
                    st.warning("Please select at least one program document to use as context.")
                else:
                    context_text = prepare_context_text(selected_guides, max_chars_per_file=3500)

                    sys_prompt = "You are KrishiAdviser — use the provided program guides to answer the user's question. Give concise, practical steps and cite the document IDs in square brackets when referring to specific guidance."
                    if lang == "Hindi":
                        sys_prompt += " Respond in Hindi and use clear simple terms."
                    else:
                        sys_prompt += " Respond in English."

                    if summarize_btn:
                        user_q = "Please provide a short actionable summary of the selected program documents (3-6 bullet points)."
                    else:
                        user_q = user_question or "Please summarize the selected program documents and list the key actionable steps for a farmer."

                    prompt_parts = [
                        "SYSTEM:\n" + sys_prompt,
                        "SOURCES:\n" + context_text,
                        "USER QUERY:\n" + user_q
                    ]
                    prompt_text = "\n\n".join(prompt_parts)

                    with st.spinner("Asking the model (using selected program guides)..."):
                        answer = generate_answer_with_gemini(prompt_text, GEMINI_API_KEY)

                    st.markdown("**AI answer (based on selected programs):**")
                    st.markdown(answer)

                    if st.checkbox("Save this program-question to chat history (for admin)"):
                        chat_row = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "context_docs": ",".join([g["id"] for g in selected_guides]),
                            "language": lang,
                            "user": name,
                            "phone": phone,
                            "location": location,
                            "crop": crop,
                            "user_question": user_q,
                            "assistant_msg": answer[:2000],
                        }
                        header = list(chat_row.keys())
                        ensure_csv_has_header(CHAT_HISTORY_FILE, header)
                        with open(CHAT_HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=header)
                            writer.writerow(chat_row)
                        st.success("Saved to chat history CSV.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Crop Calendar tab ----------------
with tab_calendar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Crop Calendar (Kharif / Rabi)")
    st.markdown("A simple table showing common crops and the typical Kharif / Rabi seasons. Edit or extend `kb/crop_calendar.csv` if you want to persist or upload a different file.")

    # Default minimal calendar; you can replace this by reading kb/crop_calendar.csv
    calendar = [
        {"crop": "Rice", "kharif": "June - October", "rabi": "Nov - Feb (less common)"},
        {"crop": "Wheat", "kharif": "-", "rabi": "Nov - Apr"},
        {"crop": "Maize", "kharif": "June - Sep", "rabi": "Oct - Feb"},
        {"crop": "Millets", "kharif": "Jun - Sep", "rabi": "Oct - Feb"},
        {"crop": "Banana", "kharif": "Year-round", "rabi": "Year-round"},
        {"crop": "Tapioca", "kharif": "Year-round", "rabi": "Year-round"},
    ]

    # Show as table
    st.table(calendar)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Fertilizer Calculator tab ----------------
with tab_fertcalc:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Fertilizer Calculator")
    st.markdown("Select a crop, enter field area (in cents or hectares) and get a simple recommended N:P:K estimate. This is a basic helper — use local extension advice for precise rates.")

    # area units
    area_unit = st.selectbox("Area unit", ["hectares", "acres", "cents"])
    area_input = st.number_input("Area (numeric)", min_value=0.0, value=0.1, step=0.1)
    sel_crop = st.selectbox("Crop for fertilizer estimate", ["rice", "wheat", "maize", "banana", "vegetable", "other"])

    # crude per-hectare baseline NPK (example values; adapt to local recommendations)
    npk_defaults = {
        "rice": (120, 60, 40),
        "wheat": (100, 50, 40),
        "maize": (150, 60, 40),
        "banana": (250, 200, 250),
        "vegetable": (120, 80, 60),
        "other": (100, 50, 40)
    }
    base_n, base_p, base_k = npk_defaults.get(sel_crop, npk_defaults["other"])

    # convert area to hectares
    if area_unit == "hectares":
        hectares = area_input
    elif area_unit == "acres":
        hectares = area_input * 0.404686
    else:  # cents (common in India: 1 cent = 40.4686 m2 = 0.0404686 ha)
        hectares = area_input * 0.00404686

    if st.button("Calculate fertilizer requirement"):
        total_n = base_n * hectares
        total_p = base_p * hectares
        total_k = base_k * hectares
        st.write(f"Estimated fertilizer requirement for {area_input} {area_unit} ({hectares:.3f} ha):")
        st.write(f"- Nitrogen (N): {total_n:.1f} kg")
        st.write(f"- Phosphorus (P₂O₅ equiv): {total_p:.1f} kg")
        st.write(f"- Potassium (K₂O equiv): {total_k:.1f} kg")
        st.markdown("**Note:** These are approximate values. For exact doses, soil tests and local extension recommendations are recommended.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Admin / Weather tab ----------------
with tab_admin:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Weather & WhatsApp Start Chat")
    st.markdown("Enter Location and Phone in the sidebar. Use the buttons below to fetch weather and optionally start a WhatsApp chat with the farmer.")

    if not location or not phone:
        st.info("Set Location and Phone in the sidebar to use weather fetch and WhatsApp features.")
    else:
        if st.button("Fetch weather now"):
            raw = fetch_weather_raw(location, OPENWEATHER_API_KEY)
            if not raw:
                st.error("Failed to fetch weather. Check OpenWeather API key or location.")
            else:
                interp = interpret_weather_conditions(raw)
                msg_text = build_weather_message(location, interp, lang=lang)
                st.markdown("**Latest weather summary:**")
                st.write(msg_text)

                # Show a button to start a WhatsApp chat (send initial weather message)
                st.write("")
                if st.button("Send WhatsApp start message"):
                    tw_from = os.environ.get("TWILIO_FROM", "whatsapp:+14155238886")  # or set your env
                    ok, info = send_whatsapp_via_twilio(TWILIO_SID, TWILIO_TOKEN, tw_from, phone, msg_text)
                    if ok:
                        st.success(f"WhatsApp message sent — SID: {info.get('sid')}")
                    else:
                        st.error(f"Failed to send WhatsApp message: {info}")


                # Also allow sending via SMS fallback (existing function)
                if st.button("Send same weather as SMS (SMS fallback)"):
                    ok2, info2 = send_whatsapp_via_twilio(TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, phone, msg_text)
                    # Note: this uses WhatsApp send. If you prefer SMS, implement send_sms via client.messages.create without whatsapp:
                    if ok2:
                        st.success(f"📩 Message sent to {phone} (via Twilio).")
                    else:
                        st.error(f"❌ Failed: {info2}")

    st.markdown("---")
    st.subheader("Admin / Debug")
    st.markdown("Recent escalations & feedback are saved on the server (CSV).")
    if ESCALATION_FILE.exists() and st.button("Show recent escalations"):
        rows = []
        with open(ESCALATION_FILE, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                if i > 50:
                    break
                rows.append(r)
        st.write(rows)

    if FEEDBACK_FILE.exists() and st.button("Show recent feedback"):
        rows = []
        with open(FEEDBACK_FILE, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                if i > 50:
                    break
                rows.append(r)
        st.write(rows)

    if CHAT_HISTORY_FILE.exists() and st.button("Show recent chat history"):
        rows = []
        with open(CHAT_HISTORY_FILE, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                if i > 100:
                    break
                rows.append(r)
        st.write(rows)

    st.markdown("---")
    st.markdown("<div class='small muted'>Server-side notes:</div>", unsafe_allow_html=True)
    st.markdown(
        "- API keys are read from environment variables (GEMINI_API_KEY, OPENWEATHER_API_KEY, TWILIO_*).\n"
        "- Uploaded images saved to `output/uploads/`.\n"
        "- For English KB files: place `kb/programs/<name>.txt` (e.g., kcc.txt).\n"
        "- For Hindi KB files: place `kb/programs/<name>.hi.txt` (e.g., kcc.hi.txt).\n"
        "- Program guides are classified automatically but you can control categories by changing filenames or content.\n"
        "- To receive farmer replies on WhatsApp, host your `whatsapp.py` Flask webhook and configure Twilio Messaging webhook to point at it.\n"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.caption(
    "KissanConnect prototype — multimodal advice. For production, move uploads to S3, secure admin, and implement SDK-native multimodal attachments for Gemini."
)
