# app_kissanconnect_updated.py
# Full updated KissanConnect prototype
# Programs tab reads language-specific KB files (*.txt for English, *.hi.txt for Hindi)
# Added Crop Calendar tab and Fertilizer Calculator tab

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

from sentence_transformers import SentenceTransformer  # For embeddings
from streamlit_mic_recorder import mic_recorder  # 🎤 Voice input

import requests
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import google.generativeai as genai  # Gemini
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
TWILIO_FROM = os.environ.get("TWILIO_FROM", "")

# default language mapping
LANG_MAP = {
    "English": "english",
    "Hindi": "hindi",
}

# Load Whisper once (cached)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

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


def send_weather_sms_verbose(twilio_sid, twilio_token, from_num, to_num, msg):
    if not (twilio_sid and twilio_token and from_num and to_num):
        return False, "Missing Twilio credentials or numbers."
    try:
        client = Client(twilio_sid, twilio_token)
        message = client.messages.create(body=msg, from_=from_num, to=to_num)
        info = {
            "sid": getattr(message, "sid", None),
            "status": getattr(message, "status", None),
            "to": getattr(message, "to", None),
            "from": getattr(message, "from_", None),
            "price": getattr(message, "price", None),
            "error_code": getattr(message, "error_code", None),
            "error_message": getattr(message, "error_message", None),
        }
        st.info(f"Twilio response: SID={info['sid']} status={info['status']}")
        print("Twilio response:", info)
        return True, info
    except TwilioRestException as tre:
        st.error(f"TwilioRestException: {tre.msg} (code {tre.code})")
        print("TwilioRestException:", tre.code, tre.msg)
        return False, {"code": tre.code, "msg": tre.msg}
    except Exception as e:
        st.error(f"Twilio send failed: {e}")
        print("Twilio send exception:", traceback.format_exc())
        return False, str(e)

# ---------------- small helpers ----------------
def ensure_csv_has_header(path: Path, header: List[str]):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()


def load_program_guides_for_lang(lang: str) -> List[Dict]:
    """
    Load only language-appropriate program files from PROGRAMS_DIR.
    - English: load *.txt excluding files that end with .hi.txt
    - Hindi: load files that end with .hi.txt
    Returns list of dicts with cleaned 'id' (strip .hi if present), 'text', and 'source'.
    """
    guides = []
    if not PROGRAMS_DIR.exists():
        return guides

    for p in sorted(PROGRAMS_DIR.glob("*.txt")):
        name = p.name
        if lang == "Hindi":
            # include only files with .hi.txt
            if not name.endswith(".hi.txt"):
                continue
            base_id = p.stem
            # p.stem for 'kcc.hi.txt' -> 'kcc.hi' -> strip trailing '.hi'
            if base_id.endswith(".hi"):
                base_id = base_id[: -3]
            text = p.read_text(encoding="utf-8")
            guides.append({"id": base_id, "text": text, "source": str(p)})
        else:
            # English: include only files that DO NOT end with .hi.txt
            if name.endswith(".hi.txt"):
                continue
            base_id = p.stem  # e.g., 'kcc'
            text = p.read_text(encoding="utf-8")
            guides.append({"id": base_id, "text": text, "source": str(p)})
    return guides


def classify_programs(guides: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Simple keyword-based classification into categories.
    Returns mapping: category -> list of guide dicts
    """
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

# helper to safely prepare context (truncate)
def prepare_context_text(guides: List[Dict], max_chars_per_file: int = 3500) -> str:
    parts = []
    for g in guides:
        text = g["text"]
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + "\n\n[TRUNCATED]"
        parts.append(f"[{g['id']}]\n" + text)
    return "\n\n".join(parts)

# ---------------- NEW: Crop Calendar data ----------------
CROP_CALENDAR = [
    {"crop": "rice", "kharif": "June - September", "rabi": "October - March (some varieties)"},
    {"crop": "maize", "kharif": "June - September", "rabi": "Oct - Feb (limited)"},
    {"crop": "millets", "kharif": "June - Sept", "rabi": "Nov - Feb (minor)"},
    {"crop": "banana", "kharif": "Year-round (planting mostly Jan-Mar & Aug-Oct)", "rabi": "—"},
    {"crop": "coconut", "kharif": "Year-round planting", "rabi": "—"},
    {"crop": "tapioca", "kharif": "Mar-Jun (planting) / Harvest varies", "rabi": "—"},
    {"crop": "vegetable", "kharif": "June - Oct (many crops)", "rabi": "Nov - Mar (many crops)"},
    {"crop": "sugarcane", "kharif": "Feb - May (planting)", "rabi": "—"},
    {"crop": "pulses", "kharif": "June - Sept (pigeon pea)", "rabi": "Oct - Feb (lentils, gram)"},
]

# ---------------- NEW: Fertilizer calc defaults & helpers ----------------
# Typical (indicative) N-P2O5-K2O recommendations (kg per hectare) — these are indicative ONLY.
# Please verify with local soil test / extension.
DEFAULT_NPK_PER_HA = {
    "rice": {"N": 120, "P2O5": 60, "K2O": 40},
    "banana": {"N": 400, "P2O5": 200, "K2O": 500},
    "coconut": {"N": 250, "P2O5": 150, "K2O": 350},
    "tapioca": {"N": 80, "P2O5": 60, "K2O": 80},
    "vegetable": {"N": 150, "P2O5": 75, "K2O": 75},
    "other": {"N": 100, "P2O5": 50, "K2O": 50},
}

# Fertilizer nutrient contents (fraction)
FERT_CONTENT = {
    "urea": {"N": 0.46, "P2O5": 0.0, "K2O": 0.0},
    "dap":  {"N": 0.18, "P2O5": 0.46, "K2O": 0.0},
    "mop":  {"N": 0.0,  "P2O5": 0.0,  "K2O": 0.6},
}

# Unit conversion helpers (to hectares)
def area_to_hectares(value: float, unit: str) -> float:
    """
    Supported units: 'hectare', 'acre', 'cent', 'sq_m', 'sq_ft'
    Note: local units like 'bigha' vary by state — ask user to provide hectares for accuracy.
    """
    unit = unit.lower()
    if unit == "hectare":
        return value
    if unit == "acre":
        return value * 0.404685642  # 1 acre = 0.404685642 ha
    if unit == "sq_m" or unit == "sqm":
        return value / 10000.0
    if unit == "sq_ft" or unit == "sqft":
        return value * 0.092903 / 10000.0
    if unit == "cent":
        # 1 cent = 435.6 sq ft ~ 40.4686 sq m -> 0.00404686 ha
        return value * 0.00404685642
    # default: return value (assume hectare) but show a warning later
    return value

def compute_nutrient_requirements(ha: float, npk_per_ha: Dict[str, float]):
    return {k: round(v * ha, 2) for k, v in npk_per_ha.items()}

def compute_fertilizer_bags(nutrient_need: Dict[str, float], bag_weights: Dict[str, float], fert_content: Dict[str, Dict[str, float]]):
    """
    Returns suggested kg of each fertilizer to supply required nutrients.
    Uses simple linear algebra: choose primary fertilizers: urea (N), DAP (P2O5 & some N), MOP (K2O).
    We'll calculate order: supply P from DAP first, K from MOP, then remaining N from UREA (accounting for N in DAP).
    """
    N_need = nutrient_need.get("N", 0.0)
    P_need = nutrient_need.get("P2O5", 0.0)
    K_need = nutrient_need.get("K2O", 0.0)

    # DAP to supply P2O5 (and also provides some N)
    dap_needed_kg = 0.0
    if fert_content["dap"]["P2O5"] > 0:
        dap_needed_kg = P_need / fert_content["dap"]["P2O5"]
    # N obtained from DAP
    N_from_dap = dap_needed_kg * fert_content["dap"]["N"]

    # MOP to supply K2O
    mop_needed_kg = 0.0
    if fert_content["mop"]["K2O"] > 0:
        mop_needed_kg = K_need / fert_content["mop"]["K2O"]

    # Remaining N requirement after DAP contribution
    N_remaining = max(0.0, N_need - N_from_dap)
    urea_needed_kg = 0.0
    if fert_content["urea"]["N"] > 0:
        urea_needed_kg = N_remaining / fert_content["urea"]["N"]

    # Convert to bag counts using bag_weights (kg per bag), default 50 kg if not provided
    result = {
        "DAP_kg": round(dap_needed_kg, 2),
        "DAP_bags": round(dap_needed_kg / bag_weights.get("dap", 50.0), 2),
        "MOP_kg": round(mop_needed_kg, 2),
        "MOP_bags": round(mop_needed_kg / bag_weights.get("mop", 50.0), 2),
        "Urea_kg": round(urea_needed_kg, 2),
        "Urea_bags": round(urea_needed_kg / bag_weights.get("urea", 50.0), 2),
    }
    return result

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
    table, th, td { border-collapse: collapse; padding:6px; }
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
phone = st.sidebar.text_input("Phone (E.164)", value="")
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

# Top-level tabs (added Crop Calendar & Fertilizer)
tab_advisory, tab_chatbot, tab_programs, tab_cropcal, tab_fertcalc, tab_admin = st.tabs(
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
        tmp_wav = "temp_audio.wav"
        with open(tmp_wav, "wb") as f:
            f.write(audio["bytes"])
        result = whisper_model.transcribe(tmp_wav)
        st.session_state["pending_user_query"] = result.get("text", "")
        st.success(f"🎤 Transcribed: {st.session_state['pending_user_query']}")

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

                # language handling in prompt
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

            # Save advice into session history
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

            # Display answer + sources + actions
            st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
            st.subheader("Answer")
            st.markdown(answer)
            if retrieved:
                st.markdown("**Retrieved sources:**")
                for r in retrieved:
                    st.markdown(f"- `{r['doc']['id']}` (score: {r['score']:.3f}) — {r['doc']['source']}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Escalate / feedback UI (save to CSV)
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

    # Chat message area
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    with st.form(key="chat_form"):
        user_msg = st.text_input("Your message")
        submit_chat = st.form_submit_button("Send")

    if submit_chat and user_msg:
        # build prompt including selected advice if any
        system_prompt = (
            "You are KrishiAdviser Chatbot. Keep answers short and practical. "
        )
        if base_context:
            system_prompt += "Use the following previous advice as context for the user:\n"
            system_prompt += base_context.get("answer","") + "\n"
        if lang == "Hindi":
            system_prompt += "Respond in Hindi.\n"
        else:
            system_prompt += "Respond in English.\n"

        prompt = "SYSTEM:\n" + system_prompt + "\nUSER:\n" + user_msg
        resp = generate_answer_with_gemini(prompt, GEMINI_API_KEY)

        # add to chat messages
        st.session_state["chat_messages"].append({"role":"user","text":user_msg, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        st.session_state["chat_messages"].append({"role":"assistant","text":resp, "time": time.strftime("%Y-%m-%d %H:%M:%S")})

        # persist chat to CSV
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

    # render chat messages
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

    # Load only language-appropriate program guides
    guides = load_program_guides_for_lang(lang)
    if not guides:
        # Explain how filenames must be placed for English vs Hindi
        st.info(
            "No program guides found for the selected language.\n\n"
            "- For English: place files as `kb/programs/<name>.txt` (e.g., kcc.txt, pmfby.txt)\n"
            "- For Hindi: place files as `kb/programs/<name>.hi.txt` (e.g., kcc.hi.txt, pmfby.hi.txt)\n"
        )
    else:
        # classify guides into categories
        cat_map = classify_programs(guides)
        categories = ["Loans", "Insurance", "Mechanization", "Policies & Subsidies", "Other"]
        selected_category = st.selectbox("Choose category", options=categories, index=0)

        available_guides = cat_map.get(selected_category, [])
        if not available_guides:
            st.info(f"No guides found in category '{selected_category}' for language {lang}.")
        else:
            # show multiselect of files in the chosen category
            guide_labels = [f"{g['id']}" for g in available_guides]
            selected_files = st.multiselect("Choose program document(s) to use as context", options=guide_labels, default=guide_labels[:1])

            # map selected labels back to guide dicts
            selected_guides = [g for g in available_guides if g['id'] in selected_files]

            # Quick preview (collapsible)
            with st.expander("Preview selected documents", expanded=False):
                for g in selected_guides:
                    st.markdown(f"**{g['id']}** — {g['source']}")
                    st.text(g['text'][:1500] + ("..." if len(g['text']) > 1500 else ""))

            st.markdown("---")
            # Question input and actions
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
                    # prepare context - truncated per-file to avoid huge prompts
                    context_text = prepare_context_text(selected_guides, max_chars_per_file=3500)

                    # build prompt
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

                    # Optionally persist this query + context selection
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
with tab_cropcal:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Crop Calendar — Kharif & Rabi seasons (indicative)")
    st.markdown("This table lists common crops and their typical planting/season windows. Local variations apply — confirm with state agriculture office.")

    # Build a simple table
    st.table(CROP_CALENDAR)

    st.markdown("---")
    st.markdown("Notes:")
    st.markdown(
        "- The seasons shown are indicative and can vary by state, variety and irrigation availability.\n"
        "- For precise sowing/harvest windows for your district, consult the local Krishibhavan or agriculture extension.\n"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Fertilizer Calculator tab ----------------
with tab_fertcalc:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Fertilizer Calculator")
    st.markdown("Select crop, enter area and unit, then compute estimated nutrient and fertilizer requirements. Values are indicative — prefer soil test recommendations.")

    # Choose crop (provide same crop set)
    fcrop = st.selectbox("Crop for calculation", options=["rice", "banana", "coconut", "tapioca", "vegetable", "other"], index=0)

    # Area input + unit selector
    col_a1, col_a2 = st.columns([2,1])
    with col_a1:
        area_value = st.number_input("Area", min_value=0.0, format="%.3f", value=1.0, help="Enter numeric area")
    with col_a2:
        area_unit = st.selectbox("Unit", options=["hectare", "acre", "cent", "sq_m", "sq_ft"], index=0)

    st.markdown("**Nutrient recommendation (kg/ha)** — edit if you have local values or soil test results.")
    # load defaults or allow custom
    if fcrop in DEFAULT_NPK_PER_HA:
        default_npk = DEFAULT_NPK_PER_HA[fcrop]
    else:
        default_npk = DEFAULT_NPK_PER_HA["other"]

    col_n, col_p, col_k = st.columns(3)
    with col_n:
        n_per_ha = st.number_input("N (kg/ha)", min_value=0.0, value=float(default_npk["N"]))
    with col_p:
        p_per_ha = st.number_input("P2O5 (kg/ha)", min_value=0.0, value=float(default_npk["P2O5"]))
    with col_k:
        k_per_ha = st.number_input("K2O (kg/ha)", min_value=0.0, value=float(default_npk["K2O"]))

    st.markdown("**Bag conventions (optional)** — default bag weight 50 kg. Change if your supplier uses different bag sizes.")
    bag_u = st.number_input("Urea bag weight (kg)", min_value=1.0, value=50.0)
    bag_d = st.number_input("DAP bag weight (kg)", min_value=1.0, value=50.0)
    bag_m = st.number_input("MOP bag weight (kg)", min_value=1.0, value=50.0)

    if st.button("Calculate fertilizer requirement"):
        hectares = area_to_hectares(area_value, area_unit)
        if hectares <= 0:
            st.error("Enter a valid positive area.")
        else:
            st.markdown(f"**Area in hectares:** {hectares:.4f} ha")
            npk_need = compute_nutrient_requirements(hectares, {"N": n_per_ha, "P2O5": p_per_ha, "K2O": k_per_ha})
            st.markdown("**Estimated nutrient requirement:**")
            st.write(f"- Nitrogen (N): **{npk_need['N']} kg**")
            st.write(f"- Phosphorus (P2O5): **{npk_need['P2O5']} kg**")
            st.write(f"- Potassium (K2O): **{npk_need['K2O']} kg**")

            st.markdown("**Suggested fertilizer amounts (approx.)** — using DAP for P, MOP for K, rest N from Urea.")
            bag_weights = {"urea": bag_u, "dap": bag_d, "mop": bag_m}
            fert_plan = compute_fertilizer_bags(npk_need, bag_weights, FERT_CONTENT)
            st.write(f"- DAP: **{fert_plan['DAP_kg']} kg** (~**{fert_plan['DAP_bags']}** bags of {bag_d} kg)")
            st.write(f"- MOP: **{fert_plan['MOP_kg']} kg** (~**{fert_plan['MOP_bags']}** bags of {bag_m} kg)")
            st.write(f"- Urea: **{fert_plan['Urea_kg']} kg** (~**{fert_plan['Urea_bags']}** bags of {bag_u} kg)")

            st.markdown("---")
            st.markdown(
                "**Important:** These calculations are approximate. Always prefer local soil-test based recommendations and check split application timing (basal/topdressing) and crop stage guidelines. For safety-critical dosing, consult local Krishibhavan or an agronomist."
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Admin / Weather tab ----------------
with tab_admin:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Weather & SMS Alerts")
    st.markdown("Enter Location and Phone in the sidebar. Use the button to fetch weather and optionally send SMS to the farmer.")

    if not location or not phone:
        st.info("Set Location and Phone in the sidebar to use weather fetch and SMS features.")
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

                if st.button("Send this weather SMS to farmer now"):
                    ok, info = send_weather_sms_verbose(TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, phone, msg_text)
                    if ok:
                        st.success(f"📩 Weather SMS sent to {phone}")
                    else:
                        st.error(f"❌ Failed to send SMS: {info}")

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
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.caption(
    "KissanConnect prototype — multimodal advice. For production, move uploads to S3, secure admin, and implement SDK-native multimodal attachments for Gemini."
)