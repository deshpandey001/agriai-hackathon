# app.py - KissanConnect Flask backend (session-aware advisory + local whisper transcription)
import os
import csv
import json
import traceback
import base64
import secrets
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from flask import Flask, request, jsonify, render_template, send_from_directory
import requests

# Optional external libs handled gracefully
try:
    import google.generativeai as genai  # for Gemini if available
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

try:
    from twilio.rest import Client as TwilioClient
    HAS_TWILIO = True
except Exception:
    HAS_TWILIO = False

# Local whisper (open-source) support (no OpenAI API key required)
# Install with: pip install -U openai-whisper
try:
    import whisper
    HAS_LOCAL_WHISPER = True
except Exception:
    HAS_LOCAL_WHISPER = False

# ---------- Configuration ----------
APP_NAME = "KissanConnect"
BASE_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
KB_DIR = BASE_DIR / "kb"
PROGRAMS_DIR = KB_DIR / "programs"
OUTPUT_DIR = BASE_DIR / "output"
SESSIONS_DIR = OUTPUT_DIR / "sessions"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

ESCALATION_FILE = OUTPUT_DIR / "escalations.csv"
FEEDBACK_FILE = OUTPUT_DIR / "feedback.csv"
CHAT_HISTORY_FILE = OUTPUT_DIR / "chat_history.csv"

# API keys (set in environment)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
TWILIO_SID = os.environ.get("TWILIO_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
TWILIO_FROM = os.environ.get("TWILIO_FROM", "")

# Create Flask app
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))

# ----------------- Utilities -----------------
def ensure_csv_has_header(path: Path, header: List[str]):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

def safe_render(template_name: str, **kwargs):
    try:
        return render_template(template_name, **kwargs)
    except Exception:
        tb = traceback.format_exc()
        return (
            f"<h3>Template render error for {template_name}</h3>"
            f"<pre>{tb}</pre>"
            f"<p>Make sure templates/{template_name} exists and is valid HTML.</p>"
        ), 500

# ---------------- Weather helpers ----------------
def fetch_weather_raw(location: str):
    if not location or not OPENWEATHER_API_KEY:
        return None
    try:
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": location, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        r = requests.get(url, params=params, timeout=8)
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
    weather_desc = weather.get("description", "")
    rain_1h = weather_json.get("rain", {}).get("1h", 0.0)
    triggers = []
    if temp is not None and temp >= 35.0:
        triggers.append("heat")
    if temp is not None and temp <= 2.0:
        triggers.append("frost")
    if rain_1h and rain_1h >= 10.0:
        triggers.append("heavy_rain")
    if wind_speed and wind_speed >= 10.0:
        triggers.append("high_wind")
    return {
        "temp": temp,
        "feels_like": feels_like,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "rain_1h": rain_1h,
        "weather_desc": weather_desc,
        "triggers": triggers,
    }

def build_weather_message(location: str, interp: Dict, lang: str = "English") -> str:
    if not interp:
        return f"Weather data for {location} is unavailable."
    if lang.lower().startswith("h"):
        parts = []
        if interp.get("triggers"):
            parts.append("अलर्ट: " + ", ".join(interp["triggers"]))
        parts.append(f"स्थान: {location}")
        parts.append(f"तापमान: {interp.get('temp')}°C (अनुभव: {interp.get('feels_like')})")
        parts.append(f"आर्द्रता: {interp.get('humidity')}%")
        parts.append(f"विवरण: {interp.get('weather_desc')}")
        return " | ".join(parts)
    else:
        parts = []
        if interp.get("triggers"):
            parts.append("ALERT: " + ", ".join(interp["triggers"]))
        parts.append(f"Location: {location}")
        parts.append(f"Temp: {interp.get('temp')}°C (Feels {interp.get('feels_like')})")
        parts.append(f"Humidity: {interp.get('humidity')}%")
        parts.append(f"Description: {interp.get('weather_desc')}")
        return " | ".join(parts)

# ---------------- Programs loader + classifier ----------------
def load_program_guides_for_lang(lang: str) -> List[Dict]:
    guides = []
    if not PROGRAMS_DIR.exists():
        return guides
    for p in sorted(PROGRAMS_DIR.glob("*.txt")):
        name = p.name
        if lang.lower().startswith("h"):
            if not name.endswith(".hi.txt"):
                continue
            base_id = p.stem
            if base_id.endswith(".hi"):
                base_id = base_id[:-3]
            text = p.read_text(encoding="utf-8")
            guides.append({"id": base_id, "text": text, "source": str(p)})
        else:
            if name.endswith(".hi.txt"):
                continue
            text = p.read_text(encoding="utf-8")
            guides.append({"id": p.stem, "text": text, "source": str(p)})
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

# ---------------- LLM wrapper (Gemini-focused) ----------------
def generate_answer(prompt_text: str) -> str:
    """
    Use Gemini if available; otherwise return a helpful fallback message.
    """
    try:
        if GEMINI_API_KEY and HAS_GENAI:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt_text)
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
            if isinstance(resp, dict) and "candidates" in resp and resp["candidates"]:
                return resp["candidates"][0].get("content", str(resp))
            return str(resp)
        else:
            # Not configured: return a clear fallback (useful during dev)
            return "(LLM not configured) — Gemini API key or SDK missing on server.\n\n" + prompt_text[:1500]
    except Exception as e:
        return f"(Gemini error: {e})"

# ---------------- Whisper transcription helper (local whisper) ----------------
# We'll lazily load a whisper model only if transcription is requested.
_WHISPER_MODEL = None
def transcribe_audio_from_base64(b64: str, model_size: str = "small") -> Optional[str]:
    """
    Transcribe audio using local whisper (no OpenAI API).
    - b64: data URL (data:audio/..;base64,...) or plain base64 string
    - model_size: whisper model size to load ("tiny","base","small","medium","large")
    Returns transcription text or None on failure.
    """
    global _WHISPER_MODEL, HAS_LOCAL_WHISPER
    if not HAS_LOCAL_WHISPER:
        print("Local whisper not available (install openai-whisper).")
        return None
    try:
        header = None
        if b64.startswith("data:"):
            header, b64 = b64.split(",", 1)
        audio_bytes = base64.b64decode(b64)

        # Save a temporary audio file (whisper uses ffmpeg to read many formats)
        fname = UPLOADS_DIR / f"audio_{secrets.token_hex(8)}.webm"
        with open(fname, "wb") as f:
            f.write(audio_bytes)

        # Lazy-load model (first use)
        if _WHISPER_MODEL is None:
            try:
                # Loading model may download it first time (large!)
                print(f"Loading whisper model '{model_size}' (this may take time the first run)...")
                _WHISPER_MODEL = whisper.load_model(model_size)
            except Exception as e:
                print("Failed to load whisper model:", e)
                return None

        # Use whisper to transcribe (it will spawn ffmpeg under the hood)
        try:
            result = _WHISPER_MODEL.transcribe(str(fname))
            # result is typically a dict with 'text' key
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            if hasattr(result, "get") and result.get("text"):
                return result.get("text")
            return str(result)
        except Exception as e:
            print("Whisper transcription error:", e)
            return None
    except Exception as e:
        print("transcribe_audio_from_base64 error:", e)
        return None

# New helper: transcribe a saved file path (used by multipart endpoint)
def transcribe_file_with_whisper(filepath: Path, model_size: str = "small") -> Optional[str]:
    global _WHISPER_MODEL, HAS_LOCAL_WHISPER
    if not HAS_LOCAL_WHISPER:
        print("Local whisper not available when transcribing file.")
        return None
    try:
        if _WHISPER_MODEL is None:
            print(f"Loading whisper model '{model_size}' for file transcription...")
            _WHISPER_MODEL = whisper.load_model(model_size)
        result = _WHISPER_MODEL.transcribe(str(filepath))
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        if hasattr(result, "get") and result.get("text"):
            return result.get("text")
        return str(result)
    except Exception as e:
        print("transcribe_file_with_whisper error:", e)
        return None

# ---------------- Twilio helper ----------------
def send_whatsapp_via_twilio(to_number: str, body_text: str):
    if not HAS_TWILIO or not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and to_number):
        return False, "Twilio not configured on server."
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(body=body_text, from_=TWILIO_FROM, to=f"whatsapp:{to_number}")
        return True, {"sid": getattr(msg, "sid", None), "status": getattr(msg, "status", None)}
    except Exception as e:
        return False, str(e)

# ---------------- Crop calendar & fertilizer ----------------
CROP_CALENDAR = [
    {"crop": "Rice", "kharif": "Jun-Sep", "rabi": "Nov-Feb"},
    {"crop": "Wheat", "kharif": "-", "rabi": "Nov-Apr"},
    {"crop": "Maize", "kharif": "Jun-Sep", "rabi": "Oct-Dec"},
    {"crop": "Banana", "kharif": "Year-round", "rabi": "Year-round"},
]

FERT_REQ = {
    "rice": {"N": 120, "P2O5": 60, "K2O": 40},
    "wheat": {"N": 150, "P2O5": 60, "K2O": 40},
    "maize": {"N": 120, "P2O5": 60, "K2O": 40},
    "banana": {"N": 300, "P2O5": 100, "K2O": 300},
}
KAND_TO_HECTARE = 0.4

def calculate_fertilizer(crop: str, area: float, area_unit="hectare"):
    crop_key = (crop or "").lower()
    if area_unit == "kand":
        area_ha = area * KAND_TO_HECTARE
    elif area_unit == "acres":
        area_ha = area * 0.404686
    else:
        area_ha = area
    req = FERT_REQ.get(crop_key)
    if not req:
        return {"error": f"No fertilizer baseline for {crop}"}
    return {k: round(v * area_ha, 2) for k, v in req.items()}

# ---------------- Session utilities ----------------
def make_session_id():
    return secrets.token_hex(8)

def save_session(session_id: str, data: dict):
    path = SESSIONS_DIR / f"{session_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_session(session_id: str) -> Optional[dict]:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_sessions(limit: int = 50) -> List[dict]:
    out = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            out.append({"id": p.stem, "created_at": s.get("created_at"), "question": s.get("question"), "crop": s.get("crop"), "location": s.get("location")})
        except Exception:
            continue
    return out

# ---------------- Template routes ----------------
@app.route("/")
def index():
    return safe_render("index.html")

@app.route("/advisory")
def advisory_page():
    return safe_render("advisory.html")

@app.route("/chatbot")
def chatbot_page():
    return safe_render("chatbot.html")

@app.route("/programs")
def programs_page():
    return safe_render("programs.html")

@app.route("/crop-calendar")
def crop_calendar_page():
    return safe_render("crop-calendar.html")

@app.route("/fertilizer-calculator")
def fert_page():
    return safe_render("fertilizer-calculator.html")

@app.route("/admin")
def admin_page():
    return safe_render("admin.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# ----------------- API endpoints -----------------
@app.route("/api/fetch_weather", methods=["POST"])
def api_fetch_weather():
    j = request.get_json() or {}
    location = j.get("location") or request.form.get("location") or ""
    lang = j.get("language", "English")
    raw = fetch_weather_raw(location)
    if not raw:
        return jsonify({"ok": False, "error": "Failed to fetch weather"}), 400
    interp = interpret_weather_conditions(raw)
    msg = build_weather_message(location, interp, lang=lang)
    return jsonify({"ok": True, "message": msg, "raw": raw, "interp": interp})

@app.route("/api/send_whatsapp", methods=["POST"])
def api_send_whatsapp():
    j = request.get_json() or {}
    phone = j.get("phone") or request.form.get("phone")
    message = j.get("message") or request.form.get("message")
    ok, info = send_whatsapp_via_twilio(phone, message)
    if ok:
        return jsonify({"ok": True, "info": info})
    return jsonify({"ok": False, "error": info}), 400

# ---------------- New multipart transcription endpoint ----------------
@app.route("/api/transcribe_local", methods=["POST"])
def api_transcribe_local():
    """
    Accepts multipart form with file field 'audio' (Blob/webm, wav etc).
    Saves to UPLOADS_DIR then transcribes using local whisper model.
    Returns JSON: { ok: True, text: "<transcribed text>" } or error.
    """
    if 'audio' not in request.files:
        return jsonify({"ok": False, "error": "No 'audio' file part"}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"ok": False, "error": "Empty filename"}), 400

    # save file
    fname = UPLOADS_DIR / f"transcribe_{secrets.token_hex(8)}_{audio_file.filename}"
    try:
        audio_file.save(str(fname))
    except Exception as e:
        print("Failed to save uploaded audio:", e)
        return jsonify({"ok": False, "error": "Failed to save audio"}), 500

    # transcribe using whisper
    if not HAS_LOCAL_WHISPER:
        print("Transcription requested but whisper not installed.")
        return jsonify({"ok": False, "error": "Local whisper not installed on server. Install 'openai-whisper'."}), 500

    text = transcribe_file_with_whisper(fname, model_size="small")
    if text is None:
        return jsonify({"ok": False, "error": "Transcription failed (check server logs for details)"}), 500

    return jsonify({"ok": True, "text": text})

@app.route("/api/get_advice", methods=["POST"])
def api_get_advice():
    """
    New: creates and returns a session_id. Accepts:
      { query, location, crop, language, image_base64 (data url), audio_base64 (data url) }
    """
    j = request.get_json(force=True)
    query = j.get("query", "").strip()
    location = j.get("location", "").strip()
    crop = j.get("crop", "").strip()
    lang = j.get("language", "English")
    image_b64 = j.get("image_base64")
    audio_b64 = j.get("audio_base64")

    if not query and not image_b64 and not audio_b64:
        return jsonify({"ok": False, "error": "Please send a question, or an image/audio for analysis."}), 400

    transcription = None
    if audio_b64:
        transcription = transcribe_audio_from_base64(audio_b64)
        if transcription:
            query = (query + "\n\n[User voice transcription: " + transcription + "]").strip()

    guides = load_program_guides_for_lang(lang)
    context_text = prepare_context_text(guides[:3]) if guides else ""

    weather_info = ""
    weather_keywords = ['weather', 'forecast', 'temperature', 'rain', 'irrigate', 'spray', 'humidity']
    if any(k in (query or "").lower() for k in weather_keywords) and location:
        raw = fetch_weather_raw(location)
        if raw:
            interp = interpret_weather_conditions(raw)
            weather_info = build_weather_message(location, interp, lang=lang)

    system_prompt = (
        "You are KrishiAdviser — a concise practical agricultural assistant.\n"
        f"Location: {location}\nCrop: {crop}\nLanguage: {lang}\n"
        f"Weather: {weather_info}\n"
        "Use the following sources when relevant and cite them in square brackets.\n"
    )

    image_note = ""
    if image_b64:
        image_note = "[Image attached: analyze plant symptoms if this is a plant photo.]\n"

    prompt_parts = ["SYSTEM:\n" + system_prompt]
    if context_text:
        prompt_parts.append("SOURCES:\n" + context_text)
    if image_note:
        prompt_parts.append("IMAGE:\n" + image_note)
    if query:
        prompt_parts.append("USER QUERY:\n" + query)
    prompt_text = "\n\n".join(prompt_parts)

    answer = generate_answer(prompt_text)

    sid = make_session_id()
    created_at = datetime.utcnow().isoformat()
    session = {
        "session_id": sid,
        "created_at": created_at,
        "language": lang,
        "question": j.get("query", ""),
        "transcription": transcription,
        "query_augmented": query,
        "location": location,
        "crop": crop,
        "weather_info": weather_info,
        "image_saved": False,
        "image_preview": None,
        "answer": answer,
        "messages": [
            {"role": "user", "text": j.get("query", ""), "ts": created_at},
            {"role": "assistant", "text": answer, "ts": created_at}
        ]
    }

    if image_b64:
        try:
            header = None
            if image_b64.startswith("data:"):
                header, image_b64 = image_b64.split(",", 1)
            img_bytes = base64.b64decode(image_b64)
            img_name = UPLOADS_DIR / f"{sid}_img.jpg"
            with open(img_name, "wb") as f:
                f.write(img_bytes)
            session["image_saved"] = True
            session["image_path"] = str(img_name)
            # we place uploads into static/uploads to be served
            static_uploads = Path(app.static_folder) / "uploads"
            static_uploads.mkdir(parents=True, exist_ok=True)
            # copy saved file to static/uploads
            to_static = static_uploads / img_name.name
            with open(to_static, "wb") as outf:
                outf.write(img_bytes)
            session["image_preview"] = f"/static/uploads/{img_name.name}"
        except Exception as e:
            print("Failed to save image:", e)

    save_session(sid, session)

    row = {
        "timestamp": created_at,
        "session_id": sid,
        "location": location,
        "crop": crop,
        "query": j.get("query", ""),
        "answer_excerpt": answer[:300],
        "language": lang,
    }
    ensure_csv_has_header(FEEDBACK_FILE, list(row.keys()))
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

    return jsonify({"ok": True, "session_id": sid, "answer": answer, "image_preview": session.get("image_preview")})

@app.route("/api/list_sessions", methods=["GET"])
def api_list_sessions():
    s = list_sessions(limit=100)
    return jsonify({"ok": True, "sessions": s})

@app.route("/api/get_session/<session_id>", methods=["GET"])
def api_get_session(session_id: str):
    s = load_session(session_id)
    if not s:
        return jsonify({"ok": False, "error": "Session not found"}), 404
    return jsonify({"ok": True, "session": s})

@app.route("/api/programs_query", methods=["POST"])
def api_programs_query():
    j = request.get_json(force=True)
    selected_docs = j.get("docs", [])
    user_q = j.get("question", "")
    lang = j.get("language", "English")
    guides = load_program_guides_for_lang(lang)
    selected = [g for g in guides if g["id"] in selected_docs] if selected_docs else guides
    if not selected:
        return jsonify({"ok": False, "error": "No program docs selected"}), 400
    context_text = prepare_context_text(selected)
    sys_prompt = "You are KrishiAdviser — use the provided program guides to answer the user's question. Cite relevant document IDs in square brackets."
    if lang.lower().startswith("h"):
        sys_prompt += " Respond in Hindi."
    else:
        sys_prompt += " Respond in English."
    user_q = user_q or "Please summarize the selected program documents and list the key actionable steps for a farmer."
    prompt_text = "\n\n".join(["SYSTEM:\n" + sys_prompt, "SOURCES:\n" + context_text, "USER QUERY:\n" + user_q])
    answer = generate_answer(prompt_text)
    return jsonify({"ok": True, "answer": answer})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    j = request.get_json(force=True)
    user_msg = (j.get("message") or "").strip()
    session_id = j.get("session_id")
    lang = j.get("language", "English")

    if not user_msg:
        return jsonify({"ok": False, "error": "Message is empty"}), 400

    sys_prompt = "You are KrishiAdviser Chatbot. Keep answers short and practical."
    context_answer = ""
    if session_id:
        s = load_session(session_id)
        if s:
            context_answer = s.get("answer", "")
            prev_msgs = s.get("messages", [])
            ctx_lines = []
            for m in prev_msgs[-6:]:
                role = m.get("role")
                txt = m.get("text", "")
                ctx_lines.append(f"[{role}] {txt}")
            ctx_block = "\n".join(ctx_lines)
            sys_prompt += "\nPrevious advisory session context:\n" + ctx_block

    if lang.lower().startswith("h"):
        sys_prompt += "\nRespond in Hindi."
    else:
        sys_prompt += "\nRespond in English."

    prompt_text = "SYSTEM:\n" + sys_prompt + "\nUSER:\n" + user_msg
    if context_answer:
        prompt_text += "\n\nCONTEXT ADVICE:\n" + context_answer

    answer = generate_answer(prompt_text)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id or "",
        "message": user_msg,
        "answer": answer,
        "language": lang,
    }
    ensure_csv_has_header(CHAT_HISTORY_FILE, list(row.keys()))
    with open(CHAT_HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

    if session_id:
        s = load_session(session_id)
        if s:
            s.setdefault("messages", []).append({"role": "user", "text": user_msg, "ts": datetime.utcnow().isoformat()})
            s.setdefault("messages", []).append({"role": "assistant", "text": answer, "ts": datetime.utcnow().isoformat()})
            save_session(session_id, s)

    return jsonify({"ok": True, "answer": answer})

@app.route("/api/fertilizer_calc", methods=["POST"])
def api_fertilizer_calc():
    j = request.get_json(force=True)
    crop = j.get("crop")
    area = float(j.get("area", 0))
    unit = j.get("unit", "hectare")
    res = calculate_fertilizer(crop, area, area_unit=unit)
    return jsonify({"ok": True, "result": res})

@app.route("/api/crop_calendar", methods=["GET"])
def api_crop_calendar():
    return jsonify({"ok": True, "calendar": CROP_CALENDAR})

@app.route("/api/list_programs", methods=["POST"])
def api_list_programs():
    j = request.get_json() or {}
    lang = j.get("language", "English")
    category = j.get("category", "")
    guides = load_program_guides_for_lang(lang)
    cat_map = classify_programs(guides)
    out_list = []
    if category and category in cat_map:
        sel = cat_map.get(category, [])
    else:
        sel = []
        for v in cat_map.values():
            sel.extend(v)
    for g in sel:
        out_list.append({
            "id": g["id"],
            "title": g["id"],
            "excerpt": g["text"][:250].replace("\n", " "),
            "text": g["text"],
            "source": g["source"]
        })
    return jsonify({"ok": True, "programs": out_list})

@app.route("/api/list_categories", methods=["GET"])
def api_list_categories():
    cats = ["Loans", "Insurance", "Mechanization", "Policies & Subsidies", "Other"]
    return jsonify({"ok": True, "categories": cats})

# Diagnostics
@app.route("/_pages", methods=["GET"])
def _pages():
    templates = [p.name for p in sorted(TEMPLATES_DIR.glob("*.html"))] if TEMPLATES_DIR.exists() else []
    routes = sorted([str(r.rule) for r in app.url_map.iter_rules() if r.endpoint != "static"])
    return jsonify({"templates_found": templates, "routes_registered": routes, "templates_dir": str(TEMPLATES_DIR)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "app": APP_NAME, "templates_dir": str(TEMPLATES_DIR), "time": datetime.utcnow().isoformat()})

# -------- Run --------
if __name__ == "__main__":
    DEFAULT_PORT = int(os.environ.get("PORT", 8501))
    port = DEFAULT_PORT
    print(f"Starting KissanConnect Flask app on port {port}...")
    print("Base dir:", BASE_DIR)
    print("Templates dir exists:", TEMPLATES_DIR.exists(), "-", str(TEMPLATES_DIR))
    print("Static dir exists:", Path(app.static_folder).exists(), "-", str(app.static_folder))
    if TEMPLATES_DIR.exists():
        print("Templates found:", [p.name for p in sorted(TEMPLATES_DIR.glob("*.html"))])
    print("Registered routes:")
    for r in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        print(f" - {r} (endpoint={r.endpoint})")
    app.run(host="0.0.0.0", port=port, debug=False)
