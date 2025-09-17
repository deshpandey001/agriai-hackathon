# app.py - Fixed reliable Flask backend for KissanConnect (programs section updated)
import os
import csv
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from flask import Flask, request, jsonify, render_template, send_from_directory
import requests

# Optional external libs handled gracefully
try:
    import google.generativeai as genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

try:
    from twilio.rest import Client as TwilioClient
    HAS_TWILIO = True
except Exception:
    HAS_TWILIO = False

# ---------- Configuration ----------
APP_NAME = "KissanConnect"
BASE_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
KB_DIR = BASE_DIR / "kb"
PROGRAMS_DIR = KB_DIR / "programs"
OUTPUT_DIR = BASE_DIR / "output"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

ESCALATION_FILE = OUTPUT_DIR / "escalations.csv"
FEEDBACK_FILE = OUTPUT_DIR / "feedback.csv"
CHAT_HISTORY_FILE = OUTPUT_DIR / "chat_history.csv"

# API keys (set in environment)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
TWILIO_SID = os.environ.get("TWILIO_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
TWILIO_FROM = os.environ.get("TWILIO_FROM", "")

# Create a single Flask app (no redefinition)
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))

# ----------------- Helpers -----------------
def ensure_csv_has_header(path: Path, header: List[str]):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

def safe_render(template_name: str, **kwargs):
    """
    Render template but return a readable error page on failure.
    """
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
    """
    Load program files (language-aware):
      - English: <name>.txt
      - Hindi: <name>.hi.txt
    Returns list of {id, text, source}
    """
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
    """
    Keyword-based classification into categories:
    Loans, Insurance, Mechanization, Policies & Subsidies, Other
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

def prepare_context_text(guides: List[Dict], max_chars_per_file: int = 3500) -> str:
    parts = []
    for g in guides:
        text = g["text"]
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + "\n\n[TRUNCATED]"
        parts.append(f"[{g['id']}]\n" + text)
    return "\n\n".join(parts)

# ---------------- LLM wrapper (Gemini stub) ----------------
def generate_answer(prompt_text: str) -> str:
    """
    Generate a response using Google's Gemini API.
    Requires GEMINI_API_KEY in environment and google-generativeai installed.
    """
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return "(Gemini not configured: Please set GEMINI_API_KEY)"
        
        genai.configure(api_key=api_key)
        # choose model that you have access to; change if using a different model name
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt_text)

        if hasattr(response, "text") and response.text:
            return response.text.strip()
        return str(response)
    except Exception as e:
        return f"(Gemini error: {e})"

# ---------------- Twilio send helper ----------------
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

# Admin/login/signup (these endpoint names must match templates)
@app.route("/admin")
def admin_page():
    return safe_render("admin.html")

# static serving
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

@app.route("/api/get_advice", methods=["POST"])
def api_get_advice():
    j = request.get_json(force=True)
    query = j.get("query", "")
    location = j.get("location", "")
    crop = j.get("crop", "")
    lang = j.get("language", "English")
    image_b64 = j.get("image_base64")
    guides = load_program_guides_for_lang(lang)
    context_text = prepare_context_text(guides[:3])
    system_prompt = (
        "You are KrishiAdviser — a concise practical agricultural assistant.\n"
        f"Location: {location}\nCrop: {crop}\nLanguage: {lang}\n"
        "Use the following sources when relevant and cite them in square brackets.\n"
    )
    prompt_parts = ["SYSTEM:\n" + system_prompt]
    if context_text:
        prompt_parts.append("SOURCES:\n" + context_text)
    if query:
        prompt_parts.append("USER QUERY:\n" + query)
    prompt_text = "\n\n".join(prompt_parts)
    if image_b64:
        prompt_text += "\n\n[Image attached — analyze visually if possible]\n" + (image_b64[:4000] if isinstance(image_b64, str) else "")
    answer = generate_answer(prompt_text)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "location": location,
        "crop": crop,
        "query": query,
        "answer_excerpt": answer[:500],
        "language": lang,
    }
    ensure_csv_has_header(FEEDBACK_FILE, list(row.keys()))
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)
    return jsonify({"ok": True, "answer": answer})

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
    user_msg = j.get("message", "")
    context_answer = j.get("context_answer", "")
    lang = j.get("language", "English")
    sys_prompt = "You are KrishiAdviser Chatbot. Keep answers short and practical."
    if context_answer:
        sys_prompt += "\nPrevious advice (context):\n" + context_answer
    if lang.lower().startswith("h"):
        sys_prompt += "\nRespond in Hindi."
    else:
        sys_prompt += "\nRespond in English."
    prompt_text = "SYSTEM:\n" + sys_prompt + "\nUSER:\n" + user_msg
    answer = generate_answer(prompt_text)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "context": context_answer[:200],
        "message": user_msg,
        "answer": answer[:2000],
        "language": lang,
    }
    ensure_csv_has_header(CHAT_HISTORY_FILE, list(row.keys()))
    with open(CHAT_HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)
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
    """
    Accepts { language: "English"|"Hindi", category: <category name or empty> }
    Returns compact program listing (id, title, excerpt, source).
    If category is present, returns only docs in that category; otherwise returns all docs.
    """
    j = request.get_json() or {}
    lang = j.get("language", "English")
    category = j.get("category", "")  # optionally filter
    guides = load_program_guides_for_lang(lang)
    cat_map = classify_programs(guides)
    out_list = []
    if category and category in cat_map:
        sel = cat_map.get(category, [])
    else:
        # flatten all categories
        sel = []
        for v in cat_map.values():
            sel.extend(v)
    for g in sel:
        out_list.append({
            "id": g["id"],
            "title": g["id"],
            "excerpt": g["text"][:250].replace("\n", " "),
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
