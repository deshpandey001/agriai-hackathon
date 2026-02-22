# KissanConnect (website_new)

KissanConnect is a Flask-based web app that provides AI-assisted agricultural advisory for farmers, including:

- Chat-style advisory powered by Google Gemini
- Weather-aware recommendations using OpenWeather
- Local speech-to-text using Whisper (no OpenAI API required)
- Scheme/program discovery and summaries
- Crop calendar and fertilizer calculator
- WhatsApp integration via Twilio (optional)
- Basic admin/analytics views backed by CSV logs

---

## Tech Stack

- **Backend:** Python, Flask
- **LLM:** Google Gemini via `google-generativeai`
- **Transcription (optional):** `openai-whisper` + FFmpeg
- **Messaging (optional):** Twilio WhatsApp API
- **Weather:** OpenWeather REST API
- **Storage:** Local JSON + CSV files under `output/`

---

## Project Structure

```text
website_new/
  app.py                 # Main Flask app
  templates/             # HTML templates (Jinja2)
  static/                # CSS, JS, images, uploads
    uploads/             # Saved user images (served via /static/uploads)
  kb/
    programs/            # .txt program/scheme content (EN + HI)
  output/
    sessions/            # Session JSON files
    uploads/             # Raw uploaded audio/images
    escalations.csv      # Escalation log (if used)
    feedback.csv         # Advisory feedback/log
    chat_history.csv     # Chatbot history
  requirements.txt       # Python dependencies
```

Key backend file: `app.py`.

---

## Prerequisites

- **Python** 3.9+ (recommended 3.10/3.11)
- **Pip** installed and on PATH
- **FFmpeg** (for Whisper audio transcription)
  - Windows: download from https://ffmpeg.org and add `ffmpeg.exe` to PATH
- (Optional) Accounts + API keys:
  - Google AI Studio (Gemini)
  - OpenWeather
  - Twilio (WhatsApp)

---

## Installation & Setup

From a terminal / PowerShell:

```powershell
cd f:\KissanConnect\website_new
```

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

(If PowerShell execution policy blocks scripts, run as Admin:)

```powershell
Set-ExecutionPolicy RemoteSigned
```

Then re-run the `Activate.ps1` command.

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

To enable local Whisper transcription:

```powershell
pip install openai-whisper
```

> Whisper also requires FFmpeg installed on the system.

---

## Environment Variables

Set these **before** running `app.py`.

### Required for full AI experience

- `GEMINI_API_KEY` – Google Gemini API key (from Google AI Studio)

### Optional

- `OPENWEATHER_API_KEY` – for `/api/fetch_weather` and weather-aware advice
- `TWILIO_SID`, `TWILIO_TOKEN`, `TWILIO_FROM` – for `/api/send_whatsapp`

Example (PowerShell):

```powershell
$env:GEMINI_API_KEY = "your-gemini-key"
$env:OPENWEATHER_API_KEY = "your-openweather-key"
$env:TWILIO_SID = "ACxxxxxxxx"
$env:TWILIO_TOKEN = "xxxxxxxx"
$env:TWILIO_FROM = "whatsapp:+1415xxxxxxx"
```

---

## Running the App (Dev Mode)

From `website_new` with venv activated:

```powershell
python app.py
```
