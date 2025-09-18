import os
import re
import base64
import datetime
import requests
import concurrent.futures
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image, UnidentifiedImageError
from io import BytesIO

# --- Load environment variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

# Basic check for essential keys
if not all([GOOGLE_API_KEY, OPENWEATHER_API_KEY, WHATSAPP_TOKEN, PHONE_NUMBER_ID, VERIFY_TOKEN]):
    raise ValueError("One or more required environment variables are not set.")

# --- Helper Functions ---
def get_weather_forecast(api_key, city_name):
    """Fetches weather forecast for a given city."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = r.json()
        if data.get("cod") != 200:
            return f"⚠️ Weather information is currently unavailable for {city_name}."
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"Current weather in {city_name}: {desc.capitalize()} with a temperature of {temp}°C."
    except requests.exceptions.RequestException as e:
        print(f"Weather API request error: {e}")
        return "⚠️ The weather service is currently experiencing issues."
    except Exception as e:
        print(f"An unexpected error occurred in get_weather_forecast: {e}")
        return "⚠️ Could not retrieve weather information."

def extract_city(text):
    """A simple function to find a city name after a keyword."""
    # This is a basic example. A more robust solution might use NLP.
    match = re.search(r"(?:weather|forecast|temperature) in (\w+)", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None

# --- Initialize Gemini model ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    print("✅ Gemini model loaded successfully.")
except Exception as e:
    print(f"❌ Critical Error: Could not load Gemini model. {e}")
    llm = None # Set to None to handle gracefully later

# --- Flask App ---
app = Flask(__name__)
MAX_LENGTH = 1600  # A safe WhatsApp message limit

# --- Send WhatsApp Message ---
def send_whatsapp_message(to, text):
    """Sends a text message via WhatsApp Cloud API."""
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text[:MAX_LENGTH]}
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        print(f"📩 Message sent to {to}: {r.json()}")
    except requests.exceptions.RequestException as e:
        print(f"❌ WhatsApp API send error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred in send_whatsapp_message: {e}")

# --- Webhook Verification (for setup) ---
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Verifies the webhook with Meta."""
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        print("✅ Webhook verified.")
        return request.args.get("hub.challenge")
    print("❌ Webhook verification failed.")
    return "Verification failed", 403

# --- Receive Messages ---
@app.route("/webhook", methods=["POST"])
def webhook():
    """Handles incoming WhatsApp messages."""
    if llm is None:
        print("❌ AI model not loaded. Cannot process messages.")
        return jsonify({"status": "error", "message": "AI model not available"}), 503

    data = request.get_json()
    print("📥 Incoming webhook payload:", data)

    try:
        entry = data["entry"][0]
        change = entry["changes"][0]
        value = change["value"]
        message = value["messages"][0]
        from_number = message["from"]
        msg_type = message.get("type")

        # --- IMAGE or DOCUMENT (as image) PROCESSING ---
        if msg_type in ["image", "document"]:
            media_id = ""
            mime_type = ""
            if msg_type == "image":
                media_id = message["image"]["id"]
                mime_type = message["image"].get("mime_type", "image/jpeg")
            elif msg_type == "document":
                # Ensure the document is an image type
                if not message["document"].get("mime_type", "").startswith("image"):
                    send_whatsapp_message(from_number, "⚠️ This bot can only analyze images. Please send a photo, not a document.")
                    return jsonify({"status": "ok"})
                media_id = message["document"]["id"]
                mime_type = message["document"]["mime_type"]

            # **REVISED TWO-STEP IMAGE DOWNLOAD**
            media_info_url = f"https://graph.facebook.com/v19.0/{media_id}"
            headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

            try:
                # 1. Get the media metadata which contains the download URL
                info_r = requests.get(media_info_url, headers=headers, timeout=10)
                info_r.raise_for_status()
                download_url = info_r.json().get("url")

                if not download_url:
                    raise ValueError("Failed to retrieve media download URL.")

                # 2. Download the actual image bytes from the URL
                image_r = requests.get(download_url, headers=headers, timeout=10)
                image_r.raise_for_status()
                image_bytes = image_r.content

                # 3. Process the image
                img = Image.open(BytesIO(image_bytes))
                img.verify()
                img_b64 = base64.b64encode(image_bytes).decode("utf-8")

                prompt = "You are Kisan Mitra 🌾. Analyze this plant image, identify potential diseases or pests, and suggest simple, actionable solutions for a farmer."
                human_msg = HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:{mime_type};base64,{img_b64}"}
                ])
                answer = llm.invoke([human_msg]).content
                send_whatsapp_message(from_number, answer)

            except (UnidentifiedImageError, requests.exceptions.RequestException) as e:
                print(f"❌ Image analysis error: {e}")
                send_whatsapp_message(from_number, "⚠️ Could not analyze the image. Please ensure it's a clear and valid photo.")
            except Exception as e:
                print(f"❌ AI error during image processing: {e}")
                send_whatsapp_message(from_number, "⚠️ An unexpected error occurred while analyzing the image. Please try again later.")

        # --- TEXT MESSAGE PROCESSING ---
        elif msg_type == "text":
            user_msg = message["text"]["body"]
            try:
                current_date = datetime.date.today().strftime("%B %d, %Y")
                weather_info = ""
                # Check for weather keywords
                if any(word in user_msg.lower() for word in ["weather", "rain", "climate", "temperature", "forecast"]):
                    city = extract_city(user_msg) or "Pune"  # Default to Pune if no city is extracted
                    weather_info = get_weather_forecast(OPENWEATHER_API_KEY, city)

                system_prompt = f"""You are Kisan Mitra 🌱, a helpful and friendly AI assistant for farmers.
                Your goal is to provide simple, clear, and practical agricultural advice.
                - Current Date: {current_date}
                - Weather Context: {weather_info if weather_info else 'Not requested by user'}
                - Language: Respond in simple, farmer-friendly language. Avoid overly technical jargon.
                """
                messages_list = [SystemMessage(content=system_prompt), HumanMessage(content=user_msg)]

                # Use a thread pool to run the AI call with a timeout
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: llm.invoke(messages_list).content)
                    answer = future.result(timeout=20) # 20-second timeout

                send_whatsapp_message(from_number, answer)

            except concurrent.futures.TimeoutError:
                print("❌ AI request timed out.")
                send_whatsapp_message(from_number, "⚠️ The AI service is taking too long to respond. Please try again in a moment.")
            except Exception as e:
                print(f"❌ AI error during text processing: {e}")
                send_whatsapp_message(from_number, "⚠️ The AI service is currently busy. Please try your request again.")

    except (KeyError, IndexError) as e:
        print(f"❌ Could not parse incoming webhook payload: {e}")
        # Ignore status updates or other non-message payloads
        pass

    return jsonify({"status": "ok"})

# --- Run Server ---
if __name__ == "__main__":
    print("🚀 Starting Kisan Mitra Flask server...")
    # For production, consider using a proper WSGI server like Gunicorn
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)