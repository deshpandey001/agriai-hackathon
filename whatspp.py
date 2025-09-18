import os
import base64
import io
import datetime
import requests
import concurrent.futures
from PIL import Image
from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse

# Load environment variables
load_dotenv()

# --- IMPORTANT: Set your API Keys ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyDDr9J9joHVOW8LBIZ-8ADwyvct29lp94c"
os.environ["OPENWEATHER_API_KEY"] = "7d273d8981d8eee42e38b8e027dfa60c"

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Google AI API key not found. Please set it.")


# --- Function to get weather data ---
def get_weather_forecast(api_key, city_name):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
    try:
        response = requests.get(complete_url)
        data = response.json()
        if data.get("cod") != "404":
            main = data.get("main", {})
            weather = data.get("weather", [{}])[0]
            temp = main.get("temp")
            desc = weather.get("description")
            return f"Current weather is {desc} with a temperature of {temp}°C."
        else:
            return "Weather information could not be retrieved for this location."
    except Exception as e:
        print(f"Weather API error: {e}")
        return "Weather information is currently unavailable."

# --- 1. Define the AI Model ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    print("Gemini model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    llm = None

# --- 2. Set up Flask Web Server ---
app = Flask(__name__, template_folder='templates')

import concurrent.futures
MAX_LENGTH = 1500

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "").strip()
    resp = MessagingResponse()
    msg = resp.message()

    if not incoming_msg:
        msg.body("Please type a message.")
        return str(resp)

    def query_ai():
        messages = [
            SystemMessage(content="You are a helpful agricultural expert."),
            HumanMessage(content=incoming_msg),
        ]
        return llm.invoke(messages).content

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            answer = executor.submit(query_ai).result(timeout=10)
            if len(answer) > MAX_LENGTH:
                answer = answer[:MAX_LENGTH] + "…"
            msg.body(answer)
    except Exception as e:
        print("AI timeout or error:", e)
        msg.body("⚠️ AI is currently taking too long. Try again later.")

    return str(resp)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return jsonify({"message": "POST request received at root"})
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    """Handles TEXT-BASED questions with DYNAMIC context."""
    data = request.get_json()
    question = data.get('question')
    location = data.get('location', 'India')

    if not question:
        return jsonify({"error": "No question provided"}), 400
        
    if not llm:
        return jsonify({"error": "AI model is not available."}), 500

    try:
        current_date = datetime.date.today().strftime("%B %d, %Y")
        weather_info = "Not requested by user." # Default value

        # --- MODIFIED: Keyword check for conditional weather fetch ---
        weather_keywords = [
            'weather', 'forecast', 'temperature', 'rain', 'rainy', 'sunny', 'wind', 'humidity',
            'climate', 'spray', 'irrigate', 'irrigation', 'water', 'kalavastha', 'mazha'
        ]
        
        if any(keyword in question.lower() for keyword in weather_keywords):
            print("Weather-related keyword detected. Fetching weather information...")
            weather_api_key = os.getenv("OPENWEATHER_API_KEY")
            weather_info = get_weather_forecast(weather_api_key, location)
        # --- End of modification ---

        system_prompt = f"""
        You are a helpful agricultural expert.
        ---
        CURRENT CONTEXT:
        - User's Location: {location}
        - Current Date: {current_date}
        - Current Weather: {weather_info}
        ---
        You MUST use this context to provide specific and timely advice. If the weather was not requested, you do not need to mention it in your answer.
        """
        
        print(f"Received question: '{question}' for location: '{location}'")
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        
        response = llm.invoke(messages)
        answer = response.content
        
        print(f"Generated answer: {answer}")
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Handles IMAGE-BASED questions."""
    data = request.get_json()
    base64_image = data.get('image')
    location = data.get('location', 'India')

    if not base64_image:
        return jsonify({"error": "No image provided"}), 400
    
    if not llm:
        return jsonify({"error": "AI model is not available."}), 500

    try:
        prompt_text = f"You are an agricultural expert. Analyze this image of a plant, which was taken near {location}. Identify any potential diseases or pests you see, describe them, and suggest a solution in simple terms. If the image is not of a plant, state that you can only analyze plant images."

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image.split(',')[1]}"},
            ]
        )
        
        print("Sending image to Gemini for analysis...")
        response = llm.invoke([message])
        answer = response.content
        
        print(f"Generated image analysis: {answer}")
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"An error occurred during image analysis: {e}")
        return jsonify({"error": "An error occurred while analyzing the image."}), 500

if __name__ == '__main__':
    print("Starting Flask server... Open http://127.0.0.1:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000)