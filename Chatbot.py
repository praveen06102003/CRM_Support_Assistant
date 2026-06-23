from flask import Flask, render_template, request, jsonify
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"



def classify_message(user_message):
    classification_prompt = f"""
    Classify the following customer message.
    Return ONLY a JSON object with two keys: "intent" and "sentiment".

    intent must be one of: billing, technical, refund, general
    sentiment must be one of: positive, neutral, negative, frustrated

    Message: "{user_message}"
    """

    payload = {
        "contents": [{"parts": [{"text": classification_prompt}]}]
    }

    response = requests.post(GEMINI_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        raw_text = result['candidates'][0]['content']['parts'][0]['text']
        raw_text = raw_text.strip().replace("```json", "").replace("```", "")
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {"intent": "general", "sentiment": "neutral"}
    else:
        return {"intent": "general", "sentiment": "neutral"}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_message = data.get('message')

    classification = classify_message(user_message)
    intent = classification.get("intent", "general")
    sentiment = classification.get("sentiment", "neutral")

    if sentiment == "frustrated":
        return jsonify({
            "response": "I can see this is frustrating — let me connect you with a human agent right away.",
            "intent": intent,
            "sentiment": sentiment
        })

    payload = {
        "contents": [{"parts": [{"text": user_message}]}]
    }

    response = requests.post(GEMINI_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        chatbot_reply = result['candidates'][0]['content']['parts'][0]['text']
        return jsonify({
            "response": chatbot_reply,
            "intent": intent,
            "sentiment": sentiment
        })
    else:
        print("Gemini API error:", response.status_code, response.text)
        return jsonify({"response": "Error: Could not connect to API."}), 500


if __name__ == '__main__':
    app.run(debug=True)