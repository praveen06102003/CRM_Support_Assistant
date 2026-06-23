from flask import Flask, render_template, request, jsonify
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are an intelligent customer support assistant for a CRM platform.
Your job is to help users with:
- Billing and subscription queries
- Technical issues and troubleshooting
- Account setup and management
- Product features and how-to questions
- General customer support queries

Guidelines:
- Always be professional, friendly, and concise
- Keep responses short and clear (2-4 sentences max)
- If the query is too complex, say "Let me connect you with our support team for further assistance"
- If someone asks something completely unrelated to customer support (personal topics, general chat), respond warmly with a creative, natural, and slightly humorous acknowledgment that fits the specific message — never use a fixed template. Then gently redirect back to support. Each response should feel unique and human, not copy-pasted.
- Never make up information you don't know
- Maintain a helpful and calm tone even if the user is frustrated"""


def call_gemini(prompt):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(GEMINI_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        print("Gemini failed:", response.status_code)
        return None


def call_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        print("Groq failed:", response.status_code, response.text)
        return None


def call_llm(prompt):
    # Try Gemini first, fall back to Groq if it fails
    reply = call_gemini(prompt)
    if reply is None:
        print("Falling back to Groq...")
        reply = call_groq(prompt)
    return reply


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

    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_message}\nAssistant:"
    reply = call_llm(full_prompt)

    if reply:
        return jsonify({
            "response": reply,
            "intent": intent,
            "sentiment": sentiment
        })
    else:
        return jsonify({"response": "Our assistant is busy right now, please try again in a moment."}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)