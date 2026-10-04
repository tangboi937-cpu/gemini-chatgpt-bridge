import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BRIDGE_API_KEY = os.environ.get("BRIDGE_API_KEY")
PORT = int(os.environ.get("PORT", "3000"))

@app.get("/")
def health():
    return jsonify({
        "name": "Gemini 3.8 Flash ChatGPT Action Bridge",
        "status": "ok",
        "endpoint": "/gemini"
    })

@app.post("/gemini")
def gemini():
    if request.headers.get("X-Bridge-Key") != BRIDGE_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    if not GEMINI_API_KEY:
        return jsonify({"error": "Server is missing GEMINI_API_KEY"}), 500

    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt")
    system_instruction = body.get("system_instruction")
    thinking_level = body.get("thinking_level", "medium")

    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "prompt is required"}), 400

    if thinking_level not in ("low", "medium", "high"):
        thinking_level = "medium"

    combined_prompt = prompt
    if system_instruction:
        combined_prompt = (
            f"System instructions:\n{system_instruction}\n\n"
            f"User:\n{prompt}"
        )

    payload = {
        "model": "gemini-3.8-flash",
        "input": combined_prompt,
        "generation_config": {
            "thinking_level": thinking_level
        }
    }

    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=120
        )
        data = response.json()

        if not response.ok:
            message = data.get("error", {}).get(
                "message", "Gemini API request failed"
            )
            return jsonify({"error": message}), response.status_code

        return jsonify({
            "answer": data.get("output_text", ""),
            "model": "gemini-3.8-flash"
        })

    except requests.RequestException:
        return jsonify({"error": "Could not reach Gemini API"}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
