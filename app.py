import os
import json
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# --- Import Core Logic and SkillBuilder instance ---
# (You already have sb and GROQ_API_KEY exported from rag_engine.py)
from rag_engine import sb as core_skill_builder, GROQ_API_KEY

# --- Flask Initialization ---
app = Flask(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create the ASK handler callable ONCE (this is what Lambda would call)
skill_handler = core_skill_builder.lambda_handler()

@app.route("/alexa", methods=["POST"])
def handle_alexa_request():
    """
    Primary endpoint that receives the HTTPS POST request from Alexa.
    It forwards the parsed JSON 'event' to the ASK handler callable.
    """
    try:
        # 1) Parse the Alexa request body into a dict event
        event = request.get_json(force=True, silent=False)  # MUST be a dict

        # 2) Invoke the ASK handler callable with (event, context)
        response = skill_handler(event, None)

        # 3) Return the ASK JSON response
        return jsonify(response)

    except Exception as e:
        logger.exception(f"Flask endpoint failed: {e}")
        # Return a robust 500 so Alexa doesn't hang
        return jsonify({
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": "The tutor server encountered an internal error. Please check the logs."
                },
                "shouldEndSession": True
            }
        }), 500

# Optional: quick health probe in browser
@app.get("/health")
def health():
    return {"ok": True, "llm_key": bool(GROQ_API_KEY)}

# --- Main Execution ---
if __name__ == "__main__":
    load_dotenv()

    print("------------------------------------------------------------------")
    print("--- Starting Flask Web Service for Alexa Skill ---")
    print(f"--- LLM Access Status: {'Set' if GROQ_API_KEY else 'MISSING'} ---")
    print("--- Running locally on port 5000 ---")
    print("------------------------------------------------------------------")
    print("ACTION REQUIRED FOR PHASE 3 COMPLETION:")
    print("1. **Terminal 1:** This window MUST stay running.")
    print("2. **Terminal 2:** Run the ngrok tunnel using the explicit file path:")
    print("   COMMAND: .\\ngrok.exe http 5000")
    print("3. **Link Endpoint:** Copy the ngrok HTTPS URL and paste it into the Alexa Console.")
    print("------------------------------------------------------------------")

    app.run(host="0.0.0.0", port=5000, debug=False)
