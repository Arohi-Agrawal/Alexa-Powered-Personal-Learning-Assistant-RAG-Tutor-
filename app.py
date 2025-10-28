# app.py
import os
import json
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# --- Import Core Logic and SkillBuilder instance ---
# Exports from rag_engine.py
from rag_engine import sb as core_skill_builder, GROQ_API_KEY

# --- Flask Initialization ---
app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create the ASK handler callable ONCE (this mimics AWS Lambda handler)
skill_handler = core_skill_builder.lambda_handler()

@app.route("/alexa", methods=["POST"])
def handle_alexa_request():
    """
    Primary endpoint that receives HTTPS POST from Alexa.
    We MUST pass a dict 'event' to the ASK handler: handler(event, context)
    """
    try:
        # Raw body (optional to keep for debugging)
        # raw = request.get_data(as_text=True)
        # logger.info(f"Incoming Alexa request: {raw[:500]}")

        # 1) Parse into dict (event)
        event = request.get_json(force=True, silent=False)

        # 2) Invoke the SkillBuilder Lambda-style handler
        response = skill_handler(event, None)

        # logger.info(f"Outgoing Alexa response: {str(response)[:500]}")
        return jsonify(response)

    except Exception as e:
        logger.exception(f"Flask /alexa endpoint failed: {e}")
        # Return a valid Alexa JSON on failure so the simulator doesn't spin forever
        return jsonify({
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "PlainText",
                    "text": "The tutor server encountered an internal error. Please try again."
                },
                "shouldEndSession": True
            }
        }), 500


# Optional: quick health probe (useful in a browser or curl)
@app.get("/health")
def health():
    return {"ok": True, "llm_key_present": bool(GROQ_API_KEY)}, 200


# Optional: tiny ping route for quick tunnel checks
@app.get("/ping")
def ping():
    return {"ok": True, "msg": "alexa tutor alive"}, 200


# --- Main Execution ---
if __name__ == "__main__":
    load_dotenv()  # load .env before printing statuses

    print("------------------------------------------------------------------")
    print("--- Starting Flask Web Service for Alexa Skill ---")
    print(f"--- LLM Access Status: {'Set' if GROQ_API_KEY else 'MISSING'} ---")
    print("--- Running locally on port 5000 ---")
    print("------------------------------------------------------------------")
    print("ACTION REQUIRED FOR PHASE 3 COMPLETION:")
    print("1. **Terminal 1:** This window MUST stay running.")
    print("2. **Terminal 2:** Run the ngrok tunnel using:")
    print("   COMMAND: .\\ngrok.exe http 5000")
    print("3. **Link Endpoint:** Copy the ngrok HTTPS URL and paste it into the Alexa Console,")
    print("   adding /alexa at the end, e.g., https://xxxx.ngrok-free.app/alexa")
    print("------------------------------------------------------------------")

    app.run(host="0.0.0.0", port=5000, debug=False)
