import os
import logging
from dotenv import load_dotenv
from typing import Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from httpx import TimeoutException

# --- Alexa SDK Imports ---
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_model import Response
# --- END Alexa SDK Imports ---

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Configuration (GLOBAL) ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
GENERATOR_MODEL = "llama-3.1-8b-instant"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Retry-safe LLM request ---
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(TimeoutException), reraise=True)
def generate_response_with_retry(client, messages, model):
    return client.chat.completions.create(model=model, messages=messages, temperature=0.2)

# --- RAG Core Logic ---
def get_rag_response(query: str, session_attributes: dict) -> Tuple[str, str]:
    """
    Retrieves relevant context, queries the LLM, and returns (answer, follow-up).
    """
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
        logger.error("API Key not set.")
        return (
            "I can’t reach the language model yet because the API key is missing.",
            'Follow-up: "Would you like help adding the key to .env?"'
        )

    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 4})

        history = session_attributes.get("history", [])
        contextual_query = f"Conversation History: {history[-1]}. NEW QUERY: {query}" if history else query

        retrieved_docs = retriever.invoke(contextual_query)
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        client = Groq(api_key=GROQ_API_KEY)
        SYSTEM_PROMPT = (
            "You are an AI Tutor that explains technical concepts in simple, structured language. "
            "Every answer should include a concise explanation followed by a follow-up suggestion "
            "formatted as: 'Follow-up: \"<suggested next question>\"'."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {contextual_query}"}
        ]

        response = generate_response_with_retry(client, messages, GENERATOR_MODEL)
        raw_response = response.choices[0].message.content.strip()

        # --- Parse for follow-up ---
        if "Follow-up:" in raw_response:
            parts = raw_response.split("Follow-up:")
            answer_text = parts[0].strip()
            follow_up_text = f"Follow-up:{parts[1].strip()}"
        else:
            answer_text = raw_response
            follow_up_text = 'Follow-up: "Would you like to explore another related topic?"'

        # --- Clean formatting ---
        answer_text = " ".join(answer_text.split())
        follow_up_text = " ".join(follow_up_text.split())

        return answer_text, follow_up_text

    except Exception as e:
        logger.error(f"RAG Function Failed: {e}")
        return (
            "I am experiencing technical difficulties reaching the tutor's brain. Please try again.",
            'Follow-up: "Would you like to retry or ask another topic?"'
        )

# ======================================================================
# --- ALEXA HANDLERS ---
# ======================================================================

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        speech_text = (
            "Welcome to the AI Tutor! "
            "I can teach you about topics like Neural Networks or Regression. "
            "What subject would you like to explore?"
        )
        reprompt_text = "You can say, 'Teach me about supervised learning.'"
        handler_input.response_builder.speak(speech_text).ask(reprompt_text)
        return handler_input.response_builder.response


class TeachMeIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("TeachMeIntent")(handler_input)

    def handle(self, handler_input):
        slots = handler_input.request_envelope.request.intent.slots
        topic_slot = slots.get("topic")
        query = f"Teach me about {topic_slot.value}" if topic_slot and topic_slot.value else "What is Artificial Intelligence?"

        attributes_manager = handler_input.attributes_manager
        session_attributes = attributes_manager.session_attributes

        answer_text, follow_up_question = get_rag_response(query, session_attributes)

        # Save for context and multi-turn memory
        session_attributes["last_topic"] = query
        session_attributes["next_question"] = follow_up_question.replace("Follow-up:", "").strip().strip('"')

        speech_output = f"{answer_text} {follow_up_question}"
        handler_input.response_builder.speak(speech_output).ask(follow_up_question)
        return handler_input.response_builder.response


class YesIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.YesIntent")(handler_input)

    def handle(self, handler_input):
        attributes_manager = handler_input.attributes_manager
        session_attributes = attributes_manager.session_attributes

        next_question = session_attributes.get("next_question")
        if not next_question:
            speech_text = "Let's pick up a new topic! What would you like to learn next?"
            handler_input.response_builder.speak(speech_text).ask(speech_text)
            return handler_input.response_builder.response

        answer_text, follow_up_question = get_rag_response(next_question, session_attributes)
        session_attributes["next_question"] = follow_up_question.replace("Follow-up:", "").strip().strip('"')

        speech_output = f"{answer_text} {follow_up_question}"
        handler_input.response_builder.speak(speech_output).ask(follow_up_question)
        return handler_input.response_builder.response


class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        return handler_input.response_builder.response


class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        speech = "Sorry, I had trouble doing what you asked. Please try again or say 'Help'."
        handler_input.response_builder.speak(speech).ask(speech)
        return handler_input.response_builder.response

# --- FINAL GLOBAL SKILL BUILDER OBJECT ---
sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(TeachMeIntentHandler())
sb.add_request_handler(YesIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_exception_handler(CatchAllExceptionHandler())
