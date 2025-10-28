# rag_engine.py
import os
import re
import logging
from typing import Tuple

from dotenv import load_dotenv

# ---- LangChain imports (prefer modern packages, fallback to community) ----
try:
    # Preferred modern packages
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # fallback

# LLM
from groq import Groq

# Robust retry for LLM calls
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from httpx import TimeoutException

# --- Alexa SDK Imports ---
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.utils import is_request_type, is_intent_name
# from ask_sdk_model import Response  # (not needed directly)
# --- END Alexa SDK Imports ---


# -----------------------------------------------------------------------------
# Logging & Config
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
GENERATOR_MODEL = "llama-3.1-8b-instant"

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Make tokenizers quiet in multi-process environments
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -----------------------------------------------------------------------------
# Retry-safe LLM request
# -----------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(TimeoutException),
    reraise=True,
)
def generate_response_with_retry(client: Groq, messages, model: str):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _build_embeddings() -> HuggingFaceEmbeddings:
    """
    Build embeddings on CPU and normalize vectors to avoid device/meta errors.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_retrieved_docs(retriever, query: str):
    """
    Support both newer and older retriever APIs.
    """
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    return []


def extract_next_question(follow_up_text: str) -> str:
    """
    Returns only the question part of:
      Follow-up: "...."
    and is tolerant to curly quotes / spacing / 'Follow up' variants.
    """
    if not follow_up_text:
        return ""
    m = re.search(r'follow[- ]?up\s*:?\s*[“"](.+?)[”"]', follow_up_text, re.I | re.S)
    if m:
        return m.group(1).strip()
    # Fallback: strip the label and common quotes/spaces
    return (
        follow_up_text
        .replace("Follow-up:", "")
        .replace("Follow up:", "")
        .strip(" \t\r\n\"“”'")
    )


# -----------------------------------------------------------------------------
# RAG Core Logic
# -----------------------------------------------------------------------------
def get_rag_response(query: str, session_attributes: dict) -> Tuple[str, str]:
    """
    Retrieves relevant context, queries the LLM, and returns (answer, follow-up).
    """

    # 1) Safety: ensure API key present
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is missing.")
        return (
            "I can’t reach the language model yet because the API key is missing.",
            'Follow-up: "Would you like help adding the key to .env?"',
        )

    # 2) Ensure vector store exists
    if not os.path.isdir(CHROMA_PATH):
        logger.warning("Chroma path not found. Did you run index_knowledge_base.py?")
        return (
            "I can’t access the knowledge base yet. Please run indexing first.",
            'Follow-up: "Shall I explain how to build the knowledge base?"',
        )

    try:
        # 3) Build retriever
        embeddings = _build_embeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 4})

        # 4) Add a little context from the previous turn (if any)
        history = session_attributes.get("history", [])
        contextual_query = (
            f"Conversation History: {history[-1]}. NEW QUERY: {query}"
            if history else query
        )

        # 5) Retrieve
        retrieved_docs = _get_retrieved_docs(retriever, contextual_query)
        context = "\n\n---\n\n".join(
            getattr(doc, "page_content", str(doc)) for doc in retrieved_docs
        )

        # 6) LLM call
        client = Groq(api_key=GROQ_API_KEY)
        SYSTEM_PROMPT = """
            You are an AI Learning Tutor. 
            You MUST ALWAYS respond in the following exact format:

            1. Begin with a clear, simple explanation (3–6 sentences max).
            2. Then ALWAYS include a follow-up question on a new line.
            3. The follow-up line MUST follow this exact format:

            Follow-up: "<a question the learner should explore next>"

            If you do NOT include the Follow-up line in the exact format shown above,
            the response will be considered INVALID.

            Do NOT include citations or web references.
            Do NOT answer too academically — explain like teaching a beginner.
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nUser Query: {contextual_query}",
            },
        ]

        response = generate_response_with_retry(client, messages, GENERATOR_MODEL)
        raw = response.choices[0].message.content.strip()

        # 7) Parse & clean
        if "Follow-up:" in raw:
            head, tail = raw.split("Follow-up:", maxsplit=1)
            answer_text = " ".join(head.split())
            follow_up_text = "Follow-up: " + " ".join(tail.split()).strip()
        else:
            answer_text = " ".join(raw.split())
            follow_up_text = 'Follow-up: "Would you like to explore another related topic?"'

        # 8) Update conversation memory
        history.append(query)
        session_attributes["history"] = history

        return answer_text, follow_up_text

    except Exception as e:
        logger.error(f"RAG Function Failed: {e}", exc_info=True)
        return (
            "I am experiencing technical difficulties reaching the tutor's brain. Please try again.",
            'Follow-up: "Would you like to retry or ask another topic?"',
        )


# -----------------------------------------------------------------------------
# ALEXA HANDLERS
# -----------------------------------------------------------------------------
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
        query = (
            f"Teach me about {topic_slot.value}"
            if topic_slot and topic_slot.value
            else "What is Artificial Intelligence?"
        )

        attributes_manager = handler_input.attributes_manager
        session_attributes = attributes_manager.session_attributes or {}

        answer_text, follow_up_question = get_rag_response(query, session_attributes)

        # Save for context and multi-turn memory
        session_attributes["last_topic"] = query
        session_attributes["next_question"] = extract_next_question(follow_up_question)

        # IMPORTANT: write back explicitly so it's available on the next turn
        attributes_manager.session_attributes = session_attributes

        speech_output = f"{answer_text} {follow_up_question}"
        handler_input.response_builder.speak(speech_output).ask(follow_up_question)
        return handler_input.response_builder.response


class YesIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("AMAZON.YesIntent")(handler_input)

    def handle(self, handler_input):
        attributes_manager = handler_input.attributes_manager
        session_attributes = attributes_manager.session_attributes or {}

        next_question = session_attributes.get("next_question")
        if not next_question:
            speech_text = "Let's pick up a new topic! What would you like to learn next?"
            handler_input.response_builder.speak(speech_text).ask(speech_text)
            return handler_input.response_builder.response

        answer_text, follow_up_question = get_rag_response(
            next_question, session_attributes
        )

        # Update for the next turn and write back
        session_attributes["next_question"] = extract_next_question(follow_up_question)
        attributes_manager.session_attributes = session_attributes

        speech_output = f"Great! {answer_text} {follow_up_question}"
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


# -----------------------------------------------------------------------------
# FINAL GLOBAL SKILL BUILDER OBJECT (imported by app.py)
# -----------------------------------------------------------------------------
sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(TeachMeIntentHandler())
sb.add_request_handler(YesIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_exception_handler(CatchAllExceptionHandler())
