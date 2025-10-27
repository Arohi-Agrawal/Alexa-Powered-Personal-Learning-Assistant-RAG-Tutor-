import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
# Set paths relative to the script location (D:\Project NLP)
DATA_DIR = "./knowledge_base"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 


def load_documents():
    """Loads all .txt files from the knowledge_base folder."""
    print("--- 1. Loading Documents (Extract) ---")
    
    # We use TextLoader because all your files are conveniently in .txt format.
    loader = DirectoryLoader(
        DATA_DIR, 
        glob="**/*.txt", 
        loader_cls=TextLoader
    )
    documents = loader.load()
    
    if not documents:
        print("No documents found! Check your 'knowledge_base' folder.")
        return []
    
    print(f"[LOAD COMPLETE]: Found {len(documents)} source files.")
    return documents


def split_documents(documents):
    """Breaks the loaded documents into small, context-rich chunks (Transform)."""
    print("--- 2. Splitting Documents (Transform) ---")
    
    # RecursiveCharacterTextSplitter is ideal for structured text like yours.
    # It attempts to split on new lines first, preserving semantic meaning.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # Max characters per chunk. 800 is a good starting point.
        chunk_overlap=80,    # 80 characters of overlap for context continuity.
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"[SPLIT COMPLETE]: Created {len(chunks)} searchable chunks.")
    return chunks


def save_to_chroma(chunks):
    """Generates embeddings and saves them to ChromaDB (Load)."""
    print("--- 3. Embedding and Storage (Load) ---")

    # ⚠️ OPTIONAL: If chroma_db already exists, delete it for a clean rebuild.
    if os.path.exists(CHROMA_PATH):
        print(f"Removing old database: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    
    # Initialize the embedding function (downloads the model if needed)
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Create the ChromaDB instance, which generates embeddings from the chunks and saves them.
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist() # Ensures data is written to disk

    print(f"\n[INDEXING COMPLETE]: The database is ready in the '{CHROMA_PATH}' folder.")


if __name__ == "__main__":
    if os.path.exists(DATA_DIR):
        all_documents = load_documents()
        if all_documents:
            all_chunks = split_documents(all_documents)
            save_to_chroma(all_chunks)
    else:
        print(f"FATAL ERROR: The data directory '{DATA_DIR}' does not exist.")


