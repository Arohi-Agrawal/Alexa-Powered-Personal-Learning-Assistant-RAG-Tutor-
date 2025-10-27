# Alexa-Powered-Personal-Learning-Assistant-RAG-Tutor-

This project transforms Amazon Alexa into an AI-powered personal learning tutor.
Instead of answering from the internet, Alexa uses a custom curated knowledge base and a Retrieval-Augmented Generation (RAG) pipeline to teach concepts step-by-step.

The tutor explains topics, keeps context, and also suggests logical follow-up questions to guide learning continuously.

---

## 🚀 Features
- Answers are based **only** on your curated educational dataset
- Uses **ChromaDB** for semantic retrieval
- Uses **Groq LLaMA 3.1** for clean and concise responses
- Supports **multi-turn intelligent follow-up conversations**
- Fully integrated with **Alexa Skills Kit (ASK)**

## 📚 Dataset Used

| File Name | Topic |
|----------|--------|
| 01_ai_intro_history.txt | Intro to AI & Evolution |
| 02_data_preprocessing.txt | Data Cleaning & Preprocessing |
| 03_supervised_learning.txt | Regression & Classification |
| 04_unsupervised_learning.txt | Clustering & Dimensionality Reduction |
| 05_evaluation_metrics.txt | Accuracy, Precision, Recall, F1 Score |
| 06_neural_nets_basics.txt | Perceptron, Activation Functions |
| 07_deep_learning_arch.txt | CNNs, RNNs, LSTMs |
| 08_nlp_embeddings.txt | Word Embeddings & Representation |
| 09_transformer_llms.txt | Transformers & LLM Behavior |
| 10_generative_rag.txt | Generative AI & RAG Concepts |

---

## 🧠 RAG System Architecture
User → Alexa → Flask Server (/alexa)
→ RAG Engine (rag_engine.py)
→ ChromaDB (Semantic Retrieval)
→ Groq LLaMA 3.1 (Answer Generation)
→ Alexa (Voice Output + Follow-Up Prompt)

---

## 🛠️ Technologies Used

| Component | Technology |
|----------|------------|
| Programming Language | Python |
| Embedding Model | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector Database | **ChromaDB** |
| LLM Provider | **Groq API (LLaMA 3.1)** |
| Backend Server | Flask |
| Tunneling | ngrok |
| Voice Interface | Alexa Skills Kit (ASK) |

---


## ⚙️ Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv .venv


### 2. Allow PowerShell Execution (Windows Only)
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass


### 3. Activate Virtual Environment
```bash
.venv\Scripts\Activate.ps1


### 4. Install Dependencies
```bash
pip install -r requirements.txt



### Build the Knowledge Base (Indexing)
```bash
python index_knowledge_base.py
This will generate the chroma_db/ vector storage.


### Test the RAG Engine
```bash
python rag_engine.py
This ensures embeddings + retrieval + generation work correctly.


### Alexa Webhook Server Setup
```bash
Install Flask + Alexa WebService SDK
pip install flask ask-sdk-webservice-support


### Run the Flask Server
```bash
python app.py

### Open a second terminal & start ngrok
```bash
.\ngrok.exe http 5000


### Copy the generated HTTPS URL
```bash
https://YOUR-NGROK-URL.ngrok-free.app/alexa



🧪 Testing in Alexa Simulator

Go to Alexa Developer Console → Test (Development)

Say:

open ai tutor
teach me about supervised learning
yes


🎯 You should now get guided multi-turn learning responses.
---

##  Author

**Arohi Agrawal**  
B.Tech (Computer Science) — Mini Project  
Year: 2025  

---

