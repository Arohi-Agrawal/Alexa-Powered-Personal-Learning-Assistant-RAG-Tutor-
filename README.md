# Alexa-Powered-Personal-Learning-Assistant-RAG-Tutor-


python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python index_knowledge_base.py

python rag_engine.py

pip install flask ask-sdk-webservice-support  

(.venv) PS D:\Project NLP> python app.py
PS D:\Project NLP> .\\ngrok.exe http 5000