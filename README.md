# 🤖 AI Python Chatbot

> ⚠️ **Confidential Client Project (Rewritten Version)**  
> This repository is a **rewrite and refactor** of a confidential client chatbot project.  
> All proprietary data, documents, and integrations have been removed or replaced with mock examples.  
> The codebase now serves as a clean, educational demonstration of AI-assisted retrieval and conversation using Python.

---

## 💬 Overview

This project is a **lightweight AI chatbot** built with **Flask** and **LangChain**, designed to demonstrate a simple yet realistic **retrieval-augmented generation (RAG)** workflow.

It runs entirely **locally (offline mock mode)** or connects to **OpenAI’s API** if a valid key is provided.  
A clean HTML + JS frontend allows real-time chat interaction through a browser.

---

## ✨ Key Features

- 🧠 **LangChain-based RAG pipeline**  
  Retrieves and synthesizes document-based context dynamically.

- ⚙️ **Dual operation modes**  
  - **Online:** Uses OpenAI Embeddings and GPT-based chat models  
  - **Offline (mock mode):** Uses `FakeEmbeddings` for safe, API-free demo operation

- 💬 **Interactive frontend**  
  Simple and responsive HTML/JavaScript chat interface built without frameworks.

- 📄 **Document retrieval**  
  Loads and indexes text or PDF documents (via Chroma + LangChain loaders).

- 🔒 **Confidential rewrite**  
  All client content and data replaced with generic, safe mock examples.

---

## 🧱 Project Structure

aichatbot/
├─ app.py # Flask backend (retrieval logic + API routes)
├─ requirements.txt # Dependency list
├─ templates/
│ └─ index.html # Frontend chat interface
└─ static/
└─ style.css # Chat UI styling


---

## 🚀 Getting Started

### 1️⃣ Clone this repository

```bash
git clone https://github.com/LiuyiLi2000/Ai-Python-Chatbot.git
cd Ai-Python-Chatbot

2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ (Optional) Add your OpenAI API key
setx OPENAI_API_KEY "sk-your-api-key"
# Restart terminal after setting


Without an API key, the chatbot runs in mock mode, generating simulated answers using local embeddings.

5️⃣ Run the Flask app
python app.py


Then open your browser at 👉 http://127.0.0.1:5000

💻 Example Questions

“Where can I get free legal help in British Columbia?”

“What is Legal Aid BC?”

“Show me organizations offering community support.”

⚙️ Tech Stack
Layer	Technologies
Backend	Flask, Python
AI / RAG	LangChain, Chroma
LLM	OpenAI GPT (or FakeEmbeddings offline mode)
Frontend	HTML, CSS, JavaScript
Data Handling	PyPDFLoader, CharacterTextSplitter
🧩 Modes of Operation
Mode	Description
Live Mode	Uses OpenAI API for embeddings & chat completions
Mock Mode	Works completely offline using FakeEmbeddings and simulated responses
🔧 Future Improvements

Add user-uploaded PDF ingestion

Support persistent chat sessions

Extend frontend with React or Streamlit

Integrate authentication for production use
