from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# ---- LangChain / OpenAI ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# Modern Runnable-based workflow (no deprecated `chains` dependency)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_community.embeddings.fake import FakeEmbeddings

app = Flask(__name__)
CORS(app)

# -------------------- CONFIG --------------------
PDF_PATH = "sample.pdf"  # Optional local reference document
api_key = os.getenv("OPENAI_API_KEY")
use_fake = not bool(api_key)  # Offline mock mode if no API key
EMBED_DIM = 1536

# -------------------- DOCUMENT PREPARATION --------------------
# 1) Try loading from sample.pdf, otherwise use safe mock documents
if os.path.exists(PDF_PATH):
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
else:
    docs = [
        Document(page_content="""
        Legal Aid BC provides free legal help for low-income individuals in British Columbia.
        Access Pro Bono offers volunteer-based legal services across BC.
        People's Law School provides free legal education and information online.
        """),
        Document(page_content="""
        BC 211 gives information and referral to community, government, and social services in BC.
        Dial 211 or visit bc211.ca.
        """),
    ]

# 2) Split documents into manageable chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Create embeddings (FakeEmbeddings for offline mock mode)
embeddings = FakeEmbeddings(size=EMBED_DIM) if use_fake else OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------- PROMPT & MODEL SETUP --------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant for residents of British Columbia, Canada.\n"
    "Always direct users to relevant institutions/resources (include names and contact details), "
    "and never provide legal advice.\n\n"
    "----context:\n{context}\n"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# LLM setup: skip model instantiation if running offline (no API key)
llm = None if use_fake else ChatOpenAI(model="gpt-4o-mini", temperature=0)

def join_docs(docs_list):
    """Combine multiple document chunks into a single context string."""
    return "\n\n".join([d.page_content for d in docs_list])

# Runnable chain: real LLM when online, mock output when offline
if use_fake:
    def fake_chain(question: str) -> str:
        ctx = join_docs(retriever.invoke(question))
        return f"(Simulated) Based on context:\n{ctx}\n\nYour question: {question}"
    chain = fake_chain
else:
    chain = (
        {
            "context": retriever | RunnableLambda(join_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    """Serve frontend HTML."""
    return render_template("index.html", use_fake=use_fake)

@app.route("/get_response", methods=["POST"])
def get_response():
    """Main chatbot endpoint: receives user question, returns AI (or simulated) answer."""
    data = request.get_json(force=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    if use_fake:
        answer = chain(question)
    else:
        answer = chain.invoke(question)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    # Development server (do not use in production)
    app.run(host="127.0.0.1", port=5000, debug=True)
