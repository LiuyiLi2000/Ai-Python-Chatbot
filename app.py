from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# ---- LangChain / OpenAI ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# Runnable 组合（不依赖过时 chains 模块）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_community.embeddings.fake import FakeEmbeddings

app = Flask(__name__)
CORS(app)

# -------------------- 配置 --------------------
PDF_PATH = "sample.pdf"  # 没有也没关系，会用假文档
api_key = os.getenv("OPENAI_API_KEY")
use_fake = not bool(api_key)  # 无 API Key 时自动进入 mock 模式（离线）
EMBED_DIM = 1536

# -------------------- 语料准备 --------------------
# 1) 文档来源：优先从 sample.pdf 读取；否则使用内置假文档（安全）
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

# 2) 切分
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) 向量化（无 API 时使用 FakeEmbeddings 完全离线）
embeddings = FakeEmbeddings(size=EMBED_DIM) if use_fake else OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------- Prompt & LLM --------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant for residents of British Columbia, Canada.\n"
    "Always redirect them to relevant institutions/resources (names and contact details) "
    "and never provide legal advice.\n\n"
    "----context:\n{context}\n"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# LLM：无 API Key 时不要实例化远端模型，直接返回“模拟回答”
llm = None if use_fake else ChatOpenAI(model="gpt-4o-mini", temperature=0)

def join_docs(docs_list):
    return "\n\n".join([d.page_content for d in docs_list])

# runnable 链（有 Key 时使用 LLM；无 Key 时用 mock）
if use_fake:
    # 纯本地 mock：检索文档 + 拼个模拟回答
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

# -------------------- 路由 --------------------
@app.route("/")
def home():
    # 前端页
    return render_template("index.html", use_fake=use_fake)

@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json(force=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    if use_fake:
        answer = chain(question)  # 本地 fake
    else:
        answer = chain.invoke(question)  # 真调用

    return jsonify({"answer": answer})

if __name__ == "__main__":
    # 生产请关掉 debug，用 WSGI/ASGI 部署
    app.run(host="127.0.0.1", port=5000, debug=True)

