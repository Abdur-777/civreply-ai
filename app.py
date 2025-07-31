import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit page config
st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="centered")

# ---- UI Styling and Layout ----
st.markdown("""
<style>
  body {
    font-family: 'Segoe UI', sans-serif;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-bottom: 10px;
  }

  .header h1 {
    font-size: 2.5rem;
    margin: 0;
  }

  .tagline {
    text-align: center;
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 20px;
  }

  .user-info-bar {
    background-color: #f0f2f6;
    padding: 10px 15px;
    border-radius: 12px;
    font-size: 0.95rem;
    color: #333;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
  }

  .user-info-bar span {
    font-weight: 500;
  }

  .plan-box {
    background-color: #eef6ff;
    padding: 12px 15px;
    border-radius: 10px;
    margin: 10px 0 30px;
    font-size: 0.9rem;
    color: #1d4ed8;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .plan-box a {
    text-decoration: none;
    background-color: #1d4ed8;
    color: white;
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 0.85rem;
  }

  .question-box {
    background-color: #f9fafb;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 16px;
    font-size: 1rem;
    color: #111827;
    width: 100%;
  }

  .question-label {
    font-size: 1rem;
    margin-bottom: 6px;
    color: #374151;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .upload-note {
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: -6px;
    margin-bottom: 18px;
  }

  .footer {
    text-align: center;
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 30px;
  }
</style>

<div class="header">
  <div style="font-size: 2rem;">🏛️</div>
  <h1>CivReply AI</h1>
</div>

<div class="tagline">
  Ask Wyndham Council anything – policies, laws, documents.
</div>

<div class="user-info-bar">
  <div><span>🧑 Council:</span> Wyndham</div>
  <div><span>🔐 Role:</span> Admin</div>
</div>

<div class="plan-box">
  <div>📦 <strong>Plan:</strong> Basic – 500 queries/month | 1 user</div>
  <a href="#">Upgrade →</a>
</div>

<div class="question-label">🔍 Ask about a local policy or form</div>
""", unsafe_allow_html=True)

# ---- Input UI ----
question = st.text_input("e.g. Do I need a permit to cut down a tree?", key="question_box", label_visibility="collapsed")

# ---- Query Tracking ----
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ---- VectorStore + Retrieval Chain Setup ----
try:
    db = FAISS.load_local("index/wyndham", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY),
        retriever=retriever
    )
except Exception as e:
    st.error("❌ Failed to load documents or embeddings. Please check your index folder.")
    st.stop()

# ---- Handle Question ----
if question:
    st.session_state.query_count += 1
    st.write("🔎 Searching Wyndham Council documents...")
    try:
        answer = qa_chain.run(question)
        st.success(answer)
    except Exception as e:
        st.error("❌ Error processing your question. Check OpenAI key or vector DB.")

# ---- Admin Upload Placeholder ----
uploaded_file = st.file_uploader("Upload new Wyndham PDFs (Admin only)", type="pdf")
if uploaded_file:
    st.warning("📥 Upload feature is admin-only and currently not connected to retraining logic.")

# ---- Footer ----
st.markdown(f"""
<div class="upload-note">
  🔒 Upload new Wyndham PDFs (Admin only) – Only verified admins can upload documents.
</div>

<div class="footer">
  ⚙️ Powered by LangChain + OpenAI | Queries used: {st.session_state.query_count} / 500  
  <br>Contact: <a href="mailto:wyndham@vic.gov.au">wyndham@vic.gov.au</a>
</div>
""", unsafe_allow_html=True)
