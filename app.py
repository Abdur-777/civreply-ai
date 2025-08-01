# [EXTENDED CivReply AI Streamlit App – Version 2.3 – Role Refinement + Sidebar Threads + Tier UI]
# Updates: Removed 'Business Owner' role, added ChatGPT-style sidebar chat history, and tier card display

import streamlit as st
import os
import json
import base64
import shutil
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pathlib import Path
import time
import pandas as pd

# === APP CONFIG ===
st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="wide")

# === SETTINGS ===
COUNCILS = ["Wyndham"]
PLAN_CONFIG = {
    "basic": {"limit": 500, "label": "Basic ($499/mo)", "features": ["PDF Q&A"]},
    "standard": {"limit": 2000, "label": "Standard ($1499/mo)", "features": ["PDF Q&A", "Form Scraping"]},
    "enterprise": {"limit": float("inf"), "label": "Enterprise ($2999+/mo)", "features": ["All Features"]}
}

# === ENV VAR CHECK ===
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("❌ Missing OpenAI API Key. Please set `OPENAI_API_KEY` in your environment.")
    st.stop()

# === SESSION INIT ===
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("query_count", 0)
st.session_state.setdefault("user_role", "Resident")
st.session_state.setdefault("council", "Wyndham")
st.session_state.setdefault("session_start", datetime.now().isoformat())
st.session_state.setdefault("plan", "basic")

# === HEADER ===
st.markdown(f"""
    <div style='text-align: center;'>
        <h1>🏛️ CivReply AI</h1>
        <p>Smarter answers for smarter communities.</p>
        <p>🧾 Active Council: <strong>{st.session_state.council}</strong> &nbsp;|&nbsp; 💼 Plan: <strong>{PLAN_CONFIG[st.session_state.plan]['label']}</strong></p>
    </div>
""", unsafe_allow_html=True)

# === INPUT CONTROLS ===
cols = st.columns(2)
with cols[0]:
    st.session_state.user_role = st.selectbox("🧑 Role", ["Resident", "Staff", "Visitor"])

# === SIDEBAR ===
with st.sidebar:
    st.image("https://www.wyndham.vic.gov.au/sites/default/files/styles/small/public/2020-06/logo_0.png", width=200)
    st.title("CivReply AI")
    nav = st.radio("📚 Menu", [
        "💬 Chat with Council AI",
        "🧾 Topic FAQs",
        "📥 Submit a Request",
        "📊 Stats & Session",
        "📤 Export Logs",
        "💡 Share Feedback",
        "📞 Contact Us",
        "⚙️ Admin Panel"])

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 🧠 Recent Q&A")
        for i, (q, a) in enumerate(st.session_state.chat_history[::-1][:5]):
            if st.button(f"Q: {q[:40]}...", key=f"history_{i}"):
                st.session_state.chat_input = q

# === PLAN CARD DISPLAY ===
st.markdown("### 💼 Plan Comparison")
cols = st.columns(3)
for i, tier in enumerate(["basic", "standard", "enterprise"]):
    with cols[i]:
        st.markdown(f"""
        <div style='border:1px solid #ddd;padding:1rem;border-radius:10px;'>
            <h4>{PLAN_CONFIG[tier]['label']}</h4>
            <ul>
                {''.join([f'<li>{feat}</li>' for feat in PLAN_CONFIG[tier]['features']])}
            </ul>
            <p><strong>Limit:</strong> {PLAN_CONFIG[tier]['limit'] if PLAN_CONFIG[tier]['limit'] != float('inf') else 'Unlimited'} queries</p>
        </div>
        """, unsafe_allow_html=True)

# === FUNCTIONS ===
def ask_ai(question, council):
    try:
        embeddings = OpenAIEmbeddings()
        index_path = f"index/{council.lower()}"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index found for {council}")
        db = FAISS.load_local(index_path, embeddings)
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(api_key=OPENAI_KEY), chain_type="stuff", retriever=retriever)
        return qa.run(question)
    except Exception as e:
        return f"[Error] Could not answer: {str(e)}"

def export_logs():
    filename = f"chatlog_{st.session_state.session_start}.txt"
    with open(filename, "w") as f:
        for q, a in st.session_state.chat_history:
            f.write(f"Q: {q}\nA: {a}\n---\n")
    return filename

def log_feedback(text, email):
    with open("feedback_log.txt", "a") as f:
        f.write(f"{datetime.now().isoformat()} | {email or 'anon'} | {text}\n")

# === QUERY LIMIT CHECK ===
plan_id = st.session_state.plan
plan_limit = PLAN_CONFIG[plan_id]['limit']

if st.session_state.query_count >= plan_limit:
    st.warning(f"❗ You’ve reached the {PLAN_CONFIG[plan_id]['label']} limit.")
    st.stop()

# === MAIN ROUTER ===
if nav == "💬 Chat with Council AI":
    st.subheader("💬 Ask Wyndham Council")
    user_input = st.chat_input("Ask a question about Wyndham policies, forms, or documents…")
    if user_input:
        st.session_state.query_count += 1
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Wyndham GPT is replying..."):
            ai_reply = ask_ai(user_input, st.session_state.council)
            with st.chat_message("ai"):
                st.markdown(f"📩 **Auto-response from Wyndham Council:**\n\n{ai_reply}")
            st.session_state.chat_history.append((user_input, ai_reply))

elif nav == "🧾 Topic FAQs":
    st.subheader("FAQs by Topic")
    topics = {
        "Waste": ["How to report a missed bin?", "Where to get a new green bin?"],
        "Pets": ["Is dog registration required?", "Off-leash park locations?"],
        "Events": ["What’s on this weekend?", "How to book community rooms?"]
    }
    topic = st.selectbox("📂 Choose a category", list(topics.keys()))
    for q in topics[topic]:
        st.markdown(f"❓ {q}")

elif nav == "📥 Submit a Request":
    st.markdown("📌 Redirecting to your council’s website.")
    st.link_button("📝 Submit Online", "https://www.wyndham.vic.gov.au/request-it")

elif nav == "📊 Stats & Session":
    st.metric("Total Questions", st.session_state.query_count)
    st.metric("Session Start", st.session_state.session_start)
    st.metric("Role", st.session_state.user_role)
    st.metric("Council", st.session_state.council)
    st.metric("Plan", PLAN_CONFIG[st.session_state.plan]['label'])

elif nav == "📤 Export Logs":
    if st.button("📄 Download Session Log"):
        fname = export_logs()
        with open(fname, "rb") as f:
            st.download_button("📥 Download", f, file_name=fname)

elif nav == "💡 Share Feedback":
    fb = st.text_area("Tell us what’s working or not...")
    email = st.text_input("Email (optional)")
    if st.button("📨 Submit"):
        log_feedback(fb, email)
        st.success("Thanks for helping improve CivReply AI!")

elif nav == "📞 Contact Us":
    st.markdown("**Call:** (03) 9742 0777  ")
    st.markdown("**Visit:** 45 Princes Hwy, Werribee  ")
    st.markdown("**Mail:** PO Box 197")
    st.link_button("Website", "https://www.wyndham.vic.gov.au")

elif nav == "⚙️ Admin Panel":
    if st.session_state.user_role != "Staff":
        st.warning("Restricted to council staff only.")
    else:
        st.subheader("📂 Upload New Council Docs")
        docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Rebuild Index"):
            if docs:
                folder = f"docs/{st.session_state.council.lower()}"
                os.makedirs(folder, exist_ok=True)
                for d in docs:
                    with open(os.path.join(folder, d.name), "wb") as f:
                        f.write(d.read())
                loader = PyPDFDirectoryLoader(folder)
                raw_docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(raw_docs)
                vecdb = FAISS.from_documents(chunks, OpenAIEmbeddings())
                vecdb.save_local(f"index/{st.session_state.council.lower()}")
                st.success("✅ Index rebuilt successfully.")
            else:
                st.warning("Please upload at least one document.")
