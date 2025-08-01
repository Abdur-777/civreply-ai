import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from urllib.parse import urlparse
import json
import shutil

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STRIPE_LINK = os.getenv("STRIPE_LINK", "https://buy.stripe.com/test_xxx")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "supersecret")
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")

# Load council configs
with open("council_config.json") as f:
    COUNCIL_CONFIG = json.load(f)

# --- Extract URL parameters ---
query_params = st.query_params
council_key = query_params.get("council", ["wyndham"])[0].lower().replace(" ", "_")
council = council_key.title().replace("_", " ")
config = COUNCIL_CONFIG.get(council_key, {})

# --- Session state ---
st.set_page_config(page_title=f"CivReply AI – {council}", page_icon="🏛️", layout="wide")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = {}
if council_key not in st.session_state.query_count:
    st.session_state.query_count[council_key] = 0
if "email_log" not in st.session_state:
    st.session_state.email_log = {}
if council_key not in st.session_state.email_log:
    st.session_state.email_log[council_key] = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if council_key not in st.session_state.feedback:
    st.session_state.feedback[council_key] = []
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# --- Sidebar Chat History ---
st.sidebar.title("🧾 Chat History")
st.sidebar.button("+ New Chat", on_click=lambda: st.session_state.chat_history.clear())
for i, item in enumerate(reversed(st.session_state.chat_history)):
    st.sidebar.markdown(f"{item['time'][:16]}: **{item['question'][:30]}...**")

# --- Admin Panel ---
with st.sidebar.expander("🔐 Admin Login"):
    if not st.session_state.is_admin:
        if st.text_input("Admin Password", type="password") == ADMIN_PASSWORD:
            st.session_state.is_admin = True
            st.success("✅ Admin access granted")

if st.session_state.is_admin:
    st.sidebar.markdown("### 📤 Upload Council PDFs")
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        save_dir = f"docs/{council_key}"
        index_dir = f"index/{council_key}"
        os.makedirs(save_dir, exist_ok=True)
        for f in os.listdir(save_dir): os.remove(os.path.join(save_dir, f))
        if os.path.exists(index_dir): shutil.rmtree(index_dir)
        for file in uploaded_files:
            with open(os.path.join(save_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
        loader = PyPDFDirectoryLoader(save_dir)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(loader.load())
        db = FAISS.from_documents(split_docs, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
        db.save_local(index_dir)
        st.sidebar.success(f"✅ Uploaded and indexed {len(uploaded_files)} PDF(s)")

# --- Email Sender ---
def send_auto_email(recipient, question, answer):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = MIMEText(f"""
    <html><body>
      <h3>Your CivReply AI Answer</h3>
      <p><strong>Question:</strong> {question}</p>
      <p><strong>Answer:</strong><br>{answer}</p>
      <p style='font-size:12px;color:#888;'>Sent {now}</p>
    </body></html>
    """, "html")
    msg["Subject"] = f"CivReply Answer: {question[:50]}"
    msg["From"] = GMAIL_USER
    msg["To"] = recipient
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASS)
            server.send_message(msg)
        st.session_state.email_log[council_key].append({"to": recipient, "question": question, "time": now})
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# --- Feedback Handler ---
def record_feedback(rating, notes=""):
    st.session_state.feedback[council_key].append({
        "time": datetime.now().isoformat(),
        "rating": rating,
        "notes": notes
    })
    st.success("✅ Feedback submitted")

# --- Chat Display ---
st.markdown(f"""
    <style>
    .chat-bubble {{
      max-width: 80%; padding: 10px 15px; border-radius: 15px; margin-bottom: 10px;
    }}
    .user {{ background: #d1fae5; align-self: flex-end; margin-left:auto; }}
    .bot {{ background: #e0e7ff; align-self: flex-start; margin-right:auto; }}
    .chat-container {{ display: flex; flex-direction: column; gap: 10px; }}
    </style>
    <div style='text-align:center'>
      <h1>🏛️ CivReply AI – {council}</h1>
      <p><em>{config.get('tagline', 'Empowering local answers.')}</em></p>
    </div>
""", unsafe_allow_html=True)

# --- Plan Info ---
plan = config.get("plan", "basic")
limits = {"basic": {"queries": 500}, "standard": {"queries": 2000}, "enterprise": {"queries": float("inf")}}[plan]
st.markdown(f"**💼 Plan: {plan.capitalize()}** | {limits['queries']} queries/mo")

# --- Chat Input + History ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for item in st.session_state.chat_history:
    st.markdown(f"<div class='chat-bubble user'>{item['question']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble bot'>{item['answer']}</div>", unsafe_allow_html=True)
    with st.expander("💬 Feedback for this response"):
        emoji = st.radio("Rate this response:", ["👍", "👎", "😐"], key=f"rate_{item['time']}")
        note = st.text_area("Optional comment", key=f"note_{item['time']}")
        if st.button("Submit Feedback", key=f"btn_{item['time']}"):
            record_feedback(emoji, note)
st.markdown("</div>", unsafe_allow_html=True)

question = st.chat_input("Ask a local question")
user_email = st.text_input("Email for response (optional)")
if question and st.session_state.query_count[council_key] < limits['queries']:
    st.session_state.query_count[council_key] += 1
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    retriever = FAISS.load_local(f"index/{council_key}/index.faiss", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)).as_retriever()
    answer = RetrievalQA.from_chain_type(llm=llm, retriever=retriever).run(question)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.chat_history.append({"question": question, "answer": answer, "time": now})
    if user_email:
        send_auto_email(user_email, question, answer)
    st.rerun()
elif question:
    st.error("Query limit reached for this plan.")
