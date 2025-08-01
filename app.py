import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import imaplib
import email
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import json
import shutil
import pandas as pd

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

# Extract council from URL or default
query_params = st.query_params
council_key = query_params.get("council", ["wyndham"])[0].lower().replace(" ", "_")
council = council_key.title().replace("_", " ")
config = COUNCIL_CONFIG.get(council_key, {})

# Page setup
st.set_page_config(page_title=f"CivReply AI - {council}", page_icon="\U0001F3DB️", layout="centered")

# --- State Management ---
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
if "user_auth" not in st.session_state:
    st.session_state.user_auth = False

# --- Admin Access ---
with st.expander("🔐 Admin Login"):
    if not st.session_state.is_admin:
        admin_input = st.text_input("Enter Admin Password", type="password")
        if admin_input == ADMIN_PASSWORD:
            st.session_state.is_admin = True
            st.success("✅ Admin access granted.")
        elif admin_input:
            st.error("❌ Incorrect password")

# --- User Login ---
with st.expander("👤 Council User Login"):
    if not st.session_state.user_auth:
        email_input = st.text_input("Enter your work email")
        if st.button("Login"):
            if email_input.endswith(f"@{council_key}.gov.au"):
                st.session_state.user_auth = True
                st.success("✅ Logged in as council user")
            else:
                st.error("❌ Invalid email domain for council")

# --- Upload Interface for Admins ---
if st.session_state.is_admin:
    st.markdown("### 📤 Upload Council PDFs")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        save_dir = f"docs/{council_key}"
        index_dir = f"index/{council_key}"
        os.makedirs(save_dir, exist_ok=True)

        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)

        for file in uploaded_files:
            with open(os.path.join(save_dir, file.name), "wb") as f:
                f.write(file.getbuffer())

        loader = PyPDFDirectoryLoader(save_dir)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local(index_dir)

        st.success(f"✅ Uploaded and indexed {len(uploaded_files)} file(s) for {council}")

        st.markdown("### 🗂️ Uploaded Files")
        for filename in os.listdir(save_dir):
            st.write(filename)
            if st.button(f"🗑 Delete {filename}", key=f"delete_{filename}"):
                os.remove(os.path.join(save_dir, filename))
                st.success(f"Deleted {filename}")

# --- Email Sending ---
def send_auto_email(recipient, question, answer):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = MIMEText(f"""
    <html><body>
      <h3 style='color:#3b82f6;'>Your CivReply AI Answer</h3>
      <p><strong>Question:</strong> {question}</p>
      <p><strong>Answer:</strong><br>{answer}</p>
      <p style='color:#6b7280;'>Sent at {now} from CivReply AI</p>
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

# --- Title + Branding ---
st.markdown(f"""
    <div style='text-align:center'>
      <h1>\U0001F3DB️ CivReply AI – {council}</h1>
      <img src="{config.get('hero_image')}" width="200" />
      <p><em>{config.get('tagline')}</em></p>
    </div>
""", unsafe_allow_html=True)

plan = config.get("plan", "basic")
limits = {
    "basic": {"queries": 500, "users": 1},
    "standard": {"queries": 2000, "users": 5},
    "enterprise": {"queries": float("inf"), "users": 20},
}[plan]

st.markdown(f"""
    <div style='background:#f1f5f9; padding:10px; border-left: 5px solid #3b82f6;'>
    💼 Plan: <strong>{plan.capitalize()}</strong> | {limits['queries']} queries/mo | {limits['users']} seats
    </div>
""", unsafe_allow_html=True)

st.info(config.get("about", "This council uses CivReply AI to assist residents with smarter answers."))

st.markdown("### 🔍 Ask a local question:")
user_question = st.text_input("Your question", placeholder="e.g., What day is bin collection?")
user_email = st.text_input("Your email (optional)", placeholder="your@email.com")

if user_question:
    if st.session_state.query_count[council_key] < limits["queries"]:
        st.session_state.query_count[council_key] += 1
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        index_path = f"index/{council_key}/index.faiss"
        retriever = FAISS.load_local(index_path, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)).as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        answer = qa_chain.run(user_question)
        st.markdown(f"**Answer:** {answer}")

        if user_email:
            if send_auto_email(user_email, user_question, answer):
                st.success("✅ Answer sent to your email.")

        st.markdown("### 🙋 Feedback")
        feedback = st.radio("Was this helpful?", ["👍", "👎"], key=f"fb_{st.session_state.query_count[council_key]}")
        comment = st.text_input("Any notes?", key=f"note_{st.session_state.query_count[council_key]}")
        if st.button("Submit Feedback"):
            st.session_state.feedback[council_key].append({
                "question": user_question,
                "answer": answer,
                "feedback": feedback,
                "comment": comment,
                "time": datetime.now().isoformat()
            })
            st.success("🙏 Thanks for your feedback!")

    else:
        st.error("You’ve reached the max query limit for this council’s plan.")

if st.session_state.get("is_admin"):
    if st.session_state.email_log[council_key]:
        st.markdown("### 📬 Email Log")
        df = pd.DataFrame(st.session_state.email_log[council_key])
        st.dataframe(df)
    if st.session_state.feedback[council_key]:
        st.markdown("### 📣 Feedback Log")
        df = pd.DataFrame(st.session_state.feedback[council_key])
        st.dataframe(df)

    st.markdown("### 🧭 Council Audit Dashboard")
    st.write("- Total Queries:", st.session_state.query_count[council_key])
    st.write("- Total Emails Sent:", len(st.session_state.email_log[council_key]))
    st.write("- Total Feedback Received:", len(st.session_state.feedback[council_key]))
