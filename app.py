# CivReply AI – Upgraded Version with Admin Upload, Multi-Council Support, Stripe, and Gmail Auto-Reply

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

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STRIPE_LINK = os.getenv("STRIPE_LINK", "https://buy.stripe.com/test_xxx")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "supersecret")
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")

st.set_page_config(page_title="CivReply AI", page_icon="\U0001F3DB️", layout="centered")

# --- Admin Auth ---
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if not st.session_state.is_admin:
    password = st.text_input("Enter Admin Password to Enable Upload", type="password")
    if password == ADMIN_PASSWORD:
        st.session_state.is_admin = True
        st.success("✅ Admin access granted.")
    elif password:
        st.error("❌ Incorrect password")

# --- Council Selector ---
councils = [
    "Wyndham", "Brimbank", "Hobsons Bay", "Melbourne", "Yarra",
    "Moreland", "Darebin", "Boroondara", "Stonnington", "Port Phillip"
]
council = st.selectbox("Choose Council", councils)
council_key = council.lower().replace(" ", "_")
index_path = f"index/{council_key}"
data_path = f"data/{council_key}"
os.makedirs(data_path, exist_ok=True)

# --- Branding ---
st.markdown(f"""
<style>
  body {{ font-family: 'Segoe UI', sans-serif; }}
  .header {{ display: flex; justify-content: center; gap: 12px; margin-bottom: 10px; }}
  .header h1 {{ font-size: 2.5rem; margin: 0; }}
  .tagline {{ text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 20px; }}
  .user-info-bar, .plan-box {{ background-color: #eef6ff; padding: 10px 15px; border-radius: 12px; margin-bottom: 20px; }}
  .question-label {{ font-size: 1rem; color: #374151; margin-bottom: 6px; }}
  .footer {{ text-align: center; font-size: 0.85rem; color: #6b7280; margin-top: 30px; }}
</style>
<div class="header">
  <div style="font-size: 2rem;">\U0001F3DB️</div>
  <h1>CivReply AI</h1>
</div>
<div class="tagline">Ask {council} Council anything – policies, laws, documents.</div>
<div class="user-info-bar">\U0001F9D1 Council: {council} | 🔐 Role: {'Admin' if st.session_state.is_admin else 'Guest'}</div>
<div class="plan-box">\U0001F4E6 Plan: Basic – 500 queries/month | 1 user | <a href='{STRIPE_LINK}' target='_blank'>Upgrade →</a></div>
<div class="question-label">🔍 Ask a local question:</div>
""", unsafe_allow_html=True)

# --- Input UI ---
question = st.text_input("e.g. Do I need a permit to cut down a tree?", key="question_box", label_visibility="collapsed")

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# --- Load Vector Index ---
try:
    db = FAISS.load_local(index_path, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY),
        retriever=retriever,
        return_source_documents=True
    )
except Exception as e:
    st.error(f"❌ Could not load index for {council}: {str(e)}")
    st.stop()

# --- Handle Question ---
if question:
    st.session_state.query_count += 1
    st.write("🔎 Searching documents...")
    try:
        result = qa_chain({"query": question})
        st.success(result["result"])
        with st.expander("📄 View sources"):
            for doc in result["source_documents"]:
                st.caption(f"• {doc.metadata.get('source', 'Unknown source')}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# --- PDF Upload + FAISS Rebuild ---
def process_and_index_directory(data_dir):
    loader = PyPDFDirectoryLoader(data_dir)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    faiss_index = FAISS.from_documents(split_docs, embeddings)
    faiss_index.save_local(index_path)

if st.session_state.is_admin:
    uploaded_files = st.file_uploader(f"📤 Upload PDFs for {council}", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(data_path, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success("✅ Files uploaded. Now click below to rebuild the index.")

    if st.button("🔄 Rebuild Index"):
        try:
            process_and_index_directory(data_path)
            st.success(f"✅ Index for {council} rebuilt.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"❌ Failed to rebuild index: {str(e)}")

# --- Gmail Auto-Reply with GPT ---
def gmail_auto_reply():
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(GMAIL_USER, GMAIL_PASS)
        imap.select("inbox")
        status, messages = imap.search(None, 'UNSEEN')

        for num in messages[0].split():
            _, data = imap.fetch(num, "(RFC822)")
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)
            sender = email.utils.parseaddr(msg['From'])[1]
            subject = msg['Subject']
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = msg.get_payload(decode=True).decode()

            prompt = f"You are an AI assistant replying to a council inquiry email. The email says: '{body}'. Write a professional and helpful reply."
            reply = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY).invoke(prompt)

            reply_msg = MIMEText(reply)
            reply_msg["Subject"] = f"Re: {subject}"
            reply_msg["From"] = GMAIL_USER
            reply_msg["To"] = sender

            smtp = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            smtp.login(GMAIL_USER, GMAIL_PASS)
            smtp.sendmail(GMAIL_USER, sender, reply_msg.as_string())
            smtp.quit()

        imap.logout()
        st.success("✅ Auto-replied to all unread emails.")
    except Exception as e:
        st.error(f"❌ Gmail auto-reply failed: {str(e)}")

if st.session_state.is_admin and st.button("📬 Auto-Reply to Council Emails"):
    gmail_auto_reply()

# --- Footer ---
st.markdown(f"""
<div class="footer">
  ⚙️ Powered by LangChain + GPT-4 | Queries used: {st.session_state.query_count} / 500<br>
  📬 Contact: <a href=\"mailto:contact@civreply.ai\">contact@civreply.ai</a>
</div>
""", unsafe_allow_html=True)
