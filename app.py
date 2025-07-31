# CivReply AI – Upgraded Version with Admin Login, Multi-Council Support, Stripe, and Gmail Auto-Reply
import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64
import imaplib
import email
import smtplib
from email.mime.text import MIMEText
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STRIPE_LINK = os.getenv("STRIPE_LINK", "https://buy.stripe.com/test_xxx")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "supersecret")
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")

st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="centered")

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
  <div style="font-size: 2rem;">🏛️</div>
  <h1>CivReply AI</h1>
</div>
<div class="tagline">Ask {council} Council anything – policies, laws, documents.</div>
<div class="user-info-bar">🧑 Council: {council} | 🔐 Role: {'Admin' if st.session_state.is_admin else 'Guest'}</div>
<div class="plan-box">📦 Plan: Basic – 500 queries/month | 1 user | <a href='{STRIPE_LINK}' target='_blank'>Upgrade →</a></div>
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
def process_and_index_pdf(uploaded_files):
    try:
        all_docs = []
        for file in uploaded_files:
            with open("temp.pdf", "wb") as f:
                f.write(file.read())
            loader = PyPDFLoader("temp.pdf")
            all_docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(all_docs)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        faiss_index = FAISS.from_documents(split_docs, embeddings)
        faiss_index.save_local(index_path)

        st.success(f"✅ Index for {council} updated with {len(uploaded_files)} file(s).")
        st.experimental_rerun()

    except Exception as e:
        st.error(f"❌ Failed to process: {str(e)}")

if st.session_state.is_admin:
    uploaded_files = st.file_uploader(f"📤 Upload PDFs for {council}", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("🔄 Rebuild Index"):
        process_and_index_pdf(uploaded_files)

# --- Gmail Auto-Reply ---
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
  📬 Contact: <a href="mailto:contact@civreply.ai">contact@civreply.ai</a>
</div>
""", unsafe_allow_html=True)
# --- Gmail Auto-Reply (Admin Only) ---
import imaplib
import smtplib
import email
from email.mime.text import MIMEText

def auto_reply_emails():
    try:
        imap_host = 'imap.gmail.com'
        smtp_host = 'smtp.gmail.com'
        smtp_port = 587

        email_user = os.getenv("GMAIL_USER")
        email_pass = os.getenv("GMAIL_PASS")

        if not email_user or not email_pass:
            st.warning("⚠️ Missing Gmail credentials in .env file.")
            return

        mail = imaplib.IMAP4_SSL(imap_host)
        mail.login(email_user, email_pass)
        mail.select("inbox")

        _, message_nums = mail.search(None, '(UNSEEN SUBJECT "CivReply")')
        message_nums = message_nums[0].split()

        if not message_nums:
            st.info("📭 No new council emails to reply to.")
            return

        for num in message_nums:
            _, data = mail.fetch(num, '(RFC822)')
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)
            from_email = email.utils.parseaddr(msg["From"])[1]
            subject = msg["Subject"]

            body = "Thank you for reaching out to CivReply AI. We will get back to you shortly.\n\nFor urgent requests, contact support@civreply.ai."

            reply = MIMEText(body)
            reply["Subject"] = f"RE: {subject}"
            reply["From"] = email_user
            reply["To"] = from_email

            server = smtplib.SMTP(smtp_host, smtp_port)
            server.starttls()
            server.login(email_user, email_pass)
            server.sendmail(email_user, from_email, reply.as_string())
            server.quit()

        st.success(f"📧 Replied to {len(message_nums)} new council email(s).")

    except Exception as e:
        st.error(f"❌ Failed to auto-reply: {str(e)}")


if st.session_state.is_admin:
    if st.button("📬 Auto-Reply to Council Emails"):
        auto_reply_emails()
