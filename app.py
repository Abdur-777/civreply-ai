import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/etc/gcs_creds.json")

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

from google.cloud import storage
import yagmail

# --- CONFIG ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_EMAIL = "civreplywyndham@gmail.com"
YAGMAIL_PASS = os.getenv("YAGMAIL_PASS")
CIVREPLY_ADMIN_PASS = os.getenv("CIVREPLY_ADMIN_PASS", "admin123")
GCS_BUCKET = os.getenv("GCS_BUCKET")
COUNCILS = ["Wyndham", "Yarra", "Casey", "Melbourne"]
LANG = "English"

st.set_page_config("CivReply AI", layout="wide", page_icon="🏛️")

# --- GCS HELPERS ---
def upload_to_gcs(local_file, bucket_name, remote_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_file)

def download_from_gcs(bucket_name, remote_path, local_file):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(remote_path)
    if blob.exists():
        blob.download_to_filename(local_file)
        return True
    return False

# --- SESSION STATE ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'plan' not in st.session_state:
    st.session_state.plan = "Basic"
if 'role' not in st.session_state:
    st.session_state.role = None
if 'pdf_index' not in st.session_state:
    st.session_state.pdf_index = None
if 'active_council' not in st.session_state:
    st.session_state.active_council = COUNCILS[0]
if 'user_type' not in st.session_state:
    st.session_state.user_type = "Resident"

# --- HEADER ---
def header():
    st.markdown(
        """
        <div style='background: linear-gradient(90deg, #48bbff 0%, #1899D6 100%); border-radius:32px; padding:24px 12px 16px 32px; margin-bottom:12px;'>
            <span style="font-size:60px;vertical-align:middle">🏛️</span>
            <span style="font-size:46px;font-weight:bold;color:white;vertical-align:middle;margin-left:16px;">
                CivReply AI
            </span>
        </div>
        """, unsafe_allow_html=True
    )

def council_status():
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    col1.markdown(f"<b>🏛️ Active Council:</b> {st.session_state.active_council}", unsafe_allow_html=True)
    col2.markdown(f"<b>💼 Plan:</b> {st.session_state.plan}", unsafe_allow_html=True)
    user_types = ["Resident", "Visitor", "Staff"]
    st.session_state.user_type = col3.selectbox("Type", user_types, index=user_types.index(st.session_state.user_type), label_visibility="collapsed")
    col4.markdown(f"<b>🌐 Language:</b> {LANG}", unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="font-size:38px;line-height:1;margin-bottom:2px;">🏛️</div>
            <div style="font-size:22px;font-weight:bold;margin-top:-5px;margin-bottom:8px;color:#1899D6">
            CivReply AI
            </div>
            """, unsafe_allow_html=True
        )
        selected = st.selectbox("Council", COUNCILS, index=COUNCILS.index(st.session_state.active_council))
        if selected != st.session_state.active_council:
            st.session_state.active_council = selected
            st.session_state.pdf_index = None
            st.session_state.chat_history = []
            st.experimental_rerun()
        nav = st.radio("",
            [
                "Chat with Council AI",
                "Submit a Request",
                "Stats & Session",
                "Share Feedback",
                "Contact Us",
                "About Us",
                "Admin Panel"
            ],
            index=0, key="nav"
        )
        st.markdown("### &nbsp;\n#### Recent Chats")
        st.info("No chats yet")
        st.markdown("---")
        st.button("🚀 Upgrade Your Plan", use_container_width=True, key="upgrade")
    return nav

def try_asking():
    st.markdown(
        """
        <div style='margin-bottom:18px;'>
            <span style="font-size:18px;color:#2295d4;font-weight:600;">💡 Try asking:</span><br>
        </div>
        """, unsafe_allow_html=True
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.button("What day is my rubbish collected?", use_container_width=True, key="q1")
    with col2: st.button("How do I apply for a pet registration?", use_container_width=True, key="q2")
    with col3: st.button("What are the rules for backyard sheds?", use_container_width=True, key="q3")
    with col4: st.button("Where can I find local events?", use_container_width=True, key="q4")

# --- PDF INDEXING/AI ---
def build_pdf_index(pdf_dir: Path, faiss_index_path: str):
    loader = PyPDFDirectoryLoader(str(pdf_dir))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(faiss_index_path)
    return vectorstore

def load_faiss_index(faiss_index_path: str, embeddings):
    if os.path.exists(faiss_index_path):
        return FAISS.load_local(faiss_index_path, embeddings)
    return None

def ai_qa(question: str, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever,
        return_source_documents=False
    )
    resp = chain({"query": question})
    return resp.get("result", "No answer found.")

def send_email(subject, contents, to=ADMIN_EMAIL):
    try:
        yag = yagmail.SMTP(ADMIN_EMAIL, YAGMAIL_PASS)
        yag.send(to=to, subject=subject, contents=contents)
        return True
    except Exception as e:
        st.error(f"Email send failed: {e}")
        return False

header()
council_status()
sidebar_choice = sidebar()

def council_index_path(council): return f"faiss_indexes/{council.lower()}_index"
def council_pdf_dir(council): return f"pdfs/{council.lower()}"

active_council = st.session_state.active_council
faiss_path = f"index/{active_council.lower()}_index"
pdf_dir = Path(f"council_docs/{active_council}")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if st.session_state.pdf_index is None:
    faiss_gcs_path = council_index_path(active_council)
    faiss_local_path = f"{faiss_path}"
    if not os.path.exists(faiss_local_path) and GCS_BUCKET:
        download_from_gcs(GCS_BUCKET, faiss_gcs_path, faiss_local_path)
    if os.path.exists(faiss_local_path):
        st.session_state.pdf_index = load_faiss_index(faiss_local_path, embeddings)

if sidebar_choice == "Chat with Council AI":
    st.markdown("### 💬 Ask " + active_council + " Council")
    try_asking()
    if st.session_state.pdf_index is None:
        st.warning("No PDFs indexed yet. Please upload council documents via the Admin Panel.")
    else:
        for sender, text in st.session_state.chat_history:
            st.markdown(
                f"<div style='background:#f4fafd;border-radius:10px;margin-bottom:8px;padding:8px 14px'><b>{sender}:</b> {text}</div>",
                unsafe_allow_html=True
            )
    q = st.text_input("Ask a question about council policies, forms, or documents...", key="ask_box")
    if st.button("Send", key="sendbtn") and q:
        st.session_state.chat_history.append(("You", q))
        if st.session_state.pdf_index is not None:
            reply = ai_qa(q, st.session_state.pdf_index)
        else:
            reply = "The AI doesn't have any council documents yet."
        st.session_state.chat_history.append(("CivReply AI", reply))
        st.experimental_rerun()

elif sidebar_choice == "Admin Panel":
    st.header("Admin Panel")
    if st.session_state.role != "admin":
        pwd = st.text_input("Enter admin password", type="password")
        if st.button("Login as admin"):
            if pwd == CIVREPLY_ADMIN_PASS:
                st.session_state.role = "admin"
                st.success("Welcome, admin.")
                st.experimental_rerun()
            else:
                st.error("Incorrect password.")
    else:
        st.write(f"Upload Council PDFs for: {active_council}")
        uploaded_pdfs = st.file_uploader("Upload multiple PDFs", accept_multiple_files=True, type="pdf")
        if uploaded_pdfs:
            pdf_dir.mkdir(parents=True, exist_ok=True)
            for pdf in uploaded_pdfs:
                with open(pdf_dir / pdf.name, "wb") as f:
                    f.write(pdf.getbuffer())
                if GCS_BUCKET:
                    upload_to_gcs(str(pdf_dir / pdf.name), GCS_BUCKET, f"{council_pdf_dir(active_council)}/{pdf.name}")
            st.session_state.pdf_index = build_pdf_index(pdf_dir, faiss_path)
            if GCS_BUCKET:
                upload_to_gcs(faiss_path, GCS_BUCKET, council_index_path(active_council))
            st.success("PDFs indexed for AI Q&A! Return to Chat to try it out.")
        if st.button("Reset Session"):
            st.session_state.chat_history = []
            st.session_state.pdf_index = None
            st.success("Session reset.")

st.markdown("""
<br>
<div style='font-size:13px;text-align:center;color:#aaa'>Made with 🏛️ CivReply AI – for Australian councils, powered by AI</div>
""", unsafe_allow_html=True)
