import streamlit as st
from pathlib import Path
import os
import yagmail
from dotenv import load_dotenv

# --- AI / PDF IMPORTS ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA   # <-- ✅ NEW: Import RetrievalQA here!

# --- CONFIG ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_EMAIL = "civreplywyndham@gmail.com"
YAGMAIL_PASS = os.getenv("YAGMAIL_PASS")  # Set this as an ENV VAR on Render, not in .env in code
CIVREPLY_ADMIN_PASS = os.getenv("CIVREPLY_ADMIN_PASS", "admin123")
COUNCIL = "Wyndham"
LANG = "English"

st.set_page_config("CivReply AI", layout="wide", page_icon="🏛️")

# --- SESSION STATE ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'plan' not in st.session_state:
    st.session_state.plan = "Basic ($499 AUD/mo)"
if 'role' not in st.session_state:
    st.session_state.role = None
if 'pdf_index' not in st.session_state:
    st.session_state.pdf_index = None

# --- THEME COLORS ---
BG = "#F7FAFC"
HEADER_BG = "linear-gradient(90deg, #48bbff 0%, #1899D6 100%)"
ACCENT = "#2295d4"

# --- HEADER ---
def header():
    st.markdown(
        f"""
        <div style='background: {HEADER_BG}; border-radius:32px; padding:24px 12px 16px 32px; margin-bottom:12px;'>
            <span style="font-size:60px;vertical-align:middle">🏛️</span>
            <span style="font-size:46px;font-weight:bold;color:white;vertical-align:middle;margin-left:16px;">
                CivReply AI
            </span>
        </div>
        """, unsafe_allow_html=True
    )

def welcome():
    st.markdown(
        f"""
        <div style='background: #e6f7fd; border-radius:24px; padding:24px 18px 20px 24px; margin-bottom:20px;'>
            <span style='font-size:28px;font-weight:700;color:#2295d4'>👋 Welcome!</span>
            <span style='font-size:21px;color:#333; font-weight:500;margin-left:7px;'>
            CivReply AI helps you find answers, policies, and services from {COUNCIL} Council instantly.<br>
            <span style='font-size:16px;color:#177bb1'>Try asking about rubbish collection, local laws, grants, rates, events and more!</span>
            </span>
        </div>
        """, unsafe_allow_html=True
    )

def council_status():
    st.markdown(
        f"""
        <div style="background:#f3fafd;border-radius:14px;padding:9px 16px 7px 16px;display:flex;align-items:center;justify-content:left;font-size:16px;margin-bottom:10px;">
            <b>🏛️ Active Council:</b>&nbsp;{COUNCIL}
            <span style="margin-left:32px"><b>💼 Plan:</b> {st.session_state.plan}</span>
            <span style="margin-left:32px"><b>🌐 Language:</b> {LANG}</span>
        </div>
        """, unsafe_allow_html=True
    )

def sidebar():
    with st.sidebar:
        # Logo (safe fallback)
        import os
        if os.path.exists("logo.png"):
            st.image("logo.png", width=170)
        else:
            st.write("🏛️ CivReply AI")
        
        st.markdown(
            """
            <div style="font-size:22px;font-weight:bold;margin-top:-10px;margin-bottom:8px;color:#1899D6">
            CivReply AI
            </div>
            """, unsafe_allow_html=True
        )
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
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.button("What day is my rubbish collected?", use_container_width=True, key="q1")
    with col2: st.button("How do I apply for a pet registration?", use_container_width=True, key="q2")
    with col3: st.button("What are the rules for backyard sheds?", use_container_width=True, key="q3")
    with col4: st.button("Where can I find local events?", use_container_width=True, key="q4")
    with col5: st.button("How do I pay my rates online?", use_container_width=True, key="q5")
    st.markdown(
        "<button style='font-size:14px;border-radius:8px;background:#e3f4fc;padding:4px 18px;border:0;margin-top:4px;'>How does CivReply AI work?</button>",
        unsafe_allow_html=True
    )

# --- PDF INDEXING/AI ---
def build_pdf_index(pdf_dir: Path):
    loader = PyPDFDirectoryLoader(str(pdf_dir))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

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

def plan_selector():
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Basic ($499/mo)", use_container_width=True):
            st.session_state.plan = "Basic ($499 AUD/mo)"
    with col2:
        if st.button("Standard ($1,499/mo)", use_container_width=True):
            st.session_state.plan = "Standard ($1,499 AUD/mo)"
    with col3:
        if st.button("Enterprise ($2,999+/mo)", use_container_width=True):
            st.session_state.plan = "Enterprise ($2,999+ AUD/mo)"

header()
council_status()
sidebar_choice = sidebar()

# === Navigation routing ===
if sidebar_choice == "Chat with Council AI":
    welcome()
    try_asking()
    st.markdown("### 💬 Ask Wyndham Council")
    if st.session_state.pdf_index is None:
        st.warning("No PDFs indexed yet. Please upload council documents via the Admin Panel.")
    else:
        for sender, text in st.session_state.chat_history:
            st.markdown(
                f"<div style='background:#f4fafd;border-radius:10px;margin-bottom:8px;padding:8px 14px'><b>{sender}:</b> {text}</div>",
                unsafe_allow_html=True
            )
    q = st.text_input("Ask a question about Wyndham policies, forms, or documents...", key="ask_box")
    if st.button("Send", key="sendbtn") and q:
        st.session_state.chat_history.append(("You", q))
        # --- Real AI response from PDF index ---
        if st.session_state.pdf_index is not None:
            reply = ai_qa(q, st.session_state.pdf_index)
        else:
            reply = "The AI doesn't have any council documents yet."
        st.session_state.chat_history.append(("CivReply AI", reply))
        st.experimental_rerun()

elif sidebar_choice == "Submit a Request":
    st.header("Submit a Request")
    with st.form("request_form"):
        req_msg = st.text_area("Describe your request or report an issue.")
        req_email = st.text_input("Your email for follow-up (optional)")
        sent = st.form_submit_button("Submit")
        if sent and req_msg:
            sent_ok = send_email("CivReply Request", f"{req_msg}\nFrom: {req_email or 'N/A'}")
            if sent_ok:
                st.success("Your request was sent to the council team. Thank you!")

elif sidebar_choice == "Stats & Session":
    st.header("Session Stats & Usage")
    st.write(f"Plan: {st.session_state.plan}")
    st.write(f"Questions asked: {len(st.session_state.chat_history)}")
    st.json(st.session_state.chat_history)

elif sidebar_choice == "Share Feedback":
    st.header("Share Feedback")
    with st.form("fbform"):
        fb_msg = st.text_area("Your feedback")
        fb_email = st.text_input("Your email (optional)")
        sent = st.form_submit_button("Send Feedback")
        if sent and fb_msg:
            sent_ok = send_email("CivReply Feedback", f"{fb_msg}\nFrom: {fb_email or 'N/A'}")
            if sent_ok:
                st.success("Thanks for your feedback!")

elif sidebar_choice == "Contact Us":
    st.header("Contact Us")
    with st.form("contactform"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        msg = st.text_area("Your Message")
        sent = st.form_submit_button("Send Message")
        if sent and msg:
            sent_ok = send_email("CivReply Contact Us", f"{msg}\nFrom: {name} <{email}>")
            if sent_ok:
                st.success("Thank you for contacting us!")

elif sidebar_choice == "About Us":
    st.header("About CivReply AI")
    st.info("CivReply AI helps you find answers about council policies, local laws, services, events, and more using advanced AI and your local council's own documents.")

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
        st.write("Upload Council PDFs to index for AI Q&A.")
        pdf_dir = st.file_uploader("Upload multiple PDFs", accept_multiple_files=True, type="pdf")
        if pdf_dir:
            doc_dir = Path("council_docs")
            doc_dir.mkdir(exist_ok=True)
            for pdf in pdf_dir:
                with open(doc_dir / pdf.name, "wb") as f:
                    f.write(pdf.getbuffer())
            st.session_state.pdf_index = build_pdf_index(doc_dir)
            st.session_state.chat_history = []  # Clear chat on re-index
            st.success("PDFs indexed for AI Q&A! Return to Chat to try it out.")
        if st.button("Reset Session"):
            st.session_state.chat_history = []
            st.session_state.pdf_index = None
            st.success("Session reset.")

st.markdown("""
<br>
<div style='font-size:13px;text-align:center;color:#aaa'>Made with 🏛️ CivReply AI – for Australian councils, powered by AI</div>
""", unsafe_allow_html=True)
