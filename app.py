import os
import streamlit as st
from datetime import datetime
import pandas as pd
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ========================== CONFIGURATION ==========================
COUNCILS = [
    "Casey", "Wyndham", "Whittlesea", "Greater Dandenong", "Hume",
    "Melton", "Brimbank", "Melbourne", "Monash", "Boroondara"
]
DEFAULT_COUNCIL = COUNCILS[0]
ADMIN_EMAILS = ["admin@civreply.ai"]

APP_ABOUT = """
CivReply AI answers any question from your local council's official documents (PDFs). Upload, index, and manage your knowledge base instantly.
"""

def get_pdf_dir(council):
    return f"docs/{council.lower().replace(' ', '_')}"

def get_index_dir(council):
    return f"index/{council.lower().replace(' ', '_')}"

os.makedirs("docs", exist_ok=True)
os.makedirs("index", exist_ok=True)

# ========================== USER & ROLE MGMT ==========================
def get_user_role(email):
    if not email:
        return "Resident"
    if email.lower() in ADMIN_EMAILS:
        return "Admin"
    return "Staff" if email.endswith("@council.gov.au") else "Resident"

if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_role" not in st.session_state:
    st.session_state.user_role = "Resident"

with st.sidebar.expander("🔐 Sign In / Role Select", expanded=True):
    user_email = st.text_input("Your email (for advanced features, leave blank for Resident):", value=st.session_state.user_email)
    if st.button("Set Email"):
        st.session_state.user_email = user_email
        st.session_state.user_role = get_user_role(user_email)

role = st.session_state.get("user_role", "Resident")

# ========================== COUNCIL SELECTION ==========================
st.sidebar.title("CivReply AI")
council = st.sidebar.selectbox("Select your council", COUNCILS, index=COUNCILS.index(DEFAULT_COUNCIL))
pdf_dir = get_pdf_dir(council)
index_dir = get_index_dir(council)
os.makedirs(pdf_dir, exist_ok=True)

st.sidebar.markdown("---")
st.sidebar.info(f"Council Knowledgebase: `{council}`")

# ========================== SIDEBAR: TEAM & HELP ==========================
with st.sidebar.expander("👥 Team", expanded=False):
    st.write("Project Lead: Abdullah Arshad\nBackend: GPT-4\nContact: admin@civreply.ai")

with st.sidebar.expander("❓ Help & FAQ", expanded=False):
    st.write("• Upload PDFs (Staff/Admin)\n• Build index\n• Ask any question\n• Download your chat history")

# ========================== USAGE STATS ==========================
if "usage_stats" not in st.session_state:
    st.session_state.usage_stats = {}

def increment_usage(council):
    st.session_state.usage_stats.setdefault(council, 0)
    st.session_state.usage_stats[council] += 1

# ========================== PAGE HEADER ==========================
st.title("🏛️ CivReply AI")
st.caption(APP_ABOUT)
st.markdown("---")

# ========================== COUNCIL INFO BLOCK ==========================
COUNCIL_INFOS = {
    "Casey": "City of Casey, southeast of Melbourne CBD. Population 365,000+.",
    "Wyndham": "Wyndham is one of Australia's fastest-growing councils, in the west.",
    "Whittlesea": "Located north of Melbourne, large multicultural community.",
    "Greater Dandenong": "Vibrant, diverse, strong manufacturing history.",
    "Hume": "Home to the airport and key industrial precincts.",
    "Melton": "One of the most rapidly urbanizing areas.",
    "Brimbank": "Biggest municipality in Melbourne’s west.",
    "Melbourne": "CBD, university, commerce, and culture hub.",
    "Monash": "Known for Monash Uni, business, leafy suburbs.",
    "Boroondara": "Historic, leafy, high quality-of-life, in inner-east."
}
with st.expander("ℹ️ About this council", expanded=False):
    st.info(COUNCIL_INFOS.get(council, "No data. Contact admin to add council info!"))

# ========================== INDEX MANAGEMENT ==========================
def build_index(pdf_dir, index_dir):
    st.info("Building index, this may take a minute for large files...")
    loader = PyPDFDirectoryLoader(pdf_dir)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_dir)
    st.success("✅ Index built successfully!")
    return db

def get_index(index_dir):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    if not os.path.exists(index_dir):
        return None
    return FAISS.load_local(index_dir, embeddings)

def council_index_status(index_dir):
    if not os.path.exists(index_dir):
        return "❌ Index missing"
    if os.path.isdir(index_dir) and os.listdir(index_dir):
        return "✅ Index ready"
    return "⚠️ Index empty"

st.sidebar.markdown(f"**Index status:** {council_index_status(index_dir)}")

# ========================== ADMIN/STAFF PDF UPLOAD ==========================
with st.sidebar.expander("📥 Upload & Index PDFs (Admin/Staff)"):
    if role in ["Admin", "Staff"]:
        uploaded_pdfs = st.file_uploader("Upload new PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Add to Knowledgebase"):
            if uploaded_pdfs:
                for doc in uploaded_pdfs:
                    save_path = os.path.join(pdf_dir, doc.name)
                    with open(save_path, "wb") as f:
                        f.write(doc.read())
                st.success("PDFs uploaded! Click 'Rebuild Index' to include them.")
            else:
                st.warning("No files uploaded.")
        if st.button("Rebuild Index"):
            build_index(pdf_dir, index_dir)
    else:
        st.info("Sign in as Staff or Admin to upload PDFs and rebuild index.")

# ========================== INDEX AUTOLOAD ==========================
@st.cache_resource(show_spinner="Loading document index…")
def get_vectorstore(index_dir):
    return get_index(index_dir)

db = get_vectorstore(index_dir)
if db is None:
    st.warning(f"No index for `{council}`. Ask an Admin/Staff to upload & index PDFs.")
    st.stop()

# ========================== QA CHAIN ==========================
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0),
    retriever=db.as_retriever(),
)

# ========================== CHAT SESSION PER COUNCIL ==========================
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {c: [] for c in COUNCILS}

def get_current_chat():
    return st.session_state.chat_sessions[council]

def add_to_chat_history(q, a):
    st.session_state.chat_sessions[council].append({
        "timestamp": datetime.now().isoformat(),
        "council": council,
        "role": role,
        "user": st.session_state.user_email,
        "question": q,
        "answer": a
    })

# ========================== MAIN Q&A UI ==========================
st.header(f"💬 {council} Council Chatbot")
st.markdown("Ask anything about policies, local laws, plans, council forms, and more:")
user_query = st.text_input("Type your question here:")

if st.button("Ask AI") or user_query:
    with st.spinner("Thinking..."):
        ai_answer = qa({"query": user_query})["result"]
    increment_usage(council)
    add_to_chat_history(user_query, ai_answer)
    st.success(ai_answer)
    st.markdown("---")
    st.markdown("##### Recent Q&A")
    for chat in reversed(get_current_chat()[-3:]):
        st.markdown(f"**Q:** {chat['question']}\n\n**A:** {chat['answer']}")

# ========================== USAGE STATS ==========================
with st.expander("📊 Usage Stats"):
    usage = st.session_state.usage_stats.get(council, 0)
    st.metric("Questions asked for this council", usage)
    st.write(f"Chat history entries (this council): {len(get_current_chat())}")
    st.write("Total chats (all councils):", sum(len(s) for s in st.session_state.chat_sessions.values()))

# ========================== CHAT EXPORT ==========================
with st.expander("🗃️ Export Chat History"):
    chat_history = get_current_chat()
    if chat_history:
        df = pd.DataFrame(chat_history)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Q&A History as CSV",
            data=csv,
            file_name=f"civreply_chat_{council}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
        )
    else:
        st.info("No chat history yet to export for this council.")

# ========================== FEEDBACK & ABOUT ==========================
with st.expander("💡 Feedback & Feature Requests"):
    feedback = st.text_area("Share feedback or suggest a new feature:")
    if st.button("Submit Feedback"):
        if feedback.strip():
            with open("feedback_log.txt", "a") as f:
                f.write(f"{datetime.now().isoformat()} | {st.session_state.user_email} | {council} | {feedback}\n")
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please enter feedback before submitting.")

with st.expander("ℹ️ About CivReply AI"):
    st.write(APP_ABOUT)
    st.write("For more features, contributions, or to request a new council, email admin@civreply.ai.")
    st.write("This is an open-source, community-powered knowledge platform.")

# ========================== BRANDING BLOCK ==========================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#b2c6d6; font-size:0.96rem; margin:32px 0 8px 0;'>"
    "Made with 🏛️ CivReply AI – for Melbourne councils, powered by AI | 2025</div>",
    unsafe_allow_html=True
)
