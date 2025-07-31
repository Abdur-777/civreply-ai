# CivReply AI – Upgraded Version with Admin Login, Multi-Council Support, Stripe, and More
import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STRIPE_LINK = os.getenv("STRIPE_LINK", "https://buy.stripe.com/test_xxx")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "supersecret")

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

# --- Footer ---
st.markdown(f"""
<div class="footer">
  ⚙️ Powered by LangChain + GPT-4 | Queries used: {st.session_state.query_count} / 500<br>
  📬 Contact: <a href="mailto:contact@civreply.ai">contact@civreply.ai</a>
</div>
""", unsafe_allow_html=True)
