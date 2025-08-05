import os
import streamlit as st
from datetime import datetime
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ============ CONFIG ============
COUNCILS = ["Wyndham", "Melbourne"]  # Add more councils here
DEFAULT_COUNCIL = COUNCILS[0]

LANGUAGES = {
    "English": "en",
    "Chinese": "zh-CN",
    "Arabic": "ar",
    "Spanish": "es",
    "Hindi": "hi",
    "Vietnamese": "vi"
}

# ============ STREAMLIT UI ============
st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="wide")

st.title("CivReply AI: Multi-Council Q&A")
st.caption("Ask questions, upload new PDFs, and instantly power your council AI with local documents.")

council = st.sidebar.selectbox("Select council", COUNCILS, index=COUNCILS.index(DEFAULT_COUNCIL))
language = st.sidebar.selectbox("Language", list(LANGUAGES.keys()), index=0)

pdf_dir = f"docs/{council.lower()}"
index_dir = f"index/{council.lower()}"

os.makedirs(pdf_dir, exist_ok=True)
os.makedirs("index", exist_ok=True)

# ============ ADMIN/STAFF PDF UPLOAD ============
with st.sidebar.expander("📥 Admin: Upload and Index PDFs"):
    role = st.selectbox("Your role", ["Resident", "Staff", "Admin"])
    if role in ["Staff", "Admin"]:
        uploaded_pdfs = st.file_uploader("Upload PDFs to index", type="pdf", accept_multiple_files=True)
        if st.button("Bulk Index PDFs"):
            if uploaded_pdfs:
                for doc in uploaded_pdfs:
                    # Save uploaded files
                    file_path = os.path.join(pdf_dir, doc.name)
                    with open(file_path, "wb") as f:
                        f.write(doc.read())
                # Build new index
                st.info("Indexing documents, please wait...")
                loader = PyPDFDirectoryLoader(pdf_dir)
                raw_docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(raw_docs)
                embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
                db = FAISS.from_documents(chunks, embeddings)
                db.save_local(index_dir)
                st.success("✅ Bulk indexing complete for " + council)
            else:
                st.warning("No files uploaded.")

# ============ INDEX LOADER ============
@st.cache_resource(show_spinner="Loading document index...")
def get_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    if not os.path.exists(index_dir):
        return None
    return FAISS.load_local(index_dir, embeddings)

db = get_vectorstore()
if db is None:
    st.warning(f"No index found for {council}. Upload and index PDFs as Admin/Staff.")
    st.stop()

# ============ RETRIEVAL QA ============
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0),
    retriever=db.as_retriever(),
)

# ============ MAIN Q&A UI ============
st.subheader(f"💬 Ask {council} Council AI")
query = st.text_input(f"Ask anything about {council} council documents:")

if query:
    with st.spinner("Thinking..."):
        answer = qa({"query": query})
    st.markdown("**Answer:**")
    st.success(answer["result"])

    # Save history (optional, add download/export later)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append((datetime.now().isoformat(), council, query, answer["result"]))

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#b2c6d6; font-size:0.96rem; margin:32px 0 8px 0;'>Made with 🏛️ CivReply AI – for Australian councils, powered by AI</div>",
    unsafe_allow_html=True
)
