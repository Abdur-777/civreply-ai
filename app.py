import streamlit as st
import os
import base64
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from datetime import datetime

# === STREAMLIT CONFIG ===
st.set_page_config(page_title="CivReply AI", page_icon="\U0001F3DB\uFE0F", layout="centered")

# === HEADER ===
st.markdown("""
    <h1 style='text-align: center;'>\U0001F3DB\uFE0F CivReply AI</h1>
    <p style='text-align: center; font-size: 1.2rem;'>Ask Wyndham Council anything – policies, laws, documents.</p>
    <hr>
""", unsafe_allow_html=True)

# === DARK MODE TOGGLE ===
dark_mode = st.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown("""<style>body { background-color: #111; color: white; }</style>""", unsafe_allow_html=True)

# === PDF UPLOADER ===
with st.expander("📤 Upload new Wyndham PDFs (Admin only)"):
    uploaded_files = st.file_uploader("Add new council documents (PDFs only):", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        os.makedirs("docs", exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join("docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("✅ Files added! Re-run app to re-index.")

# === LOAD Q&A CHAIN ===
@st.cache_resource
def load_qa():
    embeddings = OpenAIEmbeddings()
    index_dir = "faiss_index"

    if not os.path.exists(index_dir):
        with st.spinner("🔄 Indexing Wyndham documents..."):
            loader = PyPDFDirectoryLoader("docs")
            documents = loader.load()

            if not documents:
                st.error("❌ No documents found in /docs.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(documents)

            if not split_docs:
                st.error("❌ PDFs appear empty or scanned.")
                st.stop()

            db = FAISS.from_documents(split_docs, embeddings)
            db.save_local(index_dir)
    else:
        db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

qa_chain = load_qa()

# === ASK + RESPONSE ===
st.markdown("---")
query = st.text_input("🔍 Ask a question:", placeholder="e.g. How do I register a food premises?")
session = st.session_state.setdefault("history", [])

if query:
    with st.spinner("🧠 CivReply is thinking..."):
        response = qa_chain(query)
        answer = response['result']
        sources = response.get("source_documents", [])

        st.markdown("""
            <div style="background-color:#e8f5e9; padding: 1rem; border-radius: 0.5rem;">
                <strong>CivReply says:</strong><br>{}</div>
        """.format(answer), unsafe_allow_html=True)

        if sources:
            st.markdown("""<small><b>📄 Sources:</b></small>""", unsafe_allow_html=True)
            for src in sources:
                fname = os.path.basename(src.metadata.get("source", "Unknown PDF"))
                page = src.metadata.get("page", "?")
                st.markdown(f"- `{fname}`, page {page}")

        session.append({"q": query, "a": answer})

# === HISTORY ===
if session:
    with st.expander("🕘 Q&A History"):
        for i, item in enumerate(session[::-1]):
            st.markdown(f"**Q{i+1}:** {item['q']}")
            st.markdown(f"*A:* {item['a']}")
            st.markdown("---")

# === FOOTER ===
st.markdown("""
    <hr>
    <p style='font-size: 0.9rem; color: gray;'>⚙️ Powered by LangChain + OpenAI | Contact: <a href='mailto:wyndham@vic.gov.au'>wyndham@vic.gov.au</a></p>
""", unsafe_allow_html=True)
