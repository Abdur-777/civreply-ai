import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center;'>🏛️ CivReply AI</h1>
    <p style='text-align: center; font-size: 1.2rem;'>Ask Wyndham Council anything – policies, laws, documents.</p>
    <hr>
    """, unsafe_allow_html=True
)

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

    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa = load_qa()

st.markdown("---")

query = st.text_input("🔍 Ask a question:", placeholder="e.g. How do I register a food premises?")
if query:
    with st.spinner("🧠 CivReply is thinking..."):
        result = qa.run(query)
    st.markdown(
        f"""
        <div style="background-color:#e8f5e9; padding: 1rem; border-radius: 0.5rem;">
            <strong>CivReply says:</strong><br>{result}
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("<br><hr><small style='color:gray;'>⚙️ Powered by LangChain + OpenAI</small>", unsafe_allow_html=True)
