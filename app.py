import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="centered")
st.title("🏛️ CivReply AI – Ask Wyndham Council Anything")

@st.cache_resource
def load_qa():
    embeddings = OpenAIEmbeddings()
    index_dir = "faiss_index"

    if not os.path.exists(index_dir):
        with st.spinner("No FAISS index found. Creating one from /docs..."):
            print("📂 Files Render sees in /docs:", os.listdir("docs"))  # 👈 LOG LINE HERE
            loader = PyPDFDirectoryLoader("docs")
            documents = loader.load()

            if not documents:
                st.error("❌ No documents loaded. Make sure /docs contains readable PDFs.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(documents)

            if not split_docs:
                st.error("❌ No text chunks found in documents. Make sure your PDFs contain selectable text (not just scanned images).")
                st.stop()

            db = FAISS.from_documents(split_docs, embeddings)
            db.save_local(index_dir)
    else:
        db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


qa = load_qa()

query = st.text_input("Ask about Wyndham Council policies, rules, or documents:")
if query:
    with st.spinner("Thinking..."):
        result = qa.run(query)
    st.success("CivReply says:")
    st.write(result)
