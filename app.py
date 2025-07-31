import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="centered")
st.title("🏛️ CivReply AI – Ask Wyndham Council Anything")

@st.cache_resource
def load_qa():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings)
    retriever = db.as_retriever()
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa = load_qa()
query = st.text_input("Ask a question about Wyndham Council:")
if query:
    with st.spinner("Searching documents..."):
        response = qa.run(query)
    st.success("Answer:")
    st.write(response)