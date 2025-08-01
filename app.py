import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Page setup
st.set_page_config(page_title="CivReply AI – Wyndham", page_icon="🏛️", layout="wide")

# Language selection
language = st.selectbox("🌐 Choose your language", ["English", "Arabic", "Mandarin", "Hindi", "Vietnamese", "Spanish"])

# Sidebar navigation
with st.sidebar:
    st.image("https://www.wyndham.vic.gov.au/sites/default/files/styles/small/public/2020-06/logo_0.png", width=200)
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #0f2e47;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    st.title("🏙️ Wyndham City")
    st.markdown("### Navigation")
    nav_option = st.radio("Go to:", [
        "Ask a Question", 
        "Browse Topics", 
        "Submit a Request", 
        "Upcoming Events", 
        "Feedback", 
        "Contact Council", 
        "Admin Tools"
    ], index=0)

# Header
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 40px;'>🏛️ CivReply AI – Wyndham</h1>
        <p style='font-size: 18px;'>Empowering Wyndham residents with smarter answers.</p>
        <div style='font-size: 16px; margin-top: 10px;'>
            <strong>📦 Plan: Basic</strong> | 500 queries/mo
        </div>
    </div>
""", unsafe_allow_html=True)

# Email input
email = st.text_input("📩 Email for response (optional)")

# Chat logic
if nav_option == "Ask a Question":
    user_question = st.chat_input("Ask a local question")
    if user_question:
        with st.spinner("Thinking..."):
            try:
                embeddings = OpenAIEmbeddings()
                db = FAISS.load_local("index/wyndham", embeddings)
                retriever = db.as_retriever()
                qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=retriever)
                response = qa_chain.run(user_question)
                st.chat_message("ai").markdown(response)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

elif nav_option == "Browse Topics":
    st.subheader("🗂️ Browse Topics")
    st.markdown("Select a topic to explore common council queries.")
    st.button("Waste Services")
    st.button("Pets & Animals")
    st.button("Permits & Planning")
    st.button("Community Events")
    st.button("Roads & Parking")

elif nav_option == "Submit a Request":
    st.subheader("📥 Submit a Request")
    st.markdown("This feature will let residents submit bin replacement, report issues, or request documents.")

elif nav_option == "Upcoming Events":
    st.subheader("🗓️ Upcoming Events")
    st.markdown("This section can be used to showcase council events like local markets, meetings, or workshops.")

elif nav_option == "Feedback":
    st.subheader("💬 Feedback")
    feedback = st.text_area("Let us know how we can improve or what you liked!")
    if st.button("Submit Feedback"):
        st.success("Thanks for your feedback!")

elif nav_option == "Contact Council":
    st.subheader("📞 Contact Wyndham Council")
    with st.expander("Request It Online (Service or Issue)"):
        st.markdown("[Submit online request](https://www.wyndham.vic.gov.au/request-it)")
    with st.expander("Online Chat"):
        st.markdown("[Start online chat](https://www.wyndham.vic.gov.au)")
    with st.expander("Phone us"):
        st.markdown("Call: (03) 9742 0777")
    with st.expander("Visit Us in Person"):
        st.markdown("Wyndham Civic Centre, 45 Princes Hwy, Werribee")
    with st.expander("Write to Us"):
        st.markdown("Mail: PO Box 197, Werribee 3030")
    with st.expander("Provide Feedback (Complaint/Compliment)"):
        st.markdown("[Feedback form](https://www.wyndham.vic.gov.au/feedback)")
    with st.expander("Update your details"):
        st.markdown("[Update here](https://www.wyndham.vic.gov.au/update-details)")
    with st.expander("Connect With Us - Social Media"):
        st.markdown("Facebook, X, YouTube, Instagram, LinkedIn")
    with st.expander("Subscribe to an eNewsletter"):
        st.markdown("[Subscribe](https://www.wyndham.vic.gov.au/subscribe)")

elif nav_option == "Admin Tools":
    st.subheader("⚙️ Admin Tools")
    st.markdown("Upload PDFs and rebuild the knowledge base for Wyndham.")
    pdfs = st.file_uploader("Upload Council PDFs", type="pdf", accept_multiple_files=True)
    if st.button("🔄 Rebuild Index"):
        if pdfs:
            with st.spinner("Indexing documents..."):
                temp_folder = "docs/temp"
                os.makedirs(temp_folder, exist_ok=True)
                for pdf in pdfs:
                    with open(os.path.join(temp_folder, pdf.name), "wb") as f:
                        f.write(pdf.read())
                loader = PyPDFDirectoryLoader(temp_folder)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(documents)
                embeddings = OpenAIEmbeddings()
                db = FAISS.from_documents(docs, embeddings)
                db.save_local("index/wyndham")
                st.success("Index rebuilt successfully!")
        else:
            st.warning("Please upload PDF documents first.")
