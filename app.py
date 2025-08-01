import streamlit as st
import os
import json
import base64
import shutil
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pathlib import Path
import time
import pandas as pd

# === APP CONFIG ===
st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="wide")

# === SETTINGS ===
COUNCILS = ["Wyndham"]
PLAN_CONFIG = {
    "basic": {"limit": 500, "label": "Basic ($499/mo)", "features": ["PDF Q&A"]},
    "standard": {"limit": 2000, "label": "Standard ($1499/mo)", "features": ["PDF Q&A", "Form Scraping"]},
    "enterprise": {"limit": float("inf"), "label": "Enterprise ($2999+/mo)", "features": ["All Features"]},
}

# === ENV VAR CHECK ===
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("❌ Missing OpenAI API Key. Please set `OPENAI_API_KEY` in your environment.")
    st.stop()

# === SESSION INIT ===
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("query_count", 0)
st.session_state.setdefault("user_role", "Resident")
st.session_state.setdefault("council", "Wyndham")
st.session_state.setdefault("session_start", datetime.now().isoformat())
st.session_state.setdefault("plan", "basic")
st.session_state.setdefault("language", "English")

# ===== TITLE (center) =====
st.markdown("<h1 style='text-align:center;margin-bottom:0.5em;'>CivReply AI</h1>", unsafe_allow_html=True)

# ===== Controls row: Language | Role | Upgrades =====
ctrl_cols = st.columns([2,2,2], gap="large")
with ctrl_cols[0]:
    st.markdown("🌐 <b>Language</b>", unsafe_allow_html=True)
    st.session_state.language = st.selectbox(
        "",
        options=["English", "Arabic", "Chinese", "Hindi", "Spanish"],
        index=["English", "Arabic", "Chinese", "Hindi", "Spanish"].index(st.session_state.language)
        if st.session_state.get("language") in ["English", "Arabic", "Chinese", "Hindi", "Spanish"]
        else 0,
        label_visibility="collapsed",
        key="lang_selector"
    )
with ctrl_cols[1]:
    st.markdown("👤 <b>Select Role</b>", unsafe_allow_html=True)
    st.session_state.user_role = st.selectbox(
        "",
        options=["Resident", "Staff", "Visitor"],
        index=["Resident", "Staff", "Visitor"].index(st.session_state.get("user_role", "Resident")),
        label_visibility="collapsed",
        key="role_selector"
    )
with ctrl_cols[2]:
    if st.button("💼 Upgrades", use_container_width=True, key="upgrade_header"):
        st.session_state["goto_upgrades"] = True

st.markdown("---")

# === SIDEBAR (Menu & chat history) ===
with st.sidebar:
    st.image(
        "https://www.wyndham.vic.gov.au/sites/default/files/styles/small/public/2020-06/logo_0.png",
        width=200,
    )
    st.title("CivReply AI")
    nav = st.radio(
        "📚 Menu",
        [
            "💬 Chat with Council AI",
            "🧾 Topic FAQs",
            "📥 Submit a Request",
            "⬆️ Upgrades",
            "📊 Stats & Session",
            "📤 Export Logs",
            "💡 Share Feedback",
            "📞 Contact Us",
            "⚙️ Admin Panel",
        ],
    )

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 🧠 Recent Q&A")
        for i, (q, a) in enumerate(st.session_state.chat_history[::-1][:5]):
            if st.button(f"Q: {q[:40]}...", key=f"history_{i}"):
                st.session_state.chat_input = q
    st.markdown("---")
    # Bottom left upgrades button (always visible)
    if st.button("💼 Upgrades", use_container_width=True, key="sidebar_upgrade"):
        st.session_state["goto_upgrades"] = True

# Quick jump from header or sidebar upgrades button
if st.session_state.get("goto_upgrades"):
    nav = "⬆️ Upgrades"
    st.session_state["goto_upgrades"] = False

# === FUNCTIONS ===

def ask_ai(question, council):
    try:
        embeddings = OpenAIEmbeddings()
        index_path = f"index/{council.lower()}"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index found for {council}")
        db = FAISS.load_local(index_path, embeddings)
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(api_key=OPENAI_KEY), chain_type="stuff", retriever=retriever
        )
        return qa.run(question)
    except Exception as e:
        return f"[Error] Could not answer: {str(e)}"

def export_logs():
    # Write chat history to a simple text file with \n newlines (NO literal newlines)
    filename = f"chatlog_{st.session_state.session_start}.txt"
    with open(filename, "w") as f:
        for q, a in st.session_state.chat_history:
            f.write("Q: " + str(q) + "\n")
            f.write("A: " + str(a) + "\n")
            f.write("---\n")
    return filename

def log_feedback(text, email):
    # Append feedback entries to a simple log with newline separators
    with open("feedback_log.txt", "a") as f:
        entry = f"{datetime.now().isoformat()} | {email or 'anon'} | {text}\n"
        f.write(entry)

# === QUERY LIMIT CHECK ===
plan_id = st.session_state.plan
plan_limit = PLAN_CONFIG[plan_id]["limit"]

if st.session_state.query_count >= plan_limit:
    st.warning(f"❗ You’ve reached the {PLAN_CONFIG[plan_id]['label']} limit.")
    st.stop()

# === MAIN ROUTER ===
if nav == "💬 Chat with Council AI":
    st.subheader("💬 Ask Wyndham Council")
    user_input = st.chat_input("Ask a question about Wyndham policies, forms, or documents…")
    if user_input:
        st.session_state.query_count += 1
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.spinner("Wyndham GPT is replying..."):
            ai_reply = ask_ai(user_input, st.session_state.council)
            with st.chat_message("ai"):
                st.markdown(f"📩 **Auto-response from Wyndham Council:**\n\n{ai_reply}")
            st.session_state.chat_history.append((user_input, ai_reply))

elif nav == "🧾 Topic FAQs":
    st.subheader("FAQs by Topic")
    topics = {
        "Waste": ["How to report a missed bin?", "Where to get a new green bin?"],
        "Pets": ["Is dog registration required?", "Off-leash park locations?"],
        "Events": ["What’s on this weekend?", "How to book community rooms?"],
    }
    topic = st.selectbox("📂 Choose a category", list(topics.keys()))
    for q in topics[topic]:
        st.markdown(f"❓ {q}")

elif nav == "📥 Submit a Request":
    st.markdown("📌 Redirecting to your council’s website.")
    st.link_button("📝 Submit Online", "https://www.wyndham.vic.gov.au/request-it")

elif nav == "⬆️ Upgrades":
    st.title("Upgrade your plan")

    # Segmented toggle (Personal | Business)
    mode = st.segmented_control(
        "Choose plan type",
        options=["Personal", "Business"],
        default="Business",
    )

    st.write("")

    # Plan card helper
    def plan_card(title, price, period, tagline, cta_label, features):
        with st.container(border=True):
            st.subheader(title)
            price_col, _ = st.columns([1, 3])
            with price_col:
                st.markdown(f"### ${price}  \n**USD / {period}**")
            st.markdown(tagline)
            st.button(cta_label, use_container_width=True)
            st.markdown("---")
            for f in features:
                st.markdown(f"- {f}")

    if mode == "Personal":
        plan_card(
            title="Plus",
            price="20",
            period="month",
            tagline="Great for power users who want faster answers and longer context.",
            cta_label="Upgrade to Plus",
            features=[
                "Priority compute and faster responses",
                "Access to advanced models",
                "Larger context for longer chats",
                "Use on web and mobile",
            ],
        )
    else:  # Business — remove Team, keep three business tiers
        st.subheader("Compare business tiers")
        cols = st.columns(3)
        with cols[0]:
            plan_card("Basic", "499", "month", "For small teams getting started.",
                      "Choose Basic", ["PDF Q&A", "Limit: 500 queries"])
        with cols[1]:
            plan_card("Standard", "1499", "month", "Scale insights across your council.",
                      "Choose Standard", ["PDF Q&A", "Form Scraping", "Limit: 2000 queries"])
        with cols[2]:
            plan_card("Enterprise", "2999+", "month", "For large organizations that need controls and scale.",
                      "Contact Sales", ["All Features", "Limit: Unlimited queries"])

elif nav == "📊 Stats & Session":
    st.metric("Total Questions", st.session_state.query_count)
    st.metric("Session Start", st.session_state.session_start)
    st.metric("Role", st.session_state.user_role)
    st.metric("Council", st.session_state.council)
    st.metric("Plan", PLAN_CONFIG[st.session_state.plan]["label"])

elif nav == "📤 Export Logs":
    if st.button("📄 Download Session Log"):
        fname = export_logs()
        with open(fname, "rb") as f:
            st.download_button("📥 Download", f, file_name=fname)

elif nav == "💡 Share Feedback":
    fb = st.text_area("Tell us what’s working or not...")
    email = st.text_input("Email (optional)")
    if st.button("📨 Submit"):
        log_feedback(fb, email)
        st.success("Thanks for helping improve CivReply AI!")

elif nav == "📞 Contact Us":
    st.markdown("**Call:** (03) 9742 0777  ")
    st.markdown("**Visit:** 45 Princes Hwy, Werribee  ")
    st.markdown("**Mail:** PO Box 197")
    st.link_button("Website", "https://www.wyndham.vic.gov.au")

elif nav == "⚙️ Admin Panel":
    if st.session_state.user_role != "Staff":
        st.warning("Restricted to council staff only.")
    else:
        st.subheader("📂 Upload New Council Docs")
        docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Rebuild Index"):
            if docs:
                folder = f"docs/{st.session_state.council.lower()}"
                os.makedirs(folder, exist_ok=True)
                for d in docs:
                    with open(os.path.join(folder, d.name), "wb") as f:
                        f.write(d.read())
                loader = PyPDFDirectoryLoader(folder)
                raw_docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(raw_docs)
                vecdb = FAISS.from_documents(chunks, OpenAIEmbeddings())
                vecdb.save_local(f"index/{st.session_state.council.lower()}")
                st.success("✅ Index rebuilt successfully.")
            else:
                st.warning("Please upload at least one document.")
