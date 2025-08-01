import streamlit as st
import os
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# === APP CONFIG ===
st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="wide")

PLAN_CONFIG = {
    "basic": {
        "label": "Basic ($499 AUD/mo)",
        "limit": 500,
        "features": [
            "PDF Q&A (ask about any council document)",
            "Limit: 500 queries",
            "Email support (24h response)",
            "Council policy finder",
            "Mobile access",
            "Secure cloud storage",
            "Community knowledge base"
        ],
    },
    "standard": {
        "label": "Standard ($1,499 AUD/mo)",
        "limit": 2000,
        "features": [
            "Everything in Basic",
            "Limit: 2,000 queries",
            "Form Scraping (auto-extract info from forms)",
            "Immediate email & chat support",
            "Usage analytics dashboard",
            "PDF export of chats",
            "Multi-language Q&A",
            "Bulk data uploads",
            "Custom council branding"
        ],
    },
    "enterprise": {
        "label": "Enterprise ($2,999+ AUD/mo)",
        "limit": float("inf"),
        "features": [
            "Everything in Standard",
            "Limit: Unlimited queries",
            "Dedicated account manager",
            "API access & automation",
            "SLA: 99.9% uptime",
            "Single Sign-On (SSO)",
            "Staff training sessions",
            "Integration with 3rd party tools (Teams, Slack, etc.)",
            "On-premise/cloud deployment options",
            "Custom workflow automations"
        ],
    }
}

# === ENV VAR CHECK ===
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("❌ Missing OpenAI API Key. Please set `OPENAI_API_KEY` in your environment.")
    st.stop()

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("query_count", 0)
st.session_state.setdefault("user_role", "Resident")
st.session_state.setdefault("plan", "basic")
st.session_state.setdefault("language", "English")
st.session_state.setdefault("council", "Wyndham")
st.session_state.setdefault("session_start", datetime.now().isoformat())

# === HEADER ===
st.markdown(
    """
    <div style='display: flex; align-items: center; justify-content: center; margin-top: 10px; margin-bottom: 8px; gap: 18px;'>
        <img src="https://www.wyndham.vic.gov.au/sites/default/files/styles/small/public/2020-06/logo_0.png" width="44" style="border-radius:8px;" />
        <h1 style='margin-bottom: 0; margin-top: 0; font-size: 2.5rem;'>CivReply AI</h1>
    </div>
    """, unsafe_allow_html=True
)

# === PLAN BADGE & CONTROLS ===
ctrl_cols = st.columns([1,1,1], gap="large")
with ctrl_cols[0]:
    st.markdown("🌐 <b>Language</b>", unsafe_allow_html=True)
    st.session_state.language = st.selectbox(
        "",
        options=["English", "Arabic", "Chinese", "Hindi", "Spanish"],
        index=["English", "Arabic", "Chinese", "Hindi", "Spanish"].index(st.session_state.language)
        if st.session_state.get("language") in ["English", "Arabic", "Chinese", "Hindi", "Spanish"]
        else 0,
        label_visibility="collapsed"
    )
with ctrl_cols[1]:
    st.markdown("👤 <b>Select Role</b>", unsafe_allow_html=True)
    st.session_state.user_role = st.selectbox(
        "",
        options=["Resident", "Staff", "Visitor"],
        index=["Resident", "Staff", "Visitor"].index(st.session_state.get("user_role", "Resident")),
        label_visibility="collapsed"
    )
with ctrl_cols[2]:
    if st.session_state.user_role == "Staff":
        st.markdown("🛠️ <b>Admin Plan Control</b>", unsafe_allow_html=True)
        st.session_state.plan = st.selectbox(
            "",
            options=["basic", "standard", "enterprise"],
            format_func=lambda x: PLAN_CONFIG[x]["label"],
            key="admin_plan_selector"
        )

st.markdown(
    f"<div style='display:flex;justify-content:center;margin-bottom:10px;'><span style='background:#e3f0ff;color:#0a318e;padding:7px 28px;border-radius:16px;font-weight:bold;'>{PLAN_CONFIG[st.session_state.plan]['label']}</span></div>",
    unsafe_allow_html=True
)
st.divider()

# === SIDEBAR ===
with st.sidebar:
    st.title("CivReply AI")
    nav = st.radio(
        "📚 Menu",
        [
            "💬 Chat with Council AI",
            "📥 Submit a Request",
            "⬆️ Upgrades",
            "📊 Stats & Session",
            "💡 Share Feedback",
            "📞 Contact Us",
            "⚙️ Admin Panel"
        ],
    )
    st.markdown("---")
    st.markdown("#### Recent Chats")
    last_5 = [q for q, a in st.session_state.chat_history[-5:]]
    if last_5:
        for q in reversed(last_5):
            st.markdown(f"<div style='padding:7px 0; font-size:16px;'>{q}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#aaa;'>No chats yet</span>", unsafe_allow_html=True)

# === PLAN-SPECIFIC AI LOGIC ===
def ask_ai(question, council):
    plan = st.session_state.get("plan", "basic")
    embeddings = OpenAIEmbeddings()
    index_path = f"index/{council.lower()}"
    if not os.path.exists(index_path):
        return "[Error] No index found for this council"
    db = FAISS.load_local(index_path, embeddings)
    retriever = db.as_retriever()

    if plan == "basic":
        llm = ChatOpenAI(api_key=OPENAI_KEY, model="gpt-3.5-turbo")
        prompt = "You are a helpful council assistant. Only answer from the provided documents."
    elif plan == "standard":
        llm = ChatOpenAI(api_key=OPENAI_KEY, model="gpt-4o")
        prompt = "You are a council assistant with advanced extraction and analytics skills."
    elif plan == "enterprise":
        llm = ChatOpenAI(api_key=OPENAI_KEY, model="gpt-4o")
        prompt = "You are an enterprise-grade council AI with API, integrations, automations, and full workflow capabilities."
    else:
        llm = ChatOpenAI(api_key=OPENAI_KEY)
        prompt = ""

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt} if prompt else None,
    )
    return qa.run(question)

def log_feedback(text, email):
    with open("feedback_log.txt", "a") as f:
        entry = f"{datetime.now().isoformat()} | {email or 'anon'} | {text}\n"
        f.write(entry)

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

elif nav == "📥 Submit a Request":
    st.markdown("📌 Redirecting to your council’s website.")
    st.link_button("📝 Submit Online", "https://www.wyndham.vic.gov.au/request-it")

elif nav == "⬆️ Upgrades":
    st.title("Upgrade your plan")
    st.subheader("Compare business tiers")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### Basic")
        st.markdown("**$499 AUD/mo**")
        for feat in PLAN_CONFIG["basic"]["features"]:
            st.markdown(f"- {feat}")
    with cols[1]:
        st.markdown("#### Standard")
        st.markdown("**$1,499 AUD/mo**")
        for feat in PLAN_CONFIG["standard"]["features"]:
            st.markdown(f"- {feat}")
    with cols[2]:
        st.markdown("#### Enterprise")
        st.markdown("**$2,999+ AUD/mo**")
        for feat in PLAN_CONFIG["enterprise"]["features"]:
            st.markdown(f"- {feat}")

elif nav == "📊 Stats & Session":
    st.metric("Total Questions", st.session_state.query_count)
    st.metric("Session Start", st.session_state.session_start)
    st.metric("Role", st.session_state.user_role)
    st.metric("Council", st.session_state.council)
    st.metric("Plan", PLAN_CONFIG[st.session_state.plan]["label"])

elif nav == "💡 Share Feedback":
    fb = st.text_area("Tell us what’s working or not...")
    email = st.text_input("Email (optional)")
    if st.button("📨 Submit"):
        log_feedback(fb, email)
        st.success("Thanks for helping improve CivReply AI!")

elif nav == "📞 Contact Us":
    st.markdown("**Call:** (03) 9742 0777")
    st.markdown("**Visit:** 45 Princes Hwy, Werribee")
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
