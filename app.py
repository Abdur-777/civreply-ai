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

# === SETTINGS ===
COUNCILS = ["Wyndham"]
PLAN_CONFIG = {
    "basic": {"limit": 500, "label": "Basic ($499/mo)", "features": ["PDF Q&A"]},
    "standard": {"limit": 2000, "label": "Standard ($1499/mo)", "features": ["PDF Q&A", "Form Scraping"]},
    "enterprise": {"limit": float("inf"), "label": "Enterprise ($2999+/mo)", "features": ["All Features"]},
}

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("❌ Missing OpenAI API Key. Please set `OPENAI_API_KEY` in your environment.")
    st.stop()

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("query_count", 0)
st.session_state.setdefault("user_role", "Resident")
st.session_state.setdefault("council", "Wyndham")
st.session_state.setdefault("session_start", datetime.now().isoformat())
st.session_state.setdefault("plan", "basic")
st.session_state.setdefault("language", "English")

# ===== CivReply AI title with Wyndham logo and photo =====
st.markdown(
    """
    <div style='display: flex; align-items: center; justify-content: center; margin-top: 10px; margin-bottom: 8px; gap: 18px;'>
        <img src="https://www.wyndham.vic.gov.au/sites/default/files/styles/small/public/2020-06/logo_0.png" width="44" style="border-radius:8px;" />
        <h1 style='margin-bottom: 0; margin-top: 0; font-size: 2.5rem;'>CivReply AI</h1>
        <img src="https://www.wyndham.vic.gov.au/sites/default/files/styles/media_crop/public/2022-04/Wyndham-city-aerial-photo.jpg" width="52" style="border-radius:10px; box-shadow: 0 2px 10px #b5d1f1aa;" />
    </div>
    """,
    unsafe_allow_html=True
)

# ===== BLUE WYNDHAM COUNCIL BADGE =====
st.markdown(
    """
    <div style='display: flex; justify-content: center; margin-bottom: 25px;'>
        <span style="
            background: linear-gradient(90deg, #e3f0ff 60%, #c6e0ff 100%);
            color: #0a318e; font-weight: bold;
            border-radius: 16px; padding: 7px 28px; font-size: 1.1rem;
            box-shadow: 0 2px 8px #b5d1f180;">
            🏛️ Active Council: Wyndham
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# ===== Controls row: Language | Role =====
ctrl_cols = st.columns([1,1], gap="large")
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

st.markdown("---")

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
            "⚙️ Admin Panel",
        ],
    )

    # --- ChatGPT-style sidebar with 5 most recent user questions ---
    st.markdown("---")
    st.markdown("#### Chats")
    last_5 = [q for q, a in st.session_state.chat_history[-5:]]
    if last_5:
        for q in reversed(last_5):  # Most recent on top
            st.markdown(f"<div style='padding:7px 0; font-size:16px;'>{q}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#aaa;'>No chats yet</span>", unsafe_allow_html=True)
    st.markdown("---")

    # Only one Upgrades button (bottom of sidebar)
    if st.button("💼 Upgrades", use_container_width=True, key="sidebar_upgrade"):
        st.session_state["goto_upgrades"] = True

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
    filename = f"chatlog_{st.session_state.session_start}.txt"
    with open(filename, "w") as f:
        for q, a in st.session_state.chat_history:
            f.write("Q: " + str(q) + "\n")
            f.write("A: " + str(a) + "\n")
            f.write("---\n")
    return filename

def log_feedback(text, email):
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

elif nav == "📥 Submit a Request":
    st.markdown("📌 Redirecting to your council’s website.")
    st.link_button("📝 Submit Online", "https://www.wyndham.vic.gov.au/request-it")

elif nav == "⬆️ Upgrades":
    st.title("Upgrade your plan")
    mode = st.segmented_control(
        "Choose plan type",
        options=["Personal", "Business"],
        default="Business",
    )
    st.write("")

    def plan_card(title, price, period, tagline, cta_label, features):
        with st.container(border=True):
            st.subheader(title)
            price_col, _ = st.columns([1, 3])
            with price_col:
                st.markdown(f"<span style='font-size:2.2rem; font-weight:600;'>${price}</span><br><span style='font-size:1rem;'>USD / month</span>", unsafe_allow_html=True)
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
    else:
        st.subheader("Compare business tiers")
        cols = st.columns(3)
        with cols[0]:
            plan_card(
                "Basic", "499", "month", "For small teams getting started.",
                "Choose Basic",
                [
                    "PDF Q&A (ask about any council document)",
                    "Limit: 500 queries",
                    "Email support (48h response)",
                    "Council policy finder",
                    "Mobile access",
                    "Secure cloud storage",
                    "Community knowledge base",
                ]
            )
        with cols[1]:
            plan_card(
                "Standard", "1499", "month", "Scale insights across your council.",
                "Choose Standard",
                [
                    "Everything in Basic",
                    "Limit: 2000 queries",
                    "Form Scraping (auto-extract info from forms)",
                    "Priority email & chat support (24h)",
                    "Usage analytics dashboard",
                    "PDF export of chats",
                    "Multi-language Q&A",
                    "Bulk data uploads",
                    "Custom council branding",
                ]
            )
        with cols[2]:
            plan_card(
                "Enterprise", "2999+", "month", "For large organizations that need controls and scale.",
                "Contact Sales",
                [
                    "Everything in Standard",
                    "Limit: Unlimited queries",
                    "Dedicated account manager",
                    "API access & automation",
                    "SLA: 99.9% uptime",
                    "Single Sign-On (SSO)",
                    "Staff training sessions",
                    "Integration with 3rd party tools (Teams, Slack, etc.)",
                    "On-premise/cloud deployment options",
                    "Custom workflow automations",
                ]
            )

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
