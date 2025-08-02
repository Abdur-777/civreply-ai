import streamlit as st
import os
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ===== THEME AND CONFIG =====
WYNDHAM_BLUE = "#36A9E1"
WYNDHAM_DEEP = "#2078b2"
WYNDHAM_LIGHT = "#e3f3fa"
ADMIN_PASSWORD = "llama"

st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="wide")

PLAN_CONFIG = {
    "basic": {
        "label": "Basic ($499 AUD/mo)",
        "icon": "💧",
        "limit": 500,
        "features": [
            "📄 PDF Q&A (ask about any council document)",
            "🔒 Limit: 500 queries",
            "📧 Email support (24h response)",
            "📚 Council policy finder",
            "📱 Mobile access",
            "☁️ Secure cloud storage",
            "👥 Community knowledge base"
        ],
    },
    "standard": {
        "label": "Standard ($1,499 AUD/mo)",
        "icon": "🚀",
        "limit": 2000,
        "features": [
            "✅ Everything in Basic",
            "🔒 Limit: 2,000 queries",
            "📝 Form Scraping (auto-extract info from forms)",
            "⚡ Immediate email & chat support",
            "📊 Usage analytics dashboard",
            "🗃️ PDF export of chats",
            "🌏 Multi-language Q&A",
            "📦 Bulk data uploads",
            "🎨 Custom council branding",
            "<b>Contact sales to upgrade below</b>"
        ],
    },
    "enterprise": {
        "label": "Enterprise ($2,999+ AUD/mo)",
        "icon": "🏆",
        "limit": float("inf"),
        "features": [
            "✅ Everything in Standard",
            "🔓 Limit: Unlimited queries",
            "👤 Dedicated account manager",
            "🔌 API access & automation",
            "🛡️ SLA: 99.9% uptime",
            "🔐 Single Sign-On (SSO)",
            "🧑‍🏫 Staff training sessions",
            "🤖 Integration with 3rd party tools (Teams, Slack, etc.)",
            "☁️ On-premise/cloud deployment options",
            "🛠️ Custom workflow automations",
            "<b>Contact sales to upgrade below</b>"
        ],
    }
}

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
st.session_state.setdefault("admin_verified", False)

# ===== HEADER: Big Icon + Large Text =====
st.markdown(
    f"""
    <div style='background:linear-gradient(90deg,{WYNDHAM_BLUE},#7ecaf6 100%);padding:44px 0 24px 0;border-radius:0 0 44px 44px;box-shadow:0 10px 40px #cce5f7;display:flex;align-items:center;justify-content:center;gap:34px;'>
      <span style="font-size:4.2rem;line-height:1;border-radius:20px;background:rgba(255,255,255,0.09);box-shadow:0 0 22px #99cef4;padding:6px 28px 6px 20px;">🏛️</span>
      <span style='font-size:3.5rem;font-weight:900;color:#fff;letter-spacing:4px;text-shadow:0 2px 22px #36A9E180;'>CivReply AI</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ===== STATUS BAR =====
st.markdown(
    f"""
    <div style='background:{WYNDHAM_LIGHT};border-radius:16px;padding:17px 48px;display:flex;justify-content:center;align-items:center;gap:60px;margin-top:20px;margin-bottom:14px;box-shadow:0 2px 10px #b4dbf2;'>
        <div style='color:{WYNDHAM_DEEP};font-size:1.13rem;font-weight:700'>🏛️ Active Council:</div>
        <div style='font-weight:700;'>{st.session_state.get('council', 'Wyndham')}</div>
        <div style='color:{WYNDHAM_DEEP};font-size:1.13rem;font-weight:700'>📦 Plan:</div>
        <div style='font-weight:700;'>{PLAN_CONFIG.get(st.session_state.get('plan'), PLAN_CONFIG['basic'])["label"]}</div>
        <div style='color:{WYNDHAM_DEEP};font-size:1.13rem;font-weight:700'>🌐 Language:</div>
        <div style='font-weight:700;'>English</div>
    </div>
    """, unsafe_allow_html=True
)

# ===== HERO WELCOME SECTION =====
st.markdown(
    f"""
    <div style="background:{WYNDHAM_LIGHT};border-radius:20px;padding:28px 46px 22px 46px;margin:26px 0 32px 0;box-shadow:0 2px 16px #cdeafe;">
      <span style="font-size:2.25rem;font-weight:900;color:{WYNDHAM_DEEP};margin-right:8px;">👋 Welcome!</span>
      <span style="font-size:1.33rem;font-weight:500;color:#1762a6;">
        CivReply AI helps you find answers, policies, and services from Wyndham Council instantly.<br>
        <span style="font-size:1.09rem;font-weight:400;color:#287bb7;">Try asking about rubbish collection, local laws, grants, rates, events and more!</span>
      </span>
    </div>
    """,
    unsafe_allow_html=True
)

# ===== SAMPLE QUESTION BUTTONS =====
EXAMPLES = [
    "What day is my rubbish collected?",
    "How do I apply for a pet registration?",
    "What are the rules for backyard sheds?",
    "Where can I find local events?",
    "How do I pay my rates online?",
]
st.markdown(
    "<div style='margin:6px 0 6px 0;font-weight:700;color:#2176b6;'>💡 Try asking:</div>",
    unsafe_allow_html=True
)
ex_cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    with ex_cols[i]:
        if st.button(ex, key=f"ex{i}"):
            st.session_state['chat_input'] = ex

# ===== HOW IT WORKS (EXPANDER) =====
with st.expander("How does CivReply AI work?", expanded=False):
    st.markdown("""
      1. **Type your question** about council policies, forms, or services.
      2. **Our AI instantly searches** official council documents for the right answer.
      3. **Get clear, human-friendly replies** in seconds!
    """)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown(
        f"""
        <div style='background:{WYNDHAM_BLUE};padding:20px 0 14px 0;border-radius:0 0 32px 32px;box-shadow:0 4px 18px #cce5f7;margin-bottom:10px;'>
          <div style='display:flex;align-items:center;justify-content:center;gap:15px;'>
            <span style="font-size:2.5rem;">🏛️</span>
            <span style='font-size:1.65rem;font-weight:800;color:#fff;letter-spacing:0.7px;'>CivReply AI</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    nav = st.radio(
        "",
        [
            "💬 Chat with Council AI",
            "📥 Submit a Request",
            "📊 Stats & Session",
            "💡 Share Feedback",
            "📞 Contact Us",
            "ℹ️ About Us",
            "⚙️ Admin Panel"
        ],
        label_visibility="collapsed"
    )
    # Recent Chats
    st.markdown("<div style='text-align:center;font-size:1.12rem;font-weight:700;color:#235b7d;margin:16px 0 0 0;'>Recent Chats</div>", unsafe_allow_html=True)
    last_5 = [q for q, a in st.session_state.get("chat_history", [])[-5:]]
    if last_5:
        for q in reversed(last_5):
            st.markdown(f"<div style='padding:10px 0; text-align:center; font-size:15.5px;color:#2078b2;'>{q}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#7eb7d8;text-align:center;display:block;'>No chats yet</span>", unsafe_allow_html=True)
    # ---- UPGRADE PLAN CARD ----
    with st.expander("🚀 Upgrade Your Plan", expanded=False):
        for plan_key, plan in PLAN_CONFIG.items():
            st.markdown(
                f"""
                <div style='background:linear-gradient(145deg,#f2fbfe 60%,#cbe7f8 100%);border-radius:18px;box-shadow:0 4px 18px #c1e3f4;padding:18px 14px 10px 14px;margin-bottom:12px;'>
                  <div style="font-size:1.17rem;font-weight:900;color:#158ed8;margin-bottom:8px;">{plan['icon']} {plan['label'].split('(')[0]}</div>
                  <div style="font-size:1.3rem;font-weight:800;color:{WYNDHAM_BLUE};margin-bottom:6px;">{plan['label'].split('(')[1][:-4]} AUD</div>
                  <ul style="padding-left:18px;font-size:1.08rem;line-height:1.7;">
                    {''.join([f"<li style='margin-bottom:3px;color:#1374ab'>{f}</li>" for f in plan['features']])}
                  </ul>
                  {'<div style="margin-top:8px;"><a href="mailto:sales@civreply.com?subject=CivReply%20Plan%20Upgrade%20Enquiry" style="background:#36A9E1;color:#fff;font-weight:700;padding:8px 18px;border-radius:10px;text-decoration:none;display:inline-block;font-size:1.03rem;box-shadow:0 2px 6px #bae3fc;">Contact Sales</a></div>' if plan_key in ['standard', 'enterprise'] else ''}
                </div>
                """,
                unsafe_allow_html=True
            )

# =========== PLAN-SPECIFIC AI LOGIC ===========
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
        prompt = "You are a helpful council assistant. Only answer from the provided documents. Always respond in English."
    elif plan == "standard":
        llm = ChatOpenAI(api_key=OPENAI_KEY, model="gpt-4o")
        prompt = "You are a council assistant with advanced extraction and analytics skills. Always respond in English."
    elif plan == "enterprise":
        llm = ChatOpenAI(api_key=OPENAI_KEY, model="gpt-4o")
        prompt = "You are an enterprise-grade council AI with API, integrations, automations, and full workflow capabilities. Always respond in English."
    else:
        llm = ChatOpenAI(api_key=OPENAI_KEY)
        prompt = "Always respond in English."
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

# ===== MAIN APP ROUTER =====
if nav == "💬 Chat with Council AI":
    st.subheader("💬 Ask Wyndham Council")
    user_input = st.chat_input("Ask a question about Wyndham policies, forms, or documents…")
    if user_input:
        st.session_state.query_count += 1
        st.write(user_input)
        ai_reply = ask_ai(user_input, st.session_state.council)
        st.markdown(f"**Auto-response from Wyndham Council:**\n\n{ai_reply}")
        st.session_state.chat_history.append((user_input, ai_reply))

elif nav == "📥 Submit a Request":
    st.markdown("📌 Redirecting to your council’s website.")
    st.link_button("📝 Submit Online", "https://www.wyndham.vic.gov.au/request-it")

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
    st.link_button("Contact Sales", "mailto:sales@civreply.com?subject=CivReply%20Sales%20Enquiry")

elif nav == "ℹ️ About Us":
    st.markdown(
        """
        ### About Wyndham City & CivReply AI

        **Wyndham City** is one of Australia’s fastest-growing and most diverse municipalities, representing the vibrant communities of Werribee, Point Cook, Tarneit, and beyond. The council is dedicated to delivering world-class community services, sustainability initiatives, smart city technology, and transparent governance for all residents, visitors, and businesses.

        **CivReply AI** is an innovative AI-powered assistant that empowers citizens and staff to instantly access council documents, policies, and services. It enables fast, accurate answers, streamlined workflows, and modern digital experiences for everyone in Wyndham.

        *“Smarter answers for smarter communities.”*
        """,
        unsafe_allow_html=True
    )

elif nav == "⚙️ Admin Panel":
    if st.session_state.user_role != "Staff" or not st.session_state.admin_verified:
        st.warning("Restricted to council staff only. Please select 'Staff' and enter the password above.")
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

# (Optional footer)
st.markdown(
    "<div style='text-align:center; color:#b2c6d6; font-size:0.96rem; margin:32px 0 8px 0;'>Made with 🏛️ CivReply AI – for Australian councils, powered by AI</div>",
    unsafe_allow_html=True
)
