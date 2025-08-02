import streamlit as st
import os
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

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

# --- THEMED HEADER with LOGO & GRADIENT ---
st.markdown(
    f"""
    <div style='background:linear-gradient(90deg,{WYNDHAM_BLUE},#7ecaf6 90%);padding:28px 0 16px 0;border-radius:0 0 34px 34px;box-shadow:0 6px 24px #cce5f7;'>
      <div style='display:flex;align-items:center;justify-content:center;gap:20px;'>
        <img src="https://www.wyndham.vic.gov.au/sites/default/files/styles/small/public/2020-06/logo_0.png" width="66" style="border-radius:10px;box-shadow:0 0 12px #83caec;">
        <span style='font-size:2.9rem;font-weight:900;color:#fff;letter-spacing:2px;text-shadow:0 2px 10px #36A9E155;'>CivReply AI</span>
      </div>
    </div>
    """, unsafe_allow_html=True
)

# PLAN + STATUS BADGE
st.markdown(
    f"""
    <div style='background:{WYNDHAM_LIGHT};border-radius:16px;padding:15px 40px;display:flex;justify-content:center;align-items:center;gap:60px;margin-top:18px;margin-bottom:12px;box-shadow:0 2px 10px #b4dbf2;'>
        <div style='color:{WYNDHAM_DEEP};font-size:1.12rem;font-weight:700'>🏛️ Active Council:</div>
        <div style='font-weight:700;'>{st.session_state.council}</div>
        <div style='color:{WYNDHAM_DEEP};font-size:1.12rem;font-weight:700'>📦 Plan:</div>
        <div style='font-weight:700;'>{PLAN_CONFIG[st.session_state.plan]["label"]}</div>
        <div style='color:{WYNDHAM_DEEP};font-size:1.12rem;font-weight:700'>🌐 Language:</div>
        <div style='font-weight:700;'>English</div>
    </div>
    """, unsafe_allow_html=True
)

# ROLE SELECTOR CENTERED WITH CUSTOM ICON
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("<div style='margin-top:15px;font-size:1.13rem;font-weight:700;color:#2078b2;text-align:center;'>👤 Select Role</div>", unsafe_allow_html=True)
    user_role = st.selectbox(
        "",
        options=["Resident", "Staff", "Visitor"],
        index=["Resident", "Staff", "Visitor"].index(st.session_state.get("user_role", "Resident")),
        label_visibility="collapsed"
    )
    st.session_state.user_role = user_role
    if user_role == "Staff" and not st.session_state.admin_verified:
        pw = st.text_input("Enter Staff Password", type="password", key="pwcheck")
        if pw and pw == ADMIN_PASSWORD:
            st.session_state.admin_verified = True
            st.success("Staff access granted! 🎉")
            st.balloons()
        elif pw and pw != ADMIN_PASSWORD:
            st.error("Incorrect password. Please try again.")
            st.session_state.user_role = "Resident"
    if user_role == "Staff" and st.session_state.admin_verified:
        st.markdown("<div style='margin-top:10px;font-weight:700;text-align:center;'>🛠️ Admin Plan Control</div>", unsafe_allow_html=True)
        st.session_state.plan = st.selectbox(
            "",
            options=["basic", "standard", "enterprise"],
            format_func=lambda x: PLAN_CONFIG[x]["label"],
            key="admin_plan_selector",
            label_visibility="collapsed"
        )
    elif user_role != "Staff":
        st.session_state.admin_verified = False

st.markdown("<hr style='margin-top:26px;margin-bottom:5px;border:1.2px solid #d8eafe;border-radius:7px;'>", unsafe_allow_html=True)

# === SIDEBAR (with custom emoji icons) ===
with st.sidebar:
    st.markdown(
        f"""
        <div style='background:{WYNDHAM_BLUE};padding:24px 0 16px 0;border-radius:0 0 32px 32px;box-shadow:0 4px 18px #cce5f7;margin-bottom:18px;'>
          <div style='display:flex;align-items:center;justify-content:center;gap:13px;'>
            <img src="https://www.wyndham.vic.gov.au/sites/default/files/styles/small/public/2020-06/logo_0.png" width="44" style="border-radius:8px;box-shadow:0 0 10px #83caec;">
            <span style='font-size:1.6rem;font-weight:800;color:#fff;letter-spacing:0.5px;'>CivReply AI</span>
          </div>
        </div>
        """, unsafe_allow_html=True
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
    st.markdown("<div style='text-align:center;font-size:1.12rem;font-weight:700;color:#235b7d;margin:16px 0 0 0;'>Recent Chats</div>", unsafe_allow_html=True)
    last_5 = [q for q, a in st.session_state.chat_history[-5:]]
    if last_5:
        for q in reversed(last_5):
            st.markdown(f"<div style='padding:10px 0; text-align:center; font-size:15.5px;color:#2078b2;'>{q}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#7eb7d8;text-align:center;display:block;'>No chats yet</span>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='background:{WYNDHAM_LIGHT};border-radius:20px;padding:18px 20px;margin-top:25px;margin-bottom:6px;box-shadow:0 2px 14px #b4dbf2;'>
            <div style='font-size:1.25rem;font-weight:800;color:#158ed8;margin-bottom:8px;'>🚀 Upgrade Your Plan</div>
            <div style='margin-bottom:13px;line-height:1.55;color:#1e4666;'>Unlock more features, higher limits, integrations, automations, and dedicated support with Standard or Enterprise plans.</div>
            <a href='#Upgrades' style='background:{WYNDHAM_BLUE};color:#fff;font-weight:700;padding:10px 26px;border-radius:11px;text-decoration:none;display:inline-block;font-size:1.09rem;margin-top:7px;transition:background 0.2s;'>✨ View Plans</a>
        </div>
        """, unsafe_allow_html=True
    )

# === PLAN-SPECIFIC AI LOGIC (enforcing English answers) ===
def ask_ai(question, council):
    plan = st.session_state.get("plan", "basic")
    embeddings = OpenAIEmbeddings()
    index_path = f"index/{council.lower()}"
    if not os.path.exists(index_path):
        return "[Error] No index found for this council"
    db = FAISS.load_local(index_path, embeddings)
    retriever = db.as_retriever()
    # Always answer in English, even if the input is in another language.
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
        st.success("Thanks for helping improve CivReply AI! 🎉")

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

# --- Upgrades Section (hidden anchor for sidebar button) ---
st.markdown("<a name='Upgrades'></a>", unsafe_allow_html=True)
if st.session_state.get("plan") in ["basic", "standard", "enterprise"]:
    st.markdown(
        f"""
        <div style='margin-top:40px;'>
            <h2 style="color:{WYNDHAM_DEEP};font-size:2.05rem;margin-bottom:14px;font-weight:900;">Compare business tiers</h2>
            <div style="display: flex; gap: 24px; margin-bottom:24px;">
        """, unsafe_allow_html=True
    )
    cols = st.columns(3)
    for i, (plan_key, plan) in enumerate(PLAN_CONFIG.items()):
        with cols[i]:
            price = plan['label'].split('(')[1][:-4]
            st.markdown(
                f"""
                <div style='background:linear-gradient(145deg,#f2fbfe 60%,#cbe7f8 100%);border-radius:22px;box-shadow:0 8px 30px #c1e3f4;padding:32px 22px 20px 22px;margin-bottom:20px;min-height:420px;transition:box-shadow 0.2s;'>
                  <div style="font-size:1.4rem;font-weight:900;color:#158ed8;margin-bottom:10px;">{plan['icon']} {plan['label'].split('(')[0]}</div>
                  <div style="font-size:2rem;font-weight:800;color:{WYNDHAM_BLUE};margin-bottom:6px;">{price} AUD</div>
                  <div style='color:#555;margin-bottom:20px;'>/ month</div>
                  <div style="margin-bottom:10px;">
                    <ul style="padding-left:18px;font-size:1.12rem;line-height:1.7;">
                      {''.join([f"<li style='margin-bottom:4px;color:#1374ab'>{f}</li>" for f in plan['features']])}
                    </ul>
                  </div>
                  {'<div style="margin-top:18px;"><a href="mailto:sales@civreply.com?subject=CivReply%20Plan%20Upgrade%20Enquiry" style="background:#36A9E1;color:#fff;font-weight:700;padding:10px 28px;border-radius:12px;text-decoration:none;display:inline-block;font-size:1.13rem;box-shadow:0 2px 6px #bae3fc;">Contact Sales</a></div>' if plan_key in ['standard', 'enterprise'] else ''}
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("</div></div>", unsafe_allow_html=True)
