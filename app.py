import streamlit as st
import os
from datetime import datetime
import pandas as pd
from deep_translator import GoogleTranslator
from fpdf import FPDF

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import pdfplumber  # For form scraping

# ========== CONFIG ==========
WYNDHAM_BLUE = "#36A9E1"
WYNDHAM_DEEP = "#2078b2"
WYNDHAM_LIGHT = "#e3f3fa"
ADMIN_PASSWORD = "llama"        # Change to secure value in st.secrets
STAFF_PASSWORD = "staff2024"    # Change to secure value in st.secrets

LANGUAGES = {
    "English": "en",
    "Chinese": "zh-CN",
    "Arabic": "ar",
    "Spanish": "es",
    "Hindi": "hi",
    "Vietnamese": "vi",
    "Filipino": "tl",
    "Turkish": "tr",
    "French": "fr",
}

COUNCILS = [
    "Wyndham", "Melbourne", "Yarra", "Hume", "Geelong", "Brimbank", "Casey",
    "Darebin", "Maribyrnong", "Moonee Valley", "Moreland"
]

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

# ========== UTILS ==========

def translate_text(text, target_lang):
    if target_lang.lower() in ["english", "en"]:
        return text
    try:
        return GoogleTranslator(source="auto", target=target_lang.lower()).translate(text)
    except Exception as e:
        return f"[Translation error: {e}] {text}"

def send_ai_email(receiver, user_question, ai_answer, source_link):
    import yagmail
    GMAIL_USER = st.secrets["GMAIL_USER"]
    GMAIL_APP_PASSWORD = st.secrets["GMAIL_APP_PASSWORD"]
    subject = f"CivReply AI – Answer to your question: '{user_question}'"
    body = f"""
    Hello,

    Thank you for reaching out to your council!

    Here’s the answer to your question:
    ---
    {ai_answer}
    ---

    For more details, please see: {source_link}

    If you have more questions, just reply to this email.

    Best regards,  
    CivReply AI Team
    """
    yag = yagmail.SMTP(GMAIL_USER, GMAIL_APP_PASSWORD)
    yag.send(to=receiver, subject=subject, contents=body)
    # Optionally log the email to a CSV for staff review

def export_chats_to_pdf(chat_history, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "CivReply AI Chat Export", ln=True, align="C")
    pdf.ln(10)
    for idx, (q, a) in enumerate(chat_history, 1):
        pdf.multi_cell(0, 10, f"Q{idx}: {q}\nA{idx}: {a}\n\n")
    pdf.output(filename)

def log_feedback(text, email):
    with open("feedback_log.txt", "a") as f:
        entry = f"{datetime.now().isoformat()} | {email or 'anon'} | {text}\n"
        f.write(entry)

def usage_analytics():
    if not os.path.exists("usage_analytics.csv"):
        return pd.DataFrame(columns=["timestamp", "question", "council", "user", "plan"])
    return pd.read_csv("usage_analytics.csv")

def update_usage_analytics(question, council, user, plan):
    df = usage_analytics()
    new_row = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "council": council,
        "user": user,
        "plan": plan
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("usage_analytics.csv", index=False)

def advanced_form_scraping(pdf_path):
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                import re
                fields = re.findall(r"(Name|Email|Phone|Address|Date of Birth|Signature):?\s*(.+)", text)
                results.extend(fields)
    return results if results else [("No form fields found", "")]

def log_tip(user, council, tip):
    with open("community_tips.csv", "a") as f:
        entry = f"{datetime.now().isoformat()} | {user} | {council} | {tip}\n"
        f.write(entry)

# ========== STREAMLIT APP ==========

st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="wide")

# ---- Session state setup
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "query_count" not in st.session_state:
    st.session_state["query_count"] = 0
if "user_role" not in st.session_state:
    st.session_state["user_role"] = "Resident"
if "plan" not in st.session_state:
    st.session_state["plan"] = "basic"
if "council" not in st.session_state:
    st.session_state["council"] = "Wyndham"
if "session_start" not in st.session_state:
    st.session_state["session_start"] = datetime.now().isoformat()
if "admin_verified" not in st.session_state:
    st.session_state["admin_verified"] = False
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# ---- Authentication UI
st.sidebar.markdown("### 👤 User Login/Role")
if st.session_state["user_role"] == "Resident":
    login_choice = st.sidebar.selectbox("Select your role", ["Resident", "Staff", "Admin"])
    st.session_state["user_role"] = login_choice
    if login_choice == "Staff":
        pwd = st.sidebar.text_input("Staff password", type="password")
        if st.sidebar.button("Staff Login"):
            if pwd == STAFF_PASSWORD:
                st.session_state["logged_in"] = True
                st.success("Staff logged in.")
            else:
                st.error("Wrong password.")
    elif login_choice == "Admin":
        pwd = st.sidebar.text_input("Admin password", type="password")
        if st.sidebar.button("Admin Login"):
            if pwd == ADMIN_PASSWORD:
                st.session_state["logged_in"] = True
                st.session_state["admin_verified"] = True
                st.success("Admin logged in.")
            else:
                st.error("Wrong password.")
    else:
        st.session_state["logged_in"] = True  # Allow residents to use chat without password
else:
    st.session_state["logged_in"] = True

if not st.session_state["logged_in"]:
    st.warning("Please login to access full features.")
    st.stop()

# ---- Sidebar
st.sidebar.markdown("### 🏛️ CivReply AI – Australia-wide Councils")
council_selected = st.sidebar.selectbox("Select council", COUNCILS, index=COUNCILS.index(st.session_state["council"]))
st.session_state["council"] = council_selected

st.sidebar.markdown("### 🌏 Choose your language")
selected_lang_label = st.sidebar.selectbox("Language", list(LANGUAGES.keys()), index=0)
selected_lang = LANGUAGES[selected_lang_label]
st.session_state["language"] = selected_lang

st.sidebar.markdown("### 👤 Choose your plan")
plan_selected = st.sidebar.selectbox("Plan", list(PLAN_CONFIG.keys()), format_func=lambda x: PLAN_CONFIG[x]["label"])
st.session_state["plan"] = plan_selected

st.sidebar.markdown("---")
nav = st.sidebar.radio(
    "Navigation",
    [
        "💬 Chat with Council AI",
        "📥 Submit a Request",
        "📊 Stats & Analytics",
        "🗃️ Export Chats as PDF",
        "📦 Bulk Data Upload",
        "🎨 Council Branding",
        "📝 Form Scraper",
        "💡 Share Feedback",
        "📞 Contact Us",
        "ℹ️ About Us",
        "⚙️ Admin Panel",
        "🌐 Community Knowledge Base"
    ]
)

# ========== HEADER ==========
st.markdown(
    f"""
    <div style='background:linear-gradient(90deg,{WYNDHAM_BLUE},#7ecaf6 100%);padding:44px 0 24px 0;border-radius:0 0 44px 44px;box-shadow:0 10px 40px #cce5f7;display:flex;align-items:center;justify-content:center;gap:34px;'>
      <span style="font-size:4.2rem;line-height:1;border-radius:20px;background:rgba(255,255,255,0.09);box-shadow:0 0 22px #99cef4;padding:6px 28px 6px 20px;">🏛️</span>
      <span style='font-size:3.5rem;font-weight:900;color:#fff;letter-spacing:4px;text-shadow:0 2px 22px #36A9E180;'>CivReply AI</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ========== STATUS BAR ==========
st.markdown(
    f"""
    <div style='background:{WYNDHAM_LIGHT};border-radius:16px;padding:17px 48px;display:flex;justify-content:center;align-items:center;gap:60px;margin-top:20px;margin-bottom:14px;box-shadow:0 2px 10px #b4dbf2;'>
        <div style='color:{WYNDHAM_DEEP};font-size:1.13rem;font-weight:700'>🏛️ Council:</div>
        <div style='font-weight:700;'>{st.session_state["council"]}</div>
        <div style='color:{WYNDHAM_DEEP};font-size:1.13rem;font-weight:700'>📦 Plan:</div>
        <div style='font-weight:700;'>{PLAN_CONFIG[st.session_state["plan"]]["label"]}</div>
        <div style='color:{WYNDHAM_DEEP};font-size:1.13rem;font-weight:700'>🌐 Language:</div>
        <div style='font-weight:700;'>{selected_lang_label}</div>
        <div style='color:{WYNDHAM_DEEP};font-size:1.13rem;font-weight:700'>🟢 User:</div>
        <div style='font-weight:700;'>{st.session_state["user_role"]}</div>
    </div>
    """, unsafe_allow_html=True
)

# ========== MAIN ROUTES ==========

def ask_ai(question, council, lang="en", multi_council=False):
    plan = st.session_state["plan"]
    # Enforce query limits per plan!
    plan_limit = PLAN_CONFIG[plan]["limit"]
    if plan != "enterprise" and st.session_state["query_count"] >= plan_limit:
        st.warning(f"Query limit reached for {plan.capitalize()} plan. Please upgrade to continue.")
        st.stop()
    # Multi-council search stub
    embeddings = OpenAIEmbeddings()
    if not multi_council:
        index_path = f"index/{council.lower()}"
        if not os.path.exists(index_path):
            return "[Error] No index found for this council"
        db = FAISS.load_local(index_path, embeddings)
        retriever = db.as_retriever()
    else:
        # Future: Combine/merge all council indices
        return "Multi-council search is an Enterprise feature. Contact us to activate."
    llm_model = "gpt-3.5-turbo" if plan == "basic" else "gpt-4o"
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=llm_model)
    prompt = "You are a helpful council assistant. Only answer from the provided documents."
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    answer = qa.run(question)
    return answer

# --- Main App Router

if nav == "💬 Chat with Council AI":
    st.subheader(f"💬 Ask {st.session_state['council']} Council")
    if st.session_state["user_role"] == "Admin":
        multi_council_search = st.checkbox("Enterprise: Search ALL councils at once?", value=False)
    else:
        multi_council_search = False
    user_input = st.chat_input("Ask a question about council policies, forms, or documents…")
    if user_input:
        st.session_state.query_count += 1
        ai_reply = ask_ai(user_input, st.session_state["council"], lang=selected_lang, multi_council=multi_council_search)
        update_usage_analytics(user_input, st.session_state["council"], st.session_state["user_role"], st.session_state["plan"])
        ai_reply_translated = translate_text(ai_reply, selected_lang)
        st.markdown(f"**Auto-response from {st.session_state['council']} Council:**\n\n{ai_reply_translated}")
        st.session_state["chat_history"].append((user_input, ai_reply_translated))
        # PDF link extraction
        if "http" in ai_reply:
            import re
            links = re.findall(r'(https?://\S+)', ai_reply)
            for l in links:
                st.markdown(f"🔗 [View referenced document]({l})")

        # ==== EMAIL ANSWER FEATURE ====
        st.markdown("---")
        st.markdown("### 📧 Want this answer in your email?")
        receiver = st.text_input("Enter your email to receive this answer:", key="emailinput")
        source_link = links[0] if "links" in locals() and links else f"https://{st.session_state['council'].lower()}.vic.gov.au"
        if st.button("Send answer to my email"):
            if receiver and "@" in receiver and "." in receiver:
                try:
                    send_ai_email(receiver, user_input, ai_reply_translated, source_link)
                    st.success("✅ AI answer sent to your email!")
                except Exception as e:
                    st.error(f"❌ Failed to send email: {e}")
            else:
                st.error("Please enter a valid email address.")

elif nav == "📥 Submit a Request":
    st.markdown(f"📌 Redirecting to {st.session_state['council']} council’s website.")
    council_links = {
        "Wyndham": "https://www.wyndham.vic.gov.au/request-it",
        "Melbourne": "https://www.melbourne.vic.gov.au/contact-us/Pages/contact-us.aspx",
        # Add links for other councils
    }
    link = council_links.get(st.session_state["council"], "https://www.wyndham.vic.gov.au/request-it")
    st.link_button("📝 Submit Online", link)

elif nav == "📊 Stats & Analytics":
    if st.session_state["user_role"] not in ["Staff", "Admin"]:
        st.warning("Admins/Staff only. Please login.")
    else:
        st.header("📊 Usage Analytics Dashboard")
        df = usage_analytics()
        st.dataframe(df.tail(100), use_container_width=True)
        st.metric("Total Questions (all time)", len(df))
        st.metric("Session Start", st.session_state["session_start"])
        st.metric("Current Council", st.session_state["council"])
        st.metric("Current Plan", PLAN_CONFIG[st.session_state["plan"]]["label"])

elif nav == "🗃️ Export Chats as PDF":
    st.header("🗃️ Export Your Chat History")
    if st.button("Export as PDF"):
        filename = f"civreply_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        export_chats_to_pdf(st.session_state["chat_history"], filename)
        with open(filename, "rb") as f:
            st.download_button("Download PDF", data=f, file_name=filename)

elif nav == "📦 Bulk Data Upload":
    st.header("📦 Bulk PDF/Data Upload (Staff/Admin Only)")
    if st.session_state["user_role"] not in ["Staff", "Admin"]:
        st.warning("Staff/Admin access required.")
    else:
        docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Bulk Index & Add"):
            if docs:
                folder = f"docs/{st.session_state['council'].lower()}"
                os.makedirs(folder, exist_ok=True)
                for d in docs:
                    with open(os.path.join(folder, d.name), "wb") as f:
                        f.write(d.read())
                loader = PyPDFDirectoryLoader(folder)
                raw_docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(raw_docs)
                vecdb = FAISS.from_documents(chunks, OpenAIEmbeddings())
                vecdb.save_local(f"index/{st.session_state['council'].lower()}")
                st.success("✅ Bulk indexing complete.")
            else:
                st.warning("Please upload at least one PDF.")

elif nav == "🎨 Council Branding":
    st.header("🎨 Custom Council Branding")
    st.markdown("Upload branding, logos, and themes per council. (Admins only, not persistent on free Render)")
    if st.session_state["user_role"] != "Admin":
        st.warning("Admin access required.")
    else:
        uploaded_logo = st.file_uploader("Upload council logo", type=["png", "jpg", "jpeg"])
        if uploaded_logo:
            st.image(uploaded_logo, width=180)
            st.success("Logo uploaded! (Demo only)")

elif nav == "📝 Form Scraper":
    st.header("📝 Advanced Form Scraping (PDF)")
    pdf_file = st.file_uploader("Upload a PDF form to auto-extract fields", type="pdf")
    if pdf_file and st.button("Extract Form Data"):
        with open("temp_form.pdf", "wb") as f:
            f.write(pdf_file.read())
        extracted = advanced_form_scraping("temp_form.pdf")
        st.write(pd.DataFrame(extracted, columns=["Field", "Value"]))

elif nav == "💡 Share Feedback":
    st.header("💡 Share Feedback")
    fb = st.text_area("Tell us what’s working or not...")
    email = st.text_input("Email (optional)")
    if st.button("📨 Submit Feedback"):
        log_feedback(fb, email)
        st.success("Thanks for helping improve CivReply AI!")

elif nav == "📞 Contact Us":
    st.header("📞 Contact")
    st.markdown("**Call:** (03) 9742 0777")
    st.markdown("**Visit:** 45 Princes Hwy, Werribee")
    st.markdown("**Mail:** PO Box 197")
    st.link_button("Website", "https://www.wyndham.vic.gov.au")
    st.link_button("Contact Sales", "mailto:sales@civreply.com?subject=CivReply%20Sales%20Enquiry")

elif nav == "ℹ️ About Us":
    st.header("About CivReply AI")
    st.markdown("""
        CivReply AI is an innovative AI-powered assistant that empowers citizens and staff to instantly access council documents, policies, and services. It enables fast, accurate answers, streamlined workflows, and modern digital experiences for everyone in Wyndham and beyond.
        *“Smarter answers for smarter communities.”*
    """)

elif nav == "⚙️ Admin Panel":
    st.header("⚙️ Admin Panel")
    if st.session_state["user_role"] != "Admin":
        st.warning("Admin access required.")
    else:
        docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Rebuild Index"):
            if docs:
                folder = f"docs/{st.session_state['council'].lower()}"
                os.makedirs(folder, exist_ok=True)
                for d in docs:
                    with open(os.path.join(folder, d.name), "wb") as f:
                        f.write(d.read())
                loader = PyPDFDirectoryLoader(folder)
                raw_docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(raw_docs)
                vecdb = FAISS.from_documents(chunks, OpenAIEmbeddings())
                vecdb.save_local(f"index/{st.session_state['council'].lower()}")
                st.success("✅ Index rebuilt successfully.")
            else:
                st.warning("Please upload at least one document.")

elif nav == "🌐 Community Knowledge Base":
    st.header("🌐 Community Knowledge Base")
    st.markdown("Browse tips from other residents & staff, or add your own!")
    if os.path.exists("community_tips.csv"):
        tips = pd.read_csv("community_tips.csv", sep="|", names=["Timestamp", "User", "Council", "Tip"])
        st.dataframe(tips[["Timestamp", "Council", "Tip"]].tail(50), use_container_width=True)
    tip = st.text_area("Suggest a new tip or local info to share")
    if st.button("Add Tip"):
        user = st.session_state["user_role"]
        log_tip(user, st.session_state["council"], tip)
        st.success("Tip added! Pending admin approval.")

# ========== FOOTER ==========
st.markdown(
    "<div style='text-align:center; color:#b2c6d6; font-size:0.96rem; margin:32px 0 8px 0;'>Made with 🏛️ CivReply AI – for Australian councils, powered by AI</div>",
    unsafe_allow_html=True
)
