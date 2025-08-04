import streamlit as st
import os
import pandas as pd
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from deep_translator import GoogleTranslator
import yagmail
from fpdf import FPDF
import io

# ==== CONFIG & CONSTANTS ====
WYNDHAM_BLUE = "#36A9E1"
WYNDHAM_DEEP = "#2078b2"
WYNDHAM_LIGHT = "#e3f3fa"
COUNCILS = ["Wyndham", "Yarra", "Casey", "Hume", "Greater Geelong"]  # Expand as needed
LANGUAGES = ["English", "Spanish", "Chinese", "Hindi", "Arabic", "French"]
LANG_CODES = {
    "English": "en", "Spanish": "es", "Chinese": "zh-CN", "Hindi": "hi", "Arabic": "ar", "French": "fr"
}
PLAN_CONFIG = {
    "basic": {
        "label": "Basic ($499 AUD/mo)", "icon": "💧", "limit": 500,
        "features": [
            "📄 PDF Q&A (ask about any council document)",
            "🔒 Limit: 500 queries", "📧 Email support (24h response)",
            "📚 Council policy finder", "📱 Mobile access",
            "☁️ Secure cloud storage", "👥 Community knowledge base"
        ],
    },
    "standard": {
        "label": "Standard ($1,499 AUD/mo)", "icon": "🚀", "limit": 2000,
        "features": [
            "✅ Everything in Basic", "🔒 Limit: 2,000 queries",
            "📝 Form Scraping (auto-extract info from forms)",
            "⚡ Immediate email & chat support", "📊 Usage analytics dashboard",
            "🗃️ PDF export of chats", "🌏 Multi-language Q&A",
            "📦 Bulk data uploads", "🎨 Custom council branding",
            "<b>Contact sales to upgrade below</b>"
        ],
    },
    "enterprise": {
        "label": "Enterprise ($2,999+ AUD/mo)", "icon": "🏆", "limit": float("inf"),
        "features": [
            "✅ Everything in Standard", "🔓 Limit: Unlimited queries",
            "👤 Dedicated account manager", "🔌 API access & automation",
            "🛡️ SLA: 99.9% uptime", "🔐 Single Sign-On (SSO)",
            "🧑‍🏫 Staff training sessions", "🤖 Integration with 3rd party tools",
            "☁️ On-premise/cloud deployment options", "🛠️ Custom workflow automations",
            "<b>Contact sales to upgrade below</b>"
        ],
    }
}

# -- ENV/SECRET HANDLING
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
GMAIL_USER = st.secrets.get("GMAIL_USER", os.getenv("GMAIL_USER"))
GMAIL_APP_PASSWORD = st.secrets.get("GMAIL_APP_PASSWORD", os.getenv("GMAIL_APP_PASSWORD"))

if not OPENAI_KEY:
    st.error("❌ Missing OpenAI API Key. Please set it in Streamlit secrets or env.")
    st.stop()

st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="wide")
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("query_count", 0)
st.session_state.setdefault("user_role", "Resident")
st.session_state.setdefault("plan", "basic")
st.session_state.setdefault("language", "English")
st.session_state.setdefault("council", "Wyndham")
st.session_state.setdefault("session_start", datetime.now().isoformat())
st.session_state.setdefault("admin_verified", False)

# ==== TRANSLATION ====
def translate_answer(answer, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(answer)
    except Exception as e:
        return f"[Translation failed: {e}]"

# ==== EMAIL ====
def send_ai_email(receiver, user_question, ai_answer, source_link):
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        raise Exception("Gmail credentials not set in secrets or env")
    subject = f"CivReply AI – Answer to your question: '{user_question}'"
    body = f"""Hello,

Thank you for reaching out to {st.session_state.council} Council!

Here’s the answer to your question:
---
{ai_answer}
---

For more details, please see: {source_link}

If you have more questions, just reply to this email.

Best regards,  
CivReply AI Team"""
    yag = yagmail.SMTP(GMAIL_USER, GMAIL_APP_PASSWORD)
    yag.send(to=receiver, subject=subject, contents=body)

# ==== PDF EXPORT CHAT ====
def export_chat_to_pdf(chat_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "CivReply AI – Chat Transcript", ln=1, align='C')
    pdf.ln(8)
    for i, (q, a) in enumerate(chat_history):
        pdf.set_font("Arial", "B", size=12)
        pdf.multi_cell(0, 8, f"Q{i+1}: {q}")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, f"A{i+1}: {a}")
        pdf.ln(2)
    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# ==== FORM SCRAPING (basic: extracts text fields from a PDF) ====
def extract_fields_from_pdf(uploaded_pdf):
    import pdfplumber
    fields = []
    with pdfplumber.open(uploaded_pdf) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                for line in lines:
                    if ":" in line and len(line.split(":")[0]) < 50:  # crude field detector
                        fields.append(line.strip())
    return fields if fields else ["No obvious form fields detected."]

# ==== ASK AI ====
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
    else:
        llm = ChatOpenAI(api_key=OPENAI_KEY, model="gpt-4o")
        prompt = "You are a council AI with advanced analytics and form scraping. Always respond in English."
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

def log_usage():
    with open("usage_log.csv", "a") as f:
        entry = f"{datetime.now().isoformat()},{st.session_state.council},{st.session_state.query_count},{st.session_state.user_role}\n"
        f.write(entry)

# ==== UI LAYOUT ====
st.markdown(
    f"""<div style='background:linear-gradient(90deg,{WYNDHAM_BLUE},#7ecaf6 100%);padding:36px 0 20px 0;border-radius:0 0 44px 44px;box-shadow:0 10px 40px #cce5f7;display:flex;align-items:center;justify-content:center;gap:34px;'>
      <span style="font-size:4.2rem;line-height:1;border-radius:20px;background:rgba(255,255,255,0.09);box-shadow:0 0 22px #99cef4;padding:6px 28px 6px 20px;">🏛️</span>
      <span style='font-size:3.5rem;font-weight:900;color:#fff;letter-spacing:4px;text-shadow:0 2px 22px #36A9E180;'>CivReply AI</span>
    </div>""",
    unsafe_allow_html=True
)
st.markdown(
    f"""<div style='background:{WYNDHAM_LIGHT};border-radius:16px;padding:14px 36px;display:flex;justify-content:center;align-items:center;gap:48px;margin-top:12px;margin-bottom:14px;box-shadow:0 2px 10px #b4dbf2;'>
        <div style='color:{WYNDHAM_DEEP};font-size:1.07rem;font-weight:700'>🏛️ Active Council:</div>
        <div style='font-weight:700;'>{st.session_state.get('council', 'Wyndham')}</div>
        <div style='color:{WYNDHAM_DEEP};font-size:1.07rem;font-weight:700'>📦 Plan:</div>
        <div style='font-weight:700;'>{PLAN_CONFIG.get(st.session_state.get('plan'), PLAN_CONFIG['basic'])["label"]}</div>
        <div style='color:{WYNDHAM_DEEP};font-size:1.07rem;font-weight:700'>🌐 Language:</div>
        <div style='font-weight:700;'>{st.session_state.get('language', 'English')}</div>
    </div>""", unsafe_allow_html=True
)

# === Sidebar ===
with st.sidebar:
    st.markdown(
        f"""<div style='background:{WYNDHAM_BLUE};padding:20px 0 14px 0;border-radius:0 0 32px 32px;box-shadow:0 4px 18px #cce5f7;margin-bottom:10px;'>
          <div style='display:flex;align-items:center;justify-content:center;gap:15px;'>
            <span style="font-size:2.5rem;">🏛️</span>
            <span style='font-size:1.65rem;font-weight:800;color:#fff;letter-spacing:0.7px;'>CivReply AI</span>
          </div>
        </div>""",
        unsafe_allow_html=True
    )
    council_sel = st.selectbox("Choose Council:", COUNCILS, index=COUNCILS.index(st.session_state.council))
    st.session_state.council = council_sel
    nav = st.radio(
        "",
        [
            "💬 Chat with Council AI",
            "📥 Submit a Request",
            "📊 Stats & Session",
            "📈 Analytics Dashboard",
            "💡 Share Feedback",
            "📞 Contact Us",
            "ℹ️ About Us",
            "⚙️ Admin Panel"
        ],
        label_visibility="collapsed"
    )
    # Recent chats
    st.markdown("<div style='text-align:center;font-size:1.12rem;font-weight:700;color:#235b7d;margin:16px 0 0 0;'>Recent Chats</div>", unsafe_allow_html=True)
    last_5 = [q for q, a in st.session_state.get("chat_history", [])[-5:]]
    if last_5:
        for q in reversed(last_5):
            st.markdown(f"<div style='padding:10px 0; text-align:center; font-size:15.5px;color:#2078b2;'>{q}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#7eb7d8;text-align:center;display:block;'>No chats yet</span>", unsafe_allow_html=True)
    # Upgrade plan
    with st.expander("🚀 Upgrade Your Plan", expanded=False):
        for plan_key, plan in PLAN_CONFIG.items():
            st.markdown(
                f"""<div style='background:linear-gradient(145deg,#f2fbfe 60%,#cbe7f8 100%);border-radius:18px;box-shadow:0 4px 18px #c1e3f4;padding:18px 14px 10px 14px;margin-bottom:12px;'>
                  <div style="font-size:1.17rem;font-weight:900;color:#158ed8;margin-bottom:8px;">{plan['icon']} {plan['label'].split('(')[0]}</div>
                  <div style="font-size:1.3rem;font-weight:800;color:{WYNDHAM_BLUE};margin-bottom:6px;">{plan['label'].split('(')[1][:-4]} AUD</div>
                  <ul style="padding-left:18px;font-size:1.08rem;line-height:1.7;">
                    {''.join([f"<li style='margin-bottom:3px;color:#1374ab'>{f}</li>" for f in plan['features']])}
                  </ul>
                  {'<div style="margin-top:8px;"><a href="mailto:sales@civreply.com?subject=CivReply%20Plan%20Upgrade%20Enquiry" style="background:#36A9E1;color:#fff;font-weight:700;padding:8px 18px;border-radius:10px;text-decoration:none;display:inline-block;font-size:1.03rem;box-shadow:0 2px 6px #bae3fc;">Contact Sales</a></div>' if plan_key in ['standard', 'enterprise'] else ''}
                </div>""",
                unsafe_allow_html=True
            )

# ==== PLAN LIMIT ====
plan_id = st.session_state.plan
plan_limit = PLAN_CONFIG[plan_id]["limit"]
if st.session_state.query_count >= plan_limit:
    st.warning(f"❗ You’ve reached the {PLAN_CONFIG[plan_id]['label']} limit.")
    st.stop()

# ==== MAIN APP ROUTER ====
if nav == "💬 Chat with Council AI":
    st.subheader("💬 Ask your Council")
    selected_language = st.selectbox("🌏 Choose answer language:", LANGUAGES, index=LANGUAGES.index(st.session_state.language))
    st.session_state.language = selected_language
    user_input = st.chat_input("Ask a question about policies, forms, or documents…")
    if user_input:
        st.session_state.query_count += 1
        log_usage()
        ai_reply = ask_ai(user_input, st.session_state.council)
        # Translate if needed
        if selected_language != "English":
            translated_reply = translate_answer(ai_reply, LANG_CODES[selected_language])
            st.markdown(f"**Auto-response ({selected_language}):**\n\n{translated_reply}")
            st.markdown("---")
            st.markdown(f"**(English version):**\n\n{ai_reply}")
        else:
            st.markdown(f"**Auto-response:**\n\n{ai_reply}")
        st.session_state.chat_history.append((user_input, ai_reply))
        # Export and email UI
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📄 Export Chat to PDF"):
                pdf_file = export_chat_to_pdf(st.session_state.chat_history)
                st.download_button("Download PDF", pdf_file, file_name="CivReplyAI_Chat.pdf")
        with col2:
            receiver = st.text_input("Enter your email to receive this answer:", key=f"emailinput{st.session_state.query_count}")
            source_link = f"https://www.{st.session_state.council.lower()}.vic.gov.au"
            if st.button("Send answer to my email", key=f"sendemail{st.session_state.query_count}"):
                if receiver and "@" in receiver and "." in receiver:
                    try:
                        send_ai_email(receiver, user_input, ai_reply, source_link)
                        st.success("✅ AI answer sent to your email!")
                    except Exception as e:
                        st.error(f"❌ Failed to send email: {e}")
                else:
                    st.error("Please enter a valid email address.")
        with col3:
            st.write(" ")

elif nav == "📈 Analytics Dashboard":
    st.subheader("📈 Usage Analytics")
    if os.path.exists("usage_log.csv"):
        df = pd.read_csv("usage_log.csv", names=["datetime", "council", "query_count", "user_role"])
        st.dataframe(df.tail(50))
        st.metric("Total Qs", df['query_count'].sum())
        st.metric("Unique Councils", df['council'].nunique())
    else:
        st.info("No analytics data yet.")

elif nav == "📥 Submit a Request":
    st.markdown("📌 Redirecting to your council’s website.")
    st.link_button("📝 Submit Online", f"https://www.{st.session_state.council.lower()}.vic.gov.au/request-it")

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
    st.link_button("Website", f"https://www.{st.session_state.council.lower()}.vic.gov.au")
    st.link_button("Contact Sales", "mailto:sales@civreply.com?subject=CivReply%20Sales%20Enquiry")

elif nav == "ℹ️ About Us":
    st.markdown(
        """
        ### About CivReply AI

        CivReply AI is an innovative AI-powered assistant that empowers citizens and staff to instantly access council documents, policies, and services. It enables fast, accurate answers, streamlined workflows, and modern digital experiences for everyone in your council.

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
        st.subheader("📝 Form Scraping (extract fields from PDF form)")
        uploaded_form = st.file_uploader("Upload a form PDF to auto-extract fields", type="pdf", key="form")
        if uploaded_form:
            fields = extract_fields_from_pdf(uploaded_form)
            st.write("Detected form fields:")
            for field in fields:
                st.write(f"- {field}")

# === Footer ===
st.markdown(
    "<div style='text-align:center; color:#b2c6d6; font-size:0.96rem; margin:32px 0 8px 0;'>Made with 🏛️ CivReply AI – for Australian councils, powered by AI</div>",
    unsafe_allow_html=True
)
