import streamlit as st
import os
import pandas as pd
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from googletrans import Translator
from fpdf import FPDF
import pdfplumber

# ========= COUNCIL BRANDING CONFIG ==========
COUNCIL_CONFIG = {
    "Wyndham": {"logo": "wyndham_logo.png", "color": "#36A9E1"},
    "Yarra":   {"logo": "yarra_logo.png",   "color": "#a345f0"},
    # Add more councils here...
}
DEFAULT_COUNCIL = "Wyndham"

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

LANGUAGES = {
    "English": "en",
    "Chinese": "zh-cn",
    "Arabic": "ar",
    "Vietnamese": "vi",
    "Hindi": "hi"
}

st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="wide")

# ========= SECRETS & ENV ==========
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
GMAIL_USER = st.secrets.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = st.secrets.get("GMAIL_APP_PASSWORD", "")

if not OPENAI_KEY:
    st.error("❌ Missing OpenAI API Key. Please set it in .env or secrets.toml.")
    st.stop()

# ========= SESSION STATE ==========
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("query_count", 0)
st.session_state.setdefault("user_role", "Resident")
st.session_state.setdefault("plan", "basic")
st.session_state.setdefault("language", "English")
st.session_state.setdefault("council", DEFAULT_COUNCIL)
st.session_state.setdefault("session_start", datetime.now().isoformat())
st.session_state.setdefault("admin_verified", False)

# ========= THEME/BRANDING ==========
council = st.session_state.get("council", DEFAULT_COUNCIL)
color = COUNCIL_CONFIG.get(council, COUNCIL_CONFIG[DEFAULT_COUNCIL])["color"]
logo_path = COUNCIL_CONFIG.get(council, COUNCIL_CONFIG[DEFAULT_COUNCIL])["logo"]

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=120)
st.markdown(f"<style>:root {{ --main-color: {color}; }}</style>", unsafe_allow_html=True)

# ========= HEADER ==========
st.markdown(
    f"""
    <div style='background:linear-gradient(90deg,{color},#7ecaf6 100%);padding:44px 0 24px 0;border-radius:0 0 44px 44px;box-shadow:0 10px 40px #cce5f7;display:flex;align-items:center;justify-content:center;gap:34px;'>
      <span style="font-size:4.2rem;line-height:1;border-radius:20px;background:rgba(255,255,255,0.09);box-shadow:0 0 22px #99cef4;padding:6px 28px 6px 20px;">🏛️</span>
      <span style='font-size:3.5rem;font-weight:900;color:#fff;letter-spacing:4px;text-shadow:0 2px 22px {color}80;'>CivReply AI</span>
    </div>
    """, unsafe_allow_html=True
)

# ========= STATUS BAR ==========
plan = st.session_state.get("plan", "basic")
plan_label = PLAN_CONFIG.get(plan, PLAN_CONFIG["basic"])["label"]

st.markdown(
    f"""
    <div style='background:#e3f3fa;border-radius:16px;padding:17px 48px;display:flex;justify-content:center;align-items:center;gap:60px;margin-top:20px;margin-bottom:14px;box-shadow:0 2px 10px #b4dbf2;'>
        <div style='color:{color};font-size:1.13rem;font-weight:700'>🏛️ Active Council:</div>
        <div style='font-weight:700;'>{council}</div>
        <div style='color:{color};font-size:1.13rem;font-weight:700'>📦 Plan:</div>
        <div style='font-weight:700;'>{plan_label}</div>
        <div style='color:{color};font-size:1.13rem;font-weight:700'>🌐 Language:</div>
        <div style='font-weight:700;'>{st.session_state.language}</div>
    </div>
    """, unsafe_allow_html=True
)

# ========= SIDEBAR ==========
with st.sidebar:
    nav = st.radio(
        "",
        [
            "💬 Chat with Council AI",
            "📥 Submit a Request",
            "📊 Stats & Session",
            "🗃️ Export Chat",
            "💡 Share Feedback",
            "📞 Contact Us",
            "ℹ️ About Us",
            "⚙️ Admin Panel"
        ],
        label_visibility="collapsed"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    # Recent Chats
    st.markdown("<div style='text-align:center;font-size:1.12rem;font-weight:700;color:#235b7d;margin:16px 0 0 0;'>Recent Chats</div>", unsafe_allow_html=True)
    last_5 = [q for q, a in st.session_state.get("chat_history", [])[-5:]]
    if last_5:
        for q in reversed(last_5):
            st.markdown(f"<div style='padding:10px 0; text-align:center; font-size:15.5px;color:{color};'>{q}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#7eb7d8;text-align:center;display:block;'>No chats yet</span>", unsafe_allow_html=True)

    st.markdown("### <br>", unsafe_allow_html=True)
    # Plan upgrade
    with st.expander("🚀 Upgrade Your Plan", expanded=False):
        for plan_key, p in PLAN_CONFIG.items():
            st.markdown(
                f"""
                <div style='background:linear-gradient(145deg,#f2fbfe 60%,#cbe7f8 100%);border-radius:18px;box-shadow:0 4px 18px #c1e3f4;padding:18px 14px 10px 14px;margin-bottom:12px;'>
                  <div style="font-size:1.17rem;font-weight:900;color:{color};margin-bottom:8px;">{p['icon']} {p['label'].split('(')[0]}</div>
                  <div style="font-size:1.3rem;font-weight:800;color:{color};margin-bottom:6px;">{p['label'].split('(')[1][:-4]} AUD</div>
                  <ul style="padding-left:18px;font-size:1.08rem;line-height:1.7;">
                    {''.join([f"<li style='margin-bottom:3px;color:{color}'>{f}</li>" for f in p['features']])}
                  </ul>
                  {'<div style="margin-top:8px;"><a href="mailto:sales@civreply.com?subject=CivReply%20Plan%20Upgrade%20Enquiry" style="background:{color};color:#fff;font-weight:700;padding:8px 18px;border-radius:10px;text-decoration:none;display:inline-block;font-size:1.03rem;box-shadow:0 2px 6px #bae3fc;">Contact Sales</a></div>' if plan_key in ['standard', 'enterprise'] else ''}
                </div>
                """,
                unsafe_allow_html=True
            )

# ========= UTILS ==========
def log_query(user, question, council, lang):
    file = "usage_log.csv"
    row = {"user": user, "question": question, "council": council, "language": lang, "timestamp": datetime.now().isoformat()}
    if not os.path.exists(file):
        df = pd.DataFrame([row])
    else:
        df = pd.read_csv(file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(file, index=False)

def export_chat_to_pdf(chat_history, filename="chat_history.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="CivReply AI Chat History", ln=True, align='C')
    for i, (q, a) in enumerate(chat_history):
        pdf.multi_cell(0, 10, f"Q{i+1}: {q}\nA{i+1}: {a}\n")
    pdf.output(filename)
    return filename

def translate(text, dest_lang):
    if dest_lang == "en":
        return text
    try:
        translator = Translator()
        return translator.translate(text, dest=dest_lang).text
    except Exception:
        return text

def send_ai_email(receiver, user_question, ai_answer, source_link):
    import yagmail
    subject = f"CivReply AI – Answer to your question: '{user_question}'"
    body = f"""
    Hello,

    Thank you for reaching out to {council} Council!

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

def advanced_form_scrape(pdf_file):
    # Extract text
    with pdfplumber.open(pdf_file) as pdf:
        text = " ".join([page.extract_text() or "" for page in pdf.pages])
    # Now, pass text to LLM for auto-extraction of form fields
    llm = ChatOpenAI(api_key=OPENAI_KEY, model="gpt-3.5-turbo")
    prompt = f"""The following text is extracted from a council form PDF. Identify key fields (name, address, dates, numbers, options, signatures) and their filled values. If blank, mark as 'Not filled'. Present the output as a table with 'Field' and 'Value' columns.

Form Text:
{text[:4000]}"""  # Truncate for prompt safety
    return llm.invoke(prompt).content

# ========= PLAN-SPECIFIC AI LOGIC ==========
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
    answer = qa.run(question)
    # Translation
    lang_code = LANGUAGES[st.session_state.language]
    if lang_code != "en":
        answer = translate(answer, lang_code)
    return answer

def log_feedback(text, email):
    with open("feedback_log.txt", "a") as f:
        entry = f"{datetime.now().isoformat()} | {email or 'anon'} | {text}\n"
        f.write(entry)

# ===== MAIN APP ROUTER =====
if nav == "💬 Chat with Council AI":
    st.subheader(f"💬 Ask {council} Council")
    # Multi-language selector
    lang_sel = st.selectbox("Choose language", LANGUAGES.keys(), index=list(LANGUAGES).index(st.session_state.language))
    st.session_state.language = lang_sel

    user_input = st.chat_input("Ask a question about council policies, forms, or documents…")
    if user_input:
        st.session_state.query_count += 1
        log_query(st.session_state.user_role, user_input, council, lang_sel)
        ai_reply = ask_ai(user_input, council)
        st.write(user_input)
        st.markdown(f"**Auto-response from {council} Council:**\n\n{ai_reply}")
        st.session_state.chat_history.append((user_input, ai_reply))

        # ==== EMAIL ANSWER FEATURE ====
        st.markdown("---")
        st.markdown("### 📧 Want this answer in your email?")
        receiver = st.text_input("Enter your email to receive this answer:", key="emailinput")
        source_link = f"https://www.{council.lower()}.vic.gov.au"  # Demo only
        if st.button("Send answer to my email"):
            if receiver and "@" in receiver and "." in receiver:
                try:
                    send_ai_email(receiver, user_input, ai_reply, source_link)
                    st.success("✅ AI answer sent to your email!")
                except Exception as e:
                    st.error(f"❌ Failed to send email: {e}")
            else:
                st.error("Please enter a valid email address.")

elif nav == "📥 Submit a Request":
    st.markdown("📌 Redirecting to your council’s website.")
    st.link_button("📝 Submit Online", f"https://www.{council.lower()}.vic.gov.au/request-it")

elif nav == "📊 Stats & Session":
    st.metric("Total Questions", st.session_state.query_count)
    st.metric("Session Start", st.session_state.session_start)
    st.metric("Role", st.session_state.user_role)
    st.metric("Council", council)
    st.metric("Plan", plan_label)
    # Usage analytics
    if os.path.exists("usage_log.csv"):
        df = pd.read_csv("usage_log.csv")
        st.write("All User Queries (last 20):")
        st.dataframe(df.tail(20))
        st.write("Most common questions:")
        st.write(df['question'].value_counts().head())
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        st.bar_chart(df.groupby('date').size())

elif nav == "🗃️ Export Chat":
    st.markdown("### Export your conversation as PDF")
    if st.button("Export Chat as PDF"):
        fname = "my_chat.pdf"
        export_chat_to_pdf(st.session_state.chat_history, fname)
        with open(fname, "rb") as f:
            st.download_button("Download your chat PDF", f, file_name=fname)

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
    st.link_button("Website", f"https://www.{council.lower()}.vic.gov.au")
    st.link_button("Contact Sales", "mailto:sales@civreply.com?subject=CivReply%20Sales%20Enquiry")

elif nav == "ℹ️ About Us":
    st.markdown(
        f"""
        ### About {council} City & CivReply AI

        **{council} City** is one of Australia’s fastest-growing and most diverse municipalities.
        The council is dedicated to delivering world-class community services, sustainability initiatives, smart city technology, and transparent governance for all residents, visitors, and businesses.

        **CivReply AI** is an innovative AI-powered assistant that empowers citizens and staff to instantly access council documents, policies, and services. It enables fast, accurate answers, streamlined workflows, and modern digital experiences for everyone.

        *“Smarter answers for smarter communities.”*
        """,
        unsafe_allow_html=True
    )

elif nav == "⚙️ Admin Panel":
    if st.session_state.user_role != "Staff" or not st.session_state.admin_verified:
        st.warning("Restricted to council staff only. Please select 'Staff' and enter the password above.")
    else:
        st.subheader("📂 Upload New Council Docs")
        docs = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)
        if docs and st.button("Rebuild Index"):
            folder = f"docs/{council.lower()}"
            os.makedirs(folder, exist_ok=True)
            for d in docs:
                with open(os.path.join(folder, d.name), "wb") as f:
                    f.write(d.read())
            loader = PyPDFDirectoryLoader(folder)
            raw_docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(raw_docs)
            vecdb = FAISS.from_documents(chunks, OpenAIEmbeddings())
            vecdb.save_local(f"index/{council.lower()}")
            st.success("✅ Index rebuilt successfully.")
        # -------- Advanced Form Scraping -------
        st.markdown("### 📝 Form Scraping (Auto-extract info from forms)")
        form_pdf = st.file_uploader("Upload a council form for scraping (PDF)", type="pdf", key="formpdf")
        if form_pdf:
            st.info("Extracting fields, please wait...")
            extracted = advanced_form_scrape(form_pdf)
            st.markdown("#### Extracted Fields and Values:")
            st.write(extracted)

# (Optional footer)
st.markdown(
    "<div style='text-align:center; color:#b2c6d6; font-size:0.96rem; margin:32px 0 8px 0;'>Made with 🏛️ CivReply AI – for Australian councils, powered by AI</div>",
    unsafe_allow_html=True
)
