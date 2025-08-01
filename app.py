import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import imaplib
import email
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STRIPE_LINK = os.getenv("STRIPE_LINK", "https://buy.stripe.com/test_xxx")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "supersecret")
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")

# Email log store
if "email_log" not in st.session_state:
    st.session_state.email_log = []

# Query count tracker
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Feedback store
if "feedback" not in st.session_state:
    st.session_state.feedback = []

st.set_page_config(page_title="CivReply AI", page_icon="\U0001F3DB️", layout="centered")

# --- Auto-email Response Function ---
def send_auto_email(recipient, question, answer):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = MIMEText(f"""
    <html>
    <body>
      <h3 style='color:#3b82f6;'>Your CivReply AI Answer</h3>
      <p><strong>Question:</strong> {question}</p>
      <p><strong>Answer:</strong><br>{answer}</p>
      <br>
      <p style='color:#6b7280;'>Sent at {now} from CivReply AI</p>
    </body>
    </html>
    """, "html")
    msg["Subject"] = f"Your CivReply AI Answer: {question[:50]}"
    msg["From"] = GMAIL_USER
    msg["To"] = recipient
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASS)
            server.send_message(msg)
        st.session_state.email_log.append({"to": recipient, "question": question, "time": now})
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# --- Title ---
st.markdown("""
<style>
  .main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 5px;
  }
</style>
<div class="main-title">\U0001F3DB️ CivReply AI</div>
""", unsafe_allow_html=True)

# --- Council Config ---
council_landing_config = {
    "wyndham": {
        "tagline": "Empowering Wyndham residents with smarter answers.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/commons/1/1d/Wyndham_City_logo.png",
        "about": "Wyndham Council provides planning, permits, bins, and more. CivReply AI helps you navigate them effortlessly."
    }
}

# --- Council Dropdown ---
councils = list(council_landing_config.keys())
council = st.selectbox("Choose Council", [c.title().replace("_", " ") for c in councils])
council_key = council.lower().replace(" ", "_")
config = council_landing_config.get(council_key, {})
hero_image = config.get("hero_image")
tagline = config.get("tagline")
about_text = config.get("about", "")

# --- Plan Selector ---
plan_choice = st.selectbox("Choose Plan", ["Basic", "Standard", "Enterprise"])
st.session_state.plan = plan_choice.lower()

# --- Plan Setup ---
plan_limits = {
    "basic": {"queries": 500, "users": 1, "email": True},
    "standard": {"queries": 2000, "users": 5, "email": True},
    "enterprise": {"queries": float("inf"), "users": 20, "email": True},
}
plan = st.session_state.plan
plan_name = plan.capitalize()
plan_queries = plan_limits[plan]["queries"]
plan_users = plan_limits[plan]["users"]

# --- Info Bars ---
st.markdown(f"""
<div class="user-info-bar">🧑 Council: {council} | 🔐 Role: {'Admin' if st.session_state.get('is_admin') else 'Guest'}</div>
<div class="plan-box">💼 Plan: {plan_name} – {'Unlimited' if plan_queries == float('inf') else f'{plan_queries}'} instant answers/month | {plan_users} seat(s) | <a href='{STRIPE_LINK}' target='_blank'>Upgrade →</a></div>
""", unsafe_allow_html=True)

if about_text:
    st.info(about_text)

# --- Local Question Input ---
st.markdown("### 🔍 Ask a local question:")
user_question = st.text_input("Type your question here", placeholder="e.g., What day is bin collection in Wyndham?")
user_email = st.text_input("Your email (optional)", placeholder="your@email.com")

# --- Process Answer & Email ---
if user_question:
    if st.session_state.query_count < plan_queries:
        st.session_state.query_count += 1
        st.markdown("✅ Processing your question...")
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        retriever = FAISS.load_local("index/wyndham/index.faiss", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)).as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        answer = qa_chain.run(user_question)
        st.markdown(f"**Answer:** {answer}")

        if user_email and plan_limits[plan]["email"]:
            if send_auto_email(user_email, user_question, answer):
                st.success("✅ Answer sent to your email.")

        # --- Feedback ---
        st.markdown("### 🙋 Feedback")
        rating = st.slider("Rate your answer:", 1, 5, 3, key=f"rating_{st.session_state.query_count}")
        emoji_display = "⭐" * rating
        st.markdown(f"You rated: {emoji_display}")
        comment = st.text_input("Any suggestions or notes?", key=f"comment_{st.session_state.query_count}")
        if st.button("Submit Feedback"):
            st.session_state.feedback.append({
                "question": user_question,
                "answer": answer,
                "rating": rating,
                "comment": comment,
                "time": datetime.now().isoformat()
            })
            st.success("🙏 Thanks for your feedback!")
    else:
        st.error("❌ You've reached the maximum query limit for your plan.")

# --- FAQs ---
faqs = {
    "wyndham": [
        ("How do I apply for a building permit?", "https://www.wyndham.vic.gov.au/services/building-planning/building/permits"),
        ("When is bin collection day?", "https://www.wyndham.vic.gov.au/services/waste-recycling/bin-collection"),
        ("Contact Wyndham Council", "https://www.wyndham.vic.gov.au/contact-us")
    ]
}
if council_key in faqs:
    st.markdown("### ❓ Frequently Asked Questions")
    for question, link in faqs[council_key]:
        st.markdown(f"- [{question}]({link})")

# --- Optional: Admin view email log ---
if st.session_state.get("is_admin") and st.session_state.email_log:
    st.markdown("### 📬 Email Log")
    for log in st.session_state.email_log:
        st.markdown(f"- [{log['time']}] Sent to **{log['to']}** | Question: _{log['question']}_")

# --- Optional: Admin view feedback log ---
if st.session_state.get("is_admin") and st.session_state.feedback:
    st.markdown("### 📣 Feedback Log")
    for item in st.session_state.feedback:
        stars = "⭐" * item['rating']
        st.markdown(f"- [{item['time']}] | Rating: {stars} | Q: _{item['question']}_ | Comment: {item['comment']}")
