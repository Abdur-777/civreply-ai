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
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STRIPE_LINK = os.getenv("STRIPE_LINK", "https://buy.stripe.com/test_xxx")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "supersecret")
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")

st.set_page_config(page_title="CivReply AI", page_icon="\U0001F3DB️", layout="centered")

# --- Load/Track Query Count ---
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

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
    "basic": {"queries": 500, "users": 1},
    "standard": {"queries": 2000, "users": 5},
    "enterprise": {"queries": float("inf"), "users": 20},
}
plan = st.session_state.plan
plan_name = plan.capitalize()
plan_queries = plan_limits[plan]["queries"]
plan_users = plan_limits[plan]["users"]

# --- Enforce Query Limit ---
if st.session_state.query_count >= plan_queries:
    st.warning("You’ve reached your plan’s query limit. Upgrade to continue.")
    st.stop()

# --- Info Bars ---
st.markdown(f"""
<div class="user-info-bar">🧑 Council: {council} | 🔐 Role: {'Admin' if st.session_state.get('is_admin') else 'Guest'}</div>
<div class="plan-box">💼 Plan: {plan_name} – {'Unlimited' if plan_queries == float('inf') else f'{plan_queries}'} instant answers/month | {plan_users} seat(s) | <a href='{STRIPE_LINK}' target='_blank'>Upgrade →</a></div>
""", unsafe_allow_html=True)

# --- Plan Features Display ---
if plan == "basic":
    st.markdown("""
    <div style="background-color: #f9fafb; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 8px; margin-top: 10px;">
      <strong>With the Basic Plan you get:</strong>
      <ul>
        <li>✅ 500 AI-powered queries per month</li>
        <li>✅ PDF policy/document lookup (no need to search manually)</li>
        <li>✅ 24/7 availability for council-related questions</li>
        <li>✅ 1 user seat – perfect for solo operators, reception desks, or admin officers</li>
      </ul>
      <em>That’s just $1 per question – and 10x faster than calling or searching council websites.</em>
    </div>
    """, unsafe_allow_html=True)

elif plan == "standard":
    st.markdown("""
    <div style="background-color: #f9fafb; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 8px; margin-top: 10px;">
      <strong>With the Standard Plan you get:</strong>
      <ul>
        <li>✅ 2,000 AI-powered queries per month</li>
        <li>✅ Upload custom council documents and PDFs</li>
        <li>✅ 5 user seats – ideal for departments and offices</li>
      </ul>
      <em>Standard plan saves time across teams – and gives faster access to policy answers than staff alone.</em>
    </div>
    """, unsafe_allow_html=True)

elif plan == "enterprise":
    st.markdown("""
    <div style="background-color: #f9fafb; border-left: 5px solid #10b981; padding: 15px; border-radius: 8px; margin-top: 10px;">
      <strong>With the Enterprise Plan you get:</strong>
      <ul>
        <li>✅ Unlimited AI-powered queries per month</li>
        <li>✅ Upload and manage multiple council databases</li>
        <li>✅ 20+ user seats for large teams or entire departments</li>
        <li>✅ Priority support and custom integrations</li>
      </ul>
      <em>Enterprise plan powers entire councils with secure, AI-enhanced service delivery.</em>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="color: #1f2937; font-size: 0.95rem; margin-top: 10px;">
  CivReply AI is your always-on council knowledge assistant – now with team collaboration support.
</div>
""", unsafe_allow_html=True)

if about_text:
    st.info(about_text)

# --- Local Question Input ---
st.markdown("### 🔍 Ask a local question:")
user_question = st.text_input("Type your question here", placeholder="e.g., What day is bin collection in Wyndham?")
user_email = st.text_input("Your email (optional)", placeholder="your@email.com")

if user_question:
    st.session_state.query_count += 1
    st.success(f"Query submitted! You’ve used {st.session_state.query_count} of {plan_queries} allowed queries.")

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
