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

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STRIPE_LINK = os.getenv("STRIPE_LINK", "https://buy.stripe.com/test_xxx")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "supersecret")
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")

st.set_page_config(page_title="CivReply AI", page_icon="\U0001F3DB️", layout="centered")

# --- Title at very top ---
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

# --- Default council config for early logo render ---
council_landing_config = {
    "wyndham": {
        "tagline": "Empowering Wyndham residents with smarter answers.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/commons/1/1d/Wyndham_City_logo.png",
        "about": "Wyndham Council provides planning, permits, bins, and more. CivReply AI helps you navigate them effortlessly."
    },
    "brimbank": {
        "tagline": "Smart answers for Brimbank locals.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/6/65/Brimbank_City_Council_logo.png",
        "about": "Your AI-powered guide to waste collection, parking, development, and more in Brimbank."
    },
    "hobsons_bay": {
        "tagline": "Navigate Hobsons Bay Council with ease.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/f/f1/Hobsons_Bay_City_Council_logo.png",
        "about": "Explore building permits, community services, and environment info powered by CivReply AI."
    },
    "yarra": {
        "tagline": "Yarra Council services, now at your fingertips.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/f/f3/City_of_Yarra_logo.png",
        "about": "From community programs to local laws—explore all that Yarra offers with CivReply AI."
    },
    "moreland": {
        "tagline": "Simplifying Moreland’s services with AI.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/2/2f/Moreland_City_Council_logo.png",
        "about": "Your go-to assistant for understanding Moreland’s policies, recycling, and community grants."
    },
    "darebin": {
        "tagline": "Answers for Darebin residents, instantly.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/2/29/Darebin_City_Council_logo.png",
        "about": "AI-powered access to Darebin Council services, forms, and waste schedules."
    },
    "boroondara": {
        "tagline": "Explore Boroondara with clarity and confidence.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/5/55/City_of_Boroondara_Logo.png",
        "about": "Everything from permits to planning in Boroondara—smartly answered."
    },
    "stonnington": {
        "tagline": "Stonnington Council guidance made easy.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/e/eb/City_of_Stonnington_logo.png",
        "about": "Your streamlined AI interface to council services, local laws, and event info."
    },
    "port_phillip": {
        "tagline": "Port Phillip AI assistant for all services.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/b/bc/City_of_Port_Phillip_logo.png",
        "about": "Find what you need about permits, parking, and arts programs across Port Phillip."
    },
    "hume": {
        "tagline": "Helping Hume residents with everyday council queries.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/5/5e/Hume_City_Council_logo.png",
        "about": "Access services, permits, and updates for Hume City with CivReply AI."
    }
}

# Set default council logo and tagline
default_council = "Wyndham"
default_key = default_council.lower().replace(" ", "_")
default_config = council_landing_config.get(default_key, {})
hero_image = default_config.get("hero_image")
tagline = default_config.get("tagline")

# --- Council Logo & Tagline ---
st.markdown(f"""
<style>
  .header {{
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 20px;
  }}
  .tagline {{
    text-align: center;
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 20px;
  }}
</style>
<div class="header">
  <div class="tagline">{tagline}</div>
</div>
""", unsafe_allow_html=True)

# --- Admin Access ---
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if not st.session_state.is_admin:
    password = st.text_input("Enter Admin Password to Enable Upload", type="password")
    if password == ADMIN_PASSWORD:
        st.session_state.is_admin = True
        st.success("✅ Admin access granted.")
    elif password:
        st.error("❌ Incorrect password")

# --- File Uploader (Admin Only) ---
if st.session_state.is_admin:
    st.markdown("### 📤 Upload Council PDFs")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s). (Processing logic goes here.)")

# --- Council Dropdown ---
councils = list(council_landing_config.keys())
council = st.selectbox("Choose Council", [c.title().replace("_", " ") for c in councils])
council_key = council.lower().replace(" ", "_")
config = council_landing_config.get(council_key, {})
hero_image = config.get("hero_image")
tagline = config.get("tagline")
about_text = config.get("about", "")

# --- Plan Setup ---
plan_limits = {
    "basic": {"queries": 500, "users": 1},
    "standard": {"queries": 2000, "users": 5},
    "enterprise": {"queries": float("inf"), "users": 20},
}
if "plan" not in st.session_state:
    st.session_state.plan = "basic"
plan = st.session_state.plan
plan_name = plan.capitalize()
plan_queries = plan_limits[plan]["queries"]
plan_users = plan_limits[plan]["users"]

# --- Info Bars ---
st.markdown(f"""
<div class="user-info-bar">🧑 Council: {council} | 🔐 Role: {'Admin' if st.session_state.is_admin else 'Guest'}</div>
<div class="plan-box">💼 Plan: Smart Solo – {'Unlimited' if plan_queries == float('inf') else f'{plan_queries}'} instant answers/month | {plan_users} seat(s) | <a href='{STRIPE_LINK}' target='_blank' style='color:#3b82f6; text-decoration: underline;'>Upgrade →</a></div>
""", unsafe_allow_html=True)

# --- Pricing Benefits Highlight ---
st.markdown("""
<div style="background-color: #f9fafb; border-left: 5px solid #3b82f6; padding: 16px 20px; border-radius: 8px; margin-top: 20px;">
  <h4 style="margin-top: 0;">💡 What's included in the Basic Plan ($499/month):</h4>
  <ul style="margin: 0 0 10px 0; padding-left: 20px;">
    <li>✅ <strong>500 AI-powered queries</strong> per month</li>
    <li>✅ <strong>PDF support</strong> – upload council documents for instant lookup</li>
    <li>✅ <strong>24/7 smart assistant</strong> – no need to wait on hold</li>
    <li>✅ <strong>1 user seat</strong> – ideal for reception, solo staff, or public counters</li>
  </ul>
  <em>Just $1 per query — and saves hours otherwise spent digging through council sites.</em>
</div>
""", unsafe_allow_html=True)

# --- Testimonial Card ---
st.markdown("""
<div style="background-color: #f3f4f6; border-left: 5px solid #10b981; padding: 15px; border-radius: 8px; font-style: italic; color: #374151; margin-top: 20px;">
  “Our front desk saved 4 hours every week using CivReply AI. It’s like having a full-time assistant trained in council rules.”
  <div style="margin-top: 10px; font-weight: bold; color: #111827;">— Local Government Staff Member</div>
</div>
""", unsafe_allow_html=True)

# --- Final persuasive line ---
st.markdown("""
<div style="color: #1f2937; font-size: 0.95rem; margin-top: 16px;">
  🕒 For less than the cost of one staff hour, CivReply answers 500+ questions — instantly.
</div>
""", unsafe_allow_html=True)
