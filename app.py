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

st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="centered")

# --- Session State Initialization ---
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "plan" not in st.session_state:
    st.session_state.plan = "basic"
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# --- Admin Auth ---
if not st.session_state.is_admin:
    password = st.text_input("Enter Admin Password to Enable Upload", type="password")
    if password == ADMIN_PASSWORD:
        st.session_state.is_admin = True
        st.success("✅ Admin access granted.")
    elif password:
        st.error("❌ Incorrect password")

# --- Council Selector ---
councils = [
    "Wyndham", "Brimbank", "Hobsons Bay", "Melbourne", "Yarra",
    "Moreland", "Darebin", "Boroondara", "Stonnington", "Port Phillip"
]
council = st.selectbox("Choose Council", councils)
council_key = council.lower().replace(" ", "_")
index_path = f"index/{council_key}"
data_path = f"data/{council_key}"
os.makedirs(data_path, exist_ok=True)

# --- Plan Limits ---
plan_limits = {
    "basic": {"queries": 500, "users": 1},
    "standard": {"queries": 2000, "users": 5},
    "enterprise": {"queries": float("inf"), "users": 20},
}
plan_name = st.session_state.plan.capitalize()
plan_queries = plan_limits[st.session_state.plan]["queries"]
plan_users = plan_limits[st.session_state.plan]["users"]

# --- Council Configs ---
council_landing_config = {
    "wyndham": {
        "tagline": "Empowering Wyndham residents with smarter answers.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/commons/1/1d/Wyndham_City_logo.png",
        "about": "Wyndham Council provides planning, permits, bins, and more. CivReply AI helps you navigate them effortlessly."
    },
    "melbourne": {
        "tagline": "Your personal assistant for City of Melbourne policies.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/City_of_Melbourne_logo.svg/2560px-City_of_Melbourne_logo.svg.png",
        "about": "Discover services, permits, and events across central Melbourne—instantly."
    },
    "brimbank": {
        "tagline": "Smart answers for Brimbank locals.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/6/65/Brimbank_City_Council_logo.png",
        "about": "Your AI-powered guide to waste collection, parking, development, and more in Brimbank."
    },
    "yarra": {
        "tagline": "Yarra Council services, now at your fingertips.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/f/f3/City_of_Yarra_logo.png",
        "about": "From community programs to local laws—explore all that Yarra offers with CivReply AI."
    },
    "hobsons_bay": {
        "tagline": "Navigate Hobsons Bay Council with ease.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/en/f/f1/Hobsons_Bay_City_Council_logo.png",
        "about": "Explore building permits, community services, and environment info powered by CivReply AI."
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
    }
}
config = council_landing_config.get(council_key, {})
tagline = config.get("tagline", f"Ask {council} Council anything – policies, laws, documents.")
hero_image = config.get("hero_image")
about_text = config.get("about", "")

# --- Responsive Branding Section ---
st.markdown(f"""
<style>
  body {{ font-family: 'Segoe UI', sans-serif; }}
  .header {{
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 20px;
  }}
  .hero-image {{
    width: 100%;
    max-width: 300px;
    margin-bottom: 10px;
  }}
  .title {{
    font-size: 2.8rem;
    margin: 0;
    font-weight: bold;
  }}
  .tagline {{
    text-align: center;
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 20px;
  }}
  .user-info-bar, .plan-box {{
    background-color: #eef6ff;
    padding: 10px 15px;
    border-radius: 12px;
    margin-bottom: 20px;
  }}
  .question-label {{
    font-size: 1rem;
    color: #374151;
    margin-bottom: 6px;
  }}
</style>
<div class="header">
  <img src="{hero_image}" alt="Council Logo" class="hero-image" />
  <div class="title">🏛️ CivReply AI</div>
</div>
<div class="tagline">{tagline}</div>
""", unsafe_allow_html=True)

# --- FAQs ---
faqs = {
    "wyndham": [
        ("How do I apply for a building permit?", "https://www.wyndham.vic.gov.au/services/building-planning/building/permits"),
        ("When is bin collection day?", "https://www.wyndham.vic.gov.au/services/waste-recycling/bin-collection"),
        ("Contact Wyndham Council", "https://www.wyndham.vic.gov.au/contact-us")
    ],
    "melbourne": [
        ("How do I report a noise complaint?", "https://www.melbourne.vic.gov.au/community/health-support-services/noise/pages/noise-complaints.aspx"),
        ("Where can I park in the city?", "https://www.melbourne.vic.gov.au/parking-and-transport/parking/pages/parking.aspx"),
        ("Contact Melbourne Council", "https://www.melbourne.vic.gov.au/about-council/contact-us/pages/contact-us.aspx")
    ],
    # Add more if needed
}
if council_key in faqs:
    st.markdown("### ❓ Frequently Asked Questions")
    for question, link in faqs[council_key]:
        st.markdown(f"- [{question}]({link})")

# --- Info Bars ---
st.markdown(f"""
<div class="user-info-bar">🧑 Council: {council} | 🔐 Role: {'Admin' if st.session_state.is_admin else 'Guest'}</div>
<div class="plan-box">📦 Plan: {plan_name} – {'Unlimited' if plan_queries == float('inf') else f'{plan_queries}'} queries/month | {plan_users} user(s) | <a href='{STRIPE_LINK}' target='_blank'>Upgrade →</a></div>
<div class="question-label">🔍 Ask a local question:</div>
""", unsafe_allow_html=True)

if about_text:
    st.info(about_text)
