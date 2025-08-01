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
    "melbourne": {
        "tagline": "Your personal assistant for City of Melbourne policies.",
        "hero_image": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/City_of_Melbourne_logo.svg/2560px-City_of_Melbourne_logo.svg.png",
        "about": "Discover services, permits, and events across central Melbourne—instantly."
    },
    # Add other councils...
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
  .hero-image {{
    width: 100%;
    max-width: 300px;
    margin-bottom: 10px;
  }}
  .tagline {{
    text-align: center;
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 20px;
  }}
</style>
<div class="header">
  <img src="{hero_image}" alt="Council Logo" class="hero-image" />
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
<div class="plan-box">📦 Plan: {plan_name} – {'Unlimited' if plan_queries == float('inf') else f'{plan_queries}'} queries/month | {plan_users} user(s) | <a href='{STRIPE_LINK}' target='_blank'>Upgrade →</a></div>
""", unsafe_allow_html=True)

if about_text:
    st.info(about_text)

# --- Local Question Input ---
st.markdown("### 🔍 Ask a local question:")
user_question = st.text_input("Type your question here", placeholder="e.g., What day is bin collection in Wyndham?")

# --- FAQs at Bottom ---
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
    ]
}

if council_key in faqs:
    st.markdown("### ❓ Frequently Asked Questions")
    for question, link in faqs[council_key]:
        st.markdown(f"- [{question}]({link})")
