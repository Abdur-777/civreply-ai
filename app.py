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

LOCAL_LOGO = os.path.join(os.path.dirname(__file__), "b7b9830f-9785-40ad-acd0-4a3bb9ccedde.png")
REMOTE_LOGO = "https://www.wyndham.vic.gov.au/sites/default/files/styles/small/public/2020-06/logo_0.png"

st.set_page_config(page_title="CivReply AI", page_icon=LOCAL_LOGO, layout="wide")

# ---- HEADER: Big logo + text ----
st.markdown(
    f"""
    <div style='background:linear-gradient(90deg,{WYNDHAM_BLUE},#7ecaf6 100%);padding:44px 0 24px 0;border-radius:0 0 44px 44px;box-shadow:0 10px 40px #cce5f7;display:flex;align-items:center;justify-content:center;gap:28px;'>
      <img src="https://www.wyndham.vic.gov.au/sites/default/files/styles/small/public/2020-06/logo_0.png" style="width:104px;border-radius:15px;box-shadow:0 0 18px #90c9e9;">
      <span style='font-size:3.2rem;font-weight:900;color:#fff;letter-spacing:3px;text-shadow:0 2px 18px #36A9E160;'>CivReply AI</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- STATUS BAR ----
st.markdown(
    f"""
    <div style='background:{WYNDHAM_LIGHT};border-radius:16px;padding:17px 48px;display:flex;justify-content:center;align-items:center;gap:60px;margin-top:20px;margin-bottom:14px;box-shadow:0 2px 10px #b4dbf2;'>
        <div style='color:{WYNDHAM_DEEP};font-size:1.13rem;font-weight:700'>🏛️ Active Council:</div>
        <div style='font-weight:700;'>{st.session_state.get('council', 'Wyndham')}</div>
        <div style='color:{WYNDHAM_DEEP};font-size:1.13rem;font-weight:700'>📦 Plan:</div>
        <div style='font-weight:700;'>{st.session_state.get('plan', 'basic').title()}</div>
        <div style='color:{WYNDHAM_DEEP};font-size:1.13rem;font-weight:700'>🌐 Language:</div>
        <div style='font-weight:700;'>English</div>
    </div>
    """, unsafe_allow_html=True
)

# ---- HERO WELCOME BLOCK ----
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

# ---- TRY ASKING EXAMPLES ----
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

# ---- HOW IT WORKS (EXPANDER) ----
with st.expander("How does CivReply AI work?", expanded=False):
    st.markdown("""
      1. **Type your question** about council policies, forms, or services.
      2. **Our AI instantly searches** official council documents for the right answer.
      3. **Get clear, human-friendly replies** in seconds!
    """)

# --- Continue with your Role Selector, Router, etc... ---

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown(
        f"""
        <div style='background:{WYNDHAM_BLUE};padding:20px 0 14px 0;border-radius:0 0 32px 32px;box-shadow:0 4px 18px #cce5f7;margin-bottom:10px;'>
          <div style='display:flex;align-items:center;justify-content:center;gap:13px;'>
            <img src="https://www.wyndham.vic.gov.au/sites/default/files/styles/small/public/2020-06/logo_0.png" width="50" style="border-radius:9px;box-shadow:0 0 12px #90c9e9;">
            <span style='font-size:1.4rem;font-weight:800;color:#fff;letter-spacing:0.5px;'>CivReply AI</span>
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
