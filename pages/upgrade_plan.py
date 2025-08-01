import streamlit as st

st.set_page_config(page_title="Upgrade Plan", page_icon="💼", layout="wide")

st.title("💼 Upgrade Plan")
st.caption("Choose a plan that fits your council’s needs.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Basic ($499/mo)")
    st.write("- PDF Q&A")
    st.write("**Limit:** 500 queries")

with col2:
    st.subheader("Standard ($1499/mo)")
    st.write("- PDF Q&A")
    st.write("- Form Scraping")
    st.write("**Limit:** 2000 queries")

with col3:
    st.subheader("Enterprise ($2999+/mo)")
    st.write("- All Features")
    st.write("**Limit:** Unlimited queries")

st.divider()
st.page_link("app.py", label="⬅️ Back to Home")
