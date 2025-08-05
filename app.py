import streamlit as st
import os, json, requests
from datetime import datetime
import pandas as pd
from deep_translator import GoogleTranslator
from fpdf import FPDF

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import pdfplumber

from supabase import create_client
import stripe

# ============ CONFIG & INIT ============
with open("council_config.json") as f:
    COUNCIL_CONFIG = json.load(f)
COUNCILS = [c.title().replace("_", " ") for c in COUNCIL_CONFIG.keys()]
COUNCIL_KEYS = list(COUNCIL_CONFIG.keys())
LANGUAGES = {
    "English": "en", "Chinese": "zh-CN", "Arabic": "ar", "Spanish": "es",
    "Hindi": "hi", "Vietnamese": "vi", "Filipino": "tl", "Turkish": "tr", "French": "fr"
}
PLAN_CONFIG = {
    "basic":    {"label": "Basic ($499/mo)", "icon": "💧", "limit": 500, "features": ["📄 PDF Q&A", "🔒 500 queries", "📧 Email support", "📚 Policy finder", "📱 Mobile access", "☁️ Cloud storage", "👥 Knowledge base"]},
    "standard": {"label": "Standard ($1,499/mo)", "icon": "🚀", "limit": 2000, "features": ["✅ Everything in Basic", "🔒 2,000 queries", "📝 Form Scraping", "⚡ Immediate support", "📊 Analytics", "🗃️ PDF export", "🌏 Multi-language", "📦 Bulk uploads", "🎨 Branding"]},
    "enterprise": {"label": "Enterprise ($2,999+/mo)", "icon": "🏆", "limit": float("inf"), "features": ["✅ Everything in Standard", "🔓 Unlimited", "👤 Account manager", "🔌 API access", "🛡️ 99.9% SLA", "🔐 SSO", "🧑‍🏫 Training", "🤖 3rd party tools", "☁️ On-prem/cloud", "🛠️ Custom automations"]}
}

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
stripe.api_key = st.secrets["STRIPE_SECRET"]

# ========== STATE INIT ==========
st.set_page_config(page_title="CivReply AI", page_icon="🏛️", layout="wide")
for key, val in {
    "logged_in": False, "user_email": None, "user_role": "Resident", "council": COUNCILS[0], "plan": "basic",
    "chat_history": [], "stripe_customer_id": None, "session_id": None
}.items():
    if key not in st.session_state: st.session_state[key] = val

# ========== AUTH ==========
def sso_login():
    st.write("SSO: Not implemented in demo. Use Google/Auth0/Okta for prod.")
def user_login_ui():
    st.markdown("## 👤 Sign In to CivReply AI")
    user_email = st.text_input("Email")
    user_role = st.selectbox("Role", ["Resident", "Staff", "Admin"])
    pwd = st.text_input("Password", type="password") if user_role != "Resident" else ""
    if st.button("Login"):
        if user_role == "Staff" and pwd == st.secrets["STAFF_PASSWORD"]:
            st.session_state.update({"logged_in": True, "user_email": user_email, "user_role": "Staff"})
        elif user_role == "Admin" and pwd == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.update({"logged_in": True, "user_email": user_email, "user_role": "Admin"})
        elif user_role == "Resident":
            st.session_state.update({"logged_in": True, "user_email": user_email or "anon@civreply.ai", "user_role": "Resident"})
        else:
            st.error("Invalid login.")
        st.experimental_rerun()
if not st.session_state["logged_in"]:
    user_login_ui()
    st.stop()

# ========== SIDEBAR ==========
council_selected = st.sidebar.selectbox("Select council", COUNCILS, index=COUNCILS.index(st.session_state["council"]))
council_key = council_selected.lower().replace(" ", "_")
council_info = COUNCIL_CONFIG.get(council_key, {})
st.session_state["council"] = council_selected
default_plan = council_info.get("plan", "basic")
st.session_state["plan"] = st.session_state.get("plan", default_plan)

if img := council_info.get("hero_image", ""): st.sidebar.image(img, width=160)
if tl := council_info.get("tagline", ""): st.sidebar.markdown(f"**{tl}**")
if ab := council_info.get("about", ""): st.sidebar.caption(ab)
selected_lang_label = st.sidebar.selectbox("Language", list(LANGUAGES.keys()), index=0)
selected_lang = LANGUAGES[selected_lang_label]
plan_selected = st.sidebar.selectbox("Plan", list(PLAN_CONFIG.keys()), format_func=lambda x: PLAN_CONFIG[x]["label"])
st.session_state["plan"] = plan_selected

def get_stripe_customer_id(email):
    user = supabase.table("users").select("stripe_customer_id").eq("email", email).single().execute()
    return user.data["stripe_customer_id"] if user.data else None

def get_query_count(user_email, plan):
    resp = supabase.table("queries").select("id").eq("user_email", user_email).eq("plan", plan).execute()
    return len(resp.data) if resp.data else 0

def log_query(user_email, council, role, question, plan, lang):
    supabase.table("queries").insert({
        "timestamp": datetime.now().isoformat(),
        "user_email": user_email, "council": council, "role": role, "question": question, "plan": plan, "lang": lang
    }).execute()

def log_feedback(user_email, council, text): supabase.table("feedback").insert({
    "timestamp": datetime.now().isoformat(), "user_email": user_email, "council": council, "feedback": text
}).execute()

# ========== PLAN ENFORCEMENT ==========
if st.session_state["plan"] != "enterprise":
    count = get_query_count(st.session_state["user_email"], st.session_state["plan"])
    if count >= PLAN_CONFIG[st.session_state["plan"]]["limit"]:
        st.warning(f"Query limit reached for {st.session_state['plan'].capitalize()} plan. Please upgrade.")
        if st.button("Upgrade via Stripe"):
            # Stripe checkout session
            price_lookup = {"basic": "price_1...", "standard": "price_2...", "enterprise": "price_3..."}  # Set up your prices in Stripe!
            customer_id = get_stripe_customer_id(st.session_state["user_email"])
            if not customer_id:
                customer = stripe.Customer.create(email=st.session_state["user_email"])
                customer_id = customer.id
                supabase.table("users").insert({"email": st.session_state["user_email"], "stripe_customer_id": customer_id}).execute()
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=["card"],
                line_items=[{"price": price_lookup[plan_selected], "quantity": 1}],
                mode="subscription",
                success_url=st.request.url, cancel_url=st.request.url
            )
            st.markdown(f"[Continue to Stripe Checkout]({session.url})", unsafe_allow_html=True)
            st.stop()
        st.stop()

# ========== NAVIGATION ==========
st.sidebar.markdown("---")
nav = st.sidebar.radio("Navigation", [
    "💬 Chat with Council AI", "📥 Submit a Request", "📊 Stats & Analytics", "🗃️ Export Chats as PDF",
    "📦 Bulk Data Upload", "🎨 Council Branding", "📝 Form Scraper", "💡 Share Feedback", "📞 Contact Us",
    "ℹ️ About Us", "⚙️ Admin Panel", "🌐 Community Knowledge Base"
])

# ========== MAIN ROUTES ==========
def ask_ai(question, council, lang="en"):
    embeddings = OpenAIEmbeddings()
    index_path = f"index/{council.lower()}"
    if not os.path.exists(index_path): return "[Error] No index found for this council"
    db = FAISS.load_local(index_path, embeddings)
    retriever = db.as_retriever()
    model = "gpt-3.5-turbo" if st.session_state["plan"] == "basic" else "gpt-4o"
    llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model=model)
    prompt = "You are a helpful council assistant. Only answer from the provided documents."
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})
    answer = qa.run(question)
    return answer

if nav == "💬 Chat with Council AI":
    st.subheader(f"💬 Ask {st.session_state['council']} Council")
    user_input = st.chat_input("Ask a question about council policies, forms, or documents…")
    if user_input:
        ai_reply = ask_ai(user_input, st.session_state["council"], lang=selected_lang)
        ai_reply_translated = GoogleTranslator(source="auto", target=selected_lang).translate(ai_reply) if selected_lang != "en" else ai_reply
        st.markdown(f"**Auto-response from {st.session_state['council']} Council:**\n\n{ai_reply_translated}")
        st.session_state["chat_history"].append((user_input, ai_reply_translated))
        log_query(st.session_state["user_email"], st.session_state["council"], st.session_state["user_role"], user_input, st.session_state["plan"], selected_lang)

        if "http" in ai_reply:
            import re; links = re.findall(r'(https?://\S+)', ai_reply)
            for l in links: st.markdown(f"🔗 [View referenced document]({l})")
        st.markdown("---")
        st.markdown("### 📧 Want this answer in your email?")
        receiver = st.text_input("Enter your email to receive this answer:", key="emailinput")
        source_link = links[0] if "links" in locals() and links else "https://www.wyndham.vic.gov.au"
        if st.button("Send answer to my email"):
            from yagmail import SMTP
            yag = SMTP(st.secrets["GMAIL_USER"], st.secrets["GMAIL_APP_PASSWORD"])
            yag.send(to=receiver, subject=f"CivReply AI – Answer: {user_input}", contents=ai_reply_translated + f"\n\nSource: {source_link}")
            st.success("✅ AI answer sent to your email!")

elif nav == "📥 Submit a Request":
    council_links = {"Wyndham": "https://www.wyndham.vic.gov.au/request-it", "Melbourne": "https://www.melbourne.vic.gov.au/contact-us/Pages/contact-us.aspx"}
    link = council_links.get(st.session_state["council"], "https://www.wyndham.vic.gov.au/request-it")
    st.link_button("📝 Submit Online", link)

elif nav == "📊 Stats & Analytics":
    st.header("📊 Usage Analytics Dashboard")
    queries = supabase.table("queries").select("*").eq("council", st.session_state["council"]).order("timestamp", desc=True).limit(200).execute()
    df = pd.DataFrame(queries.data) if queries.data else pd.DataFrame([])
    st.dataframe(df.tail(100), use_container_width=True)
    st.metric("Total Questions (all time)", len(df))
    st.metric("Current Plan", PLAN_CONFIG[st.session_state["plan"]]["label"])

elif nav == "🗃️ Export Chats as PDF":
    st.header("🗃️ Export Your Chat History")
    if st.button("Export as PDF"):
        filename = f"civreply_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "CivReply AI Chat Export", ln=True, align="C"); pdf.ln(10)
        for idx, (q, a) in enumerate(st.session_state["chat_history"], 1): pdf.multi_cell(0, 10, f"Q{idx}: {q}\nA{idx}: {a}\n\n")
        pdf.output(filename)
        with open(filename, "rb") as f: st.download_button("Download PDF", data=f, file_name=filename)

elif nav == "📦 Bulk Data Upload":
    st.header("📦 Bulk PDF/Data Upload (Staff/Admin Only)")
    if st.session_state["user_role"] not in ["Staff", "Admin"]: st.warning("Staff/Admin access required.")
    else:
        docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Bulk Index & Add"):
            if docs:
                folder = f"docs/{st.session_state['council'].lower()}"
                os.makedirs(folder, exist_ok=True)
                for d in docs:
                    with open(os.path.join(folder, d.name), "wb") as f: f.write(d.read())
                loader = PyPDFDirectoryLoader(folder)
                raw_docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(raw_docs)
                vecdb = FAISS.from_documents(chunks, OpenAIEmbeddings())
                vecdb.save_local(f"index/{st.session_state['council'].lower()}")
                st.success("✅ Bulk indexing complete.")

elif nav == "🎨 Council Branding":
    st.header("🎨 Custom Council Branding (Admins Only)")
    if st.session_state["user_role"] != "Admin": st.warning("Admin access required.")
    else:
        uploaded_logo = st.file_uploader("Upload council logo", type=["png", "jpg", "jpeg"])
        if uploaded_logo: st.image(uploaded_logo, width=180); st.success("Logo uploaded! (Demo only)")

elif nav == "📝 Form Scraper":
    st.header("📝 Advanced Form Scraping (PDF)")
    pdf_file = st.file_uploader("Upload a PDF form to auto-extract fields", type="pdf")
    if pdf_file and st.button("Extract Form Data"):
        with open("temp_form.pdf", "wb") as f: f.write(pdf_file.read())
        with pdfplumber.open("temp_form.pdf") as pdf:
            results = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    import re; fields = re.findall(r"(Name|Email|Phone|Address|Date of Birth|Signature):?\s*(.+)", text)
                    results.extend(fields)
            results = results if results else [("No form fields found", "")]
        st.write(pd.DataFrame(results, columns=["Field", "Value"]))

elif nav == "💡 Share Feedback":
    st.header("💡 Share Feedback")
    fb = st.text_area("Tell us what’s working or not...")
    if st.button("📨 Submit Feedback"):
        log_feedback(st.session_state["user_email"], st.session_state["council"], fb)
        st.success("Thanks for helping improve CivReply AI!")

elif nav == "🌐 Community Knowledge Base":
    st.header("🌐 Community Knowledge Base")
    tips = supabase.table("community_tips").select("*").order("timestamp", desc=True).limit(50).execute()
    tips_df = pd.DataFrame(tips.data) if tips.data else pd.DataFrame([])
    st.dataframe(tips_df[["timestamp", "council", "tip"]].tail(50), use_container_width=True)
    tip = st.text_area("Suggest a new tip or local info to share")
    if st.button("Add Tip"):
        supabase.table("community_tips").insert({
            "timestamp": datetime.now().isoformat(), "user_email": st.session_state["user_email"],
            "council": st.session_state["council"], "tip": tip
        }).execute()
        st.success("Tip added! Pending admin approval.")

# ...About, Admin, Contact, etc. as in earlier template...

st.markdown(
    "<div style='text-align:center; color:#b2c6d6; font-size:0.96rem; margin:32px 0 8px 0;'>Made with 🏛️ CivReply AI – for Australian councils, powered by AI</div>",
    unsafe_allow_html=True
)
