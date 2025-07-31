# CivReply AI – Upgraded Version with Admin Login, Multi-Council Support, Stripe, and GPT-Powered Gmail Auto-Reply

import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain\_openai import ChatOpenAI, OpenAIEmbeddings
from langchain\_community.vectorstores import FAISS
from dotenv import load\_dotenv
from langchain\_community.document\_loaders import PyPDFLoader
from langchain.text\_splitter import RecursiveCharacterTextSplitter
import imaplib
import email
import smtplib
from email.mime.text import MIMEText

# Load environment variables

load\_dotenv()
OPENAI\_API\_KEY = os.getenv("OPENAI\_API\_KEY")
STRIPE\_LINK = os.getenv("STRIPE\_LINK", "[https://buy.stripe.com/test\_xxx](https://buy.stripe.com/test_xxx)")
ADMIN\_PASSWORD = os.getenv("ADMIN\_PASSWORD", "supersecret")
GMAIL\_USER = os.getenv("GMAIL\_USER")
GMAIL\_PASS = os.getenv("GMAIL\_PASS")

st.set\_page\_config(page\_title="CivReply AI", page\_icon="🏛️", layout="centered")

# --- Admin Auth ---

if "is\_admin" not in st.session\_state:
st.session\_state.is\_admin = False

if not st.session\_state.is\_admin:
password = st.text\_input("Enter Admin Password to Enable Upload", type="password")
if password == ADMIN\_PASSWORD:
st.session\_state.is\_admin = True
st.success("✅ Admin access granted.")
elif password:
st.error("❌ Incorrect password")

# --- Council Selector ---

councils = \[
"Wyndham", "Brimbank", "Hobsons Bay", "Melbourne", "Yarra",
"Moreland", "Darebin", "Boroondara", "Stonnington", "Port Phillip"
]
council = st.selectbox("Choose Council", councils)
council\_key = council.lower().replace(" ", "\_")
index\_path = f"index/{council\_key}"

# --- Branding ---

st.markdown(f"""

<style>
  body {{ font-family: 'Segoe UI', sans-serif; }}
  .header {{ display: flex; justify-content: center; gap: 12px; margin-bottom: 10px; }}
  .header h1 {{ font-size: 2.5rem; margin: 0; }}
  .tagline {{ text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 20px; }}
  .user-info-bar, .plan-box {{ background-color: #eef6ff; padding: 10px 15px; border-radius: 12px; margin-bottom: 20px; }}
  .question-label {{ font-size: 1rem; color: #374151; margin-bottom: 6px; }}
  .footer {{ text-align: center; font-size: 0.85rem; color: #6b7280; margin-top: 30px; }}
</style>

<div class="header">
  <div style="font-size: 2rem;">🏛️</div>
  <h1>CivReply AI</h1>
</div>
<div class="tagline">Ask {council} Council anything – policies, laws, documents.</div>
<div class="user-info-bar">🧑 Council: {council} | 🔐 Role: {'Admin' if st.session_state.is_admin else 'Guest'}</div>
<div class="plan-box">📦 Plan: Basic – 500 queries/month | 1 user | <a href='{STRIPE_LINK}' target='_blank'>Upgrade →</a></div>
<div class="question-label">🔍 Ask a local question:</div>
""", unsafe_allow_html=True)

# --- Input UI ---

question = st.text\_input("e.g. Do I need a permit to cut down a tree?", key="question\_box", label\_visibility="collapsed")

if "query\_count" not in st.session\_state:
st.session\_state.query\_count = 0

# --- Load Vector Index ---

try:
db = FAISS.load\_local(index\_path, OpenAIEmbeddings(openai\_api\_key=OPENAI\_API\_KEY), allow\_dangerous\_deserialization=True)
retriever = db.as\_retriever()
qa\_chain = RetrievalQA.from\_chain\_type(
llm=ChatOpenAI(model="gpt-4", openai\_api\_key=OPENAI\_API\_KEY),
retriever=retriever,
return\_source\_documents=True
)
except Exception as e:
st.error(f"❌ Could not load index for {council}: {str(e)}")
st.stop()

# --- Handle Question ---

if question:
st.session\_state.query\_count += 1
st.write("🔎 Searching documents...")
try:
result = qa\_chain({"query": question})
st.success(result\["result"])
with st.expander("📄 View sources"):
for doc in result\["source\_documents"]:
st.caption(f"• {doc.metadata.get('source', 'Unknown source')}")
except Exception as e:
st.error(f"❌ Error: {str(e)}")

# --- PDF Upload + FAISS Rebuild ---

def process\_and\_index\_pdf(uploaded\_files):
try:
all\_docs = \[]
for file in uploaded\_files:
with open("temp.pdf", "wb") as f:
f.write(file.read())
loader = PyPDFLoader("temp.pdf")
all\_docs.extend(loader.load())

```
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    faiss_index = FAISS.from_documents(split_docs, embeddings)
    faiss_index.save_local(index_path)

    st.success(f"✅ Index for {council} updated with {len(uploaded_files)} file(s).")
    st.experimental_rerun()

except Exception as e:
    st.error(f"❌ Failed to process: {str(e)}")
```

if st.session\_state.is\_admin:
uploaded\_files = st.file\_uploader(f"📤 Upload PDFs for {council}", type="pdf", accept\_multiple\_files=True)
if uploaded\_files and st.button("🔄 Rebuild Index"):
process\_and\_index\_pdf(uploaded\_files)

# --- Gmail Auto-Reply with GPT ---

def gmail\_auto\_reply():
try:
imap = imaplib.IMAP4\_SSL("imap.gmail.com")
imap.login(GMAIL\_USER, GMAIL\_PASS)
imap.select("inbox")
status, messages = imap.search(None, 'UNSEEN')

```
    for num in messages[0].split():
        _, data = imap.fetch(num, "(RFC822)")
        raw_email = data[0][1]
        msg = email.message_from_bytes(raw_email)
        sender = email.utils.parseaddr(msg['From'])[1]
        subject = msg['Subject']
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = msg.get_payload(decode=True).decode()

        prompt = f"You are an AI assistant replying to a council inquiry email. The email says: '{body}'. Write a professional and helpful reply."
        reply = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY).invoke(prompt)

        reply_msg = MIMEText(reply)
        reply_msg["Subject"] = f"Re: {subject}"
        reply_msg["From"] = GMAIL_USER
        reply_msg["To"] = sender

        smtp = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        smtp.login(GMAIL_USER, GMAIL_PASS)
        smtp.sendmail(GMAIL_USER, sender, reply_msg.as_string())
        smtp.quit()

    imap.logout()
    st.success("✅ Auto-replied to all unread emails.")
except Exception as e:
    st.error(f"❌ Gmail auto-reply failed: {str(e)}")
```

if st.session\_state.is\_admin and st.button("📬 Auto-Reply to Council Emails"):
gmail\_auto\_reply()

# --- Footer ---

st.markdown(f"""

<div class="footer">
  ⚙️ Powered by LangChain + GPT-4 | Queries used: {st.session_state.query_count} / 500<br>
  📬 Contact: <a href="mailto:contact@civreply.ai">contact@civreply.ai</a>
</div>
""", unsafe_allow_html=True)
