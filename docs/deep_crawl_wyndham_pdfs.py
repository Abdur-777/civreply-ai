import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time

BASE = "https://www.wyndham.vic.gov.au"
START = BASE + "/"
SAVE_DIR = "docs/wyndham"
os.makedirs(SAVE_DIR, exist_ok=True)

visited_pages = set()
found_pdfs = set()
queue = [START]
domain = urlparse(BASE).netloc

def is_pdf(url):
    return url.lower().endswith('.pdf')

def is_internal(url):
    parsed = urlparse(url)
    return (parsed.netloc == domain) or (parsed.netloc == "")

while queue:
    url = queue.pop(0)
    if url in visited_pages:
        continue
    print("Visiting:", url)
    try:
        resp = requests.get(url, timeout=10)
        if not resp.ok:
            continue
        visited_pages.add(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Find all links
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            if is_pdf(link):
                found_pdfs.add(link.split("?")[0])
            elif is_internal(link) and link not in visited_pages and link not in queue:
                # Only crawl HTML pages, not files
                if any(link.lower().endswith(ext) for ext in [".jpg", ".png", ".gif", ".zip", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"]):
                    continue
                queue.append(link)
        time.sleep(0.15)  # polite crawling
    except Exception as e:
        print("Error:", url, e)

print(f"\nFound {len(found_pdfs)} PDFs. Downloading...\n")

for url in sorted(found_pdfs):
    fname = os.path.basename(url)
    save_path = os.path.join(SAVE_DIR, fname)
    if os.path.exists(save_path):
        print(f"Already have {fname}, skipping.")
        continue
    print("Downloading:", fname)
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("application/pdf"):
            with open(save_path, "wb") as f:
                f.write(r.content)
        else:
            print("Failed to download", url)
    except Exception as e:
        print("Error downloading", url, e)

print("\n✅ Done! All Wyndham PDFs in", SAVE_DIR)
