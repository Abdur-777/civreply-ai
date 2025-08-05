import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
import subprocess

BASE = "https://www.wyndham.vic.gov.au"
START = BASE + "/"
SAVE_DIR = "docs/wyndham"
os.makedirs(SAVE_DIR, exist_ok=True)

visited_pages = set()
found_pdfs = set()
queue = [(START, 0)]
domain = urlparse(BASE).netloc
MAX_DEPTH = 10        # You can set this as high as you want for "deepness"
MAX_PAGES = 5000      # Optional: put a safety limit on total crawled pages

def is_pdf(url):
    return url.lower().endswith('.pdf')

def is_internal(url):
    parsed = urlparse(url)
    return (parsed.netloc == domain) or (parsed.netloc == "")

def open_pdf_mac(path):
    try:
        subprocess.run(["open", path])
    except Exception as e:
        print(f"Could not open {path}: {e}")

pages_crawled = 0
while queue and pages_crawled < MAX_PAGES:
    url, depth = queue.pop(0)
    if url in visited_pages or depth > MAX_DEPTH:
        continue
    print(f"Visiting (depth {depth}): {url}")
    try:
        resp = requests.get(url, timeout=15)
        if not resp.ok:
            continue
        visited_pages.add(url)
        pages_crawled += 1
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"].split("#")[0])  # ignore URL fragments
            if is_pdf(link):
                found_pdfs.add(link.split("?")[0])
            elif is_internal(link) and link not in visited_pages:
                if not any(link.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".zip", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".mp4", ".mov", ".mp3"]):
                    queue.append((link, depth + 1))
        time.sleep(0.1)  # polite crawling
    except Exception as e:
        print(f"Error visiting {url}: {e}")

print(f"\nFound {len(found_pdfs)} PDFs. Downloading...\n")

for url in sorted(found_pdfs):
    fname = os.path.basename(url.split("?")[0])
    save_path = os.path.join(SAVE_DIR, fname)
    if os.path.exists(save_path):
        print(f"Already have {fname}, skipping.")
        continue
    print("Downloading:", fname)
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("application/pdf"):
            with open(save_path, "wb") as f:
                f.write(r.content)
            print(f"Saved to: {save_path}")
            open_pdf_mac(save_path)
        else:
            print(f"Failed to download {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

print("\n✅ Done! All Wyndham PDFs in", SAVE_DIR)
print(f"Total pages crawled: {pages_crawled}")
print(f"Total PDFs found: {len(found_pdfs)}")
