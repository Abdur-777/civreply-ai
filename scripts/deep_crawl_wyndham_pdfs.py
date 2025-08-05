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
domain = urlparse(BASE).netloc
queue = [START]

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

while queue:
    url = queue.pop(0)
    if url in visited_pages:
        continue
    try:
        resp = requests.get(url, timeout=12)
        if not resp.ok:
            continue
        visited_pages.add(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        # For every link on the page
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"].split("#")[0])
            if is_pdf(link):
                fname = os.path.basename(link.split("?")[0])
                save_path = os.path.join(SAVE_DIR, fname)
                if not os.path.exists(save_path):
                    print(f"Downloading: {fname}")
                    try:
                        r = requests.get(link, timeout=20)
                        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("application/pdf"):
                            with open(save_path, "wb") as f:
                                f.write(r.content)
                            print(f"Saved to: {save_path}")
                            open_pdf_mac(save_path)
                        else:
                            print(f"Failed to download {link}")
                    except Exception as e:
                        print(f"Error downloading {link}: {e}")
                else:
                    print(f"Already have {fname}, skipping.")
            elif is_internal(link) and link not in visited_pages and link not in queue:
                if not any(link.lower().endswith(ext) for ext in [
                    ".jpg", ".jpeg", ".png", ".gif", ".zip", ".doc", ".docx",
                    ".xls", ".xlsx", ".ppt", ".pptx", ".mp4", ".mov", ".mp3"
                ]):
                    queue.append(link)
        time.sleep(0.1)  # Be polite to the server
    except Exception as e:
        print(f"Error visiting {url}: {e}")

print("\n✅ Done! All Wyndham PDFs in", SAVE_DIR)
