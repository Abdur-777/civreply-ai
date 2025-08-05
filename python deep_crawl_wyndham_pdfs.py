import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import threading
import queue
import time

BASE = "https://www.wyndham.vic.gov.au"
START = BASE + "/"
SAVE_DIR = "docs/wyndham"
os.makedirs(SAVE_DIR, exist_ok=True)

visited = set()
pdf_downloaded = set()
link_queue = queue.Queue()
link_queue.put(START)
domain = urlparse(BASE).netloc

MAX_THREADS = 8  # adjust for your connection
MAX_DEPTH = 5    # reasonable for council websites

def download_pdf(pdf_url):
    filename = os.path.join(SAVE_DIR, pdf_url.split('/')[-1])
    if os.path.exists(filename):
        pdf_downloaded.add(pdf_url)
        return
    try:
        r = requests.get(pdf_url, timeout=20)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)
        print("✅ Downloaded", pdf_url)
        pdf_downloaded.add(pdf_url)
    except Exception as e:
        print("❌ Failed to download", pdf_url, e)
        with open("failed_pdfs.txt", "a") as log:
            log.write(f"{pdf_url}\n")

def crawl_page(url, depth):
    if url in visited or depth > MAX_DEPTH:
        return
    visited.add(url)
    try:
        resp = requests.get(url, timeout=10)
        if not resp.ok:
            return
        soup = BeautifulSoup(resp.text, "html.parser")
        # Find PDFs
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(url, href)
            if ".pdf" in href.lower():
                if full_url not in pdf_downloaded:
                    download_pdf(full_url)
            elif is_internal(full_url):
                if full_url not in visited:
                    link_queue.put((full_url, depth+1))
    except Exception as e:
        print("❌ Error crawling", url, e)

def is_internal(url):
    parsed = urlparse(url)
    return (parsed.netloc == domain) or (parsed.netloc == "")

def worker():
    while True:
        try:
            url, depth = link_queue.get(timeout=10)
        except queue.Empty:
            return
        try:
            crawl_page(url, depth)
        finally:
            link_queue.task_done()

# Seed the queue with (url, depth)
link_queue = queue.Queue()
link_queue.put((START, 0))

threads = []
for _ in range(MAX_THREADS):
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    threads.append(t)

start_time = time.time()
link_queue.join()
print("All done! Time:", round(time.time()-start_time, 2), "seconds")
