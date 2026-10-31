"""
app/ingestion/loaders.py
Loaders for .md, .txt, .html files and URLs. Each returns list[Document] with metadata:
source, title, doc_id, file_type.

URL loading is hardened: http(s) only, public IPs only (blocks localhost / private networks /
cloud metadata addresses), redirects re-validated, 5 MB response cap.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import socket
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

MAX_URL_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 3


def _slug(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower().strip())
    return re.sub(r"[\s_-]+", "_", text)[:60]


def _derive_title(path_or_url: str) -> str:
    return Path(path_or_url).stem.replace("_", " ").title()


def _html_to_text(raw_html: str) -> tuple[str, Optional[str]]:
    """Strip nav/boilerplate tags and convert to Markdown-ish text. Returns (text, page_title)."""
    import html2text
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style", "noscript"]):
        tag.decompose()
    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else None

    conv = html2text.HTML2Text()
    conv.ignore_links = False
    conv.ignore_images = True
    conv.body_width = 0
    return conv.handle(str(soup)), page_title


def _file_doc(path: Path, text: str, title: Optional[str], doc_id: Optional[str], file_type: str) -> List[Document]:
    return [Document(
        page_content=text,
        metadata={"source": str(path), "title": title or _derive_title(str(path)),
                  "doc_id": doc_id or path.stem, "file_type": file_type},
    )]


def load_markdown(path: Path, title: Optional[str] = None, doc_id: Optional[str] = None) -> List[Document]:
    return _file_doc(path, path.read_text(encoding="utf-8", errors="replace"), title, doc_id, "markdown")


def load_text(path: Path, title: Optional[str] = None, doc_id: Optional[str] = None) -> List[Document]:
    return _file_doc(path, path.read_text(encoding="utf-8", errors="replace"), title, doc_id, "text")


def load_html_file(path: Path, title: Optional[str] = None, doc_id: Optional[str] = None) -> List[Document]:
    text, page_title = _html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    return _file_doc(path, text, title or page_title, doc_id, "html")


def load_file(path: Path, title: Optional[str] = None, doc_id: Optional[str] = None) -> List[Document]:
    ext = path.suffix.lower()
    if ext in (".md", ".markdown"):
        return load_markdown(path, title, doc_id)
    if ext == ".txt":
        return load_text(path, title, doc_id)
    if ext in (".html", ".htm"):
        return load_html_file(path, title, doc_id)
    raise ValueError(f"Unsupported file type: {ext}")


# ── URL loading ────────────────────────────────────────────────────────────────
def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only http(s) URLs are allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve host '{parsed.hostname}'") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("URL points to a non-public address")


def _fetch(url: str) -> requests.Response:
    """GET with SSRF checks on every hop and a response size cap."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        _validate_public_url(current)
        resp = requests.get(current, timeout=30, stream=True, allow_redirects=False,
                            headers={"User-Agent": "RAG-Docs-Bot/1.0"})
        if resp.is_redirect and resp.headers.get("location"):
            current = urljoin(current, resp.headers["location"])
            resp.close()
            continue
        resp.raise_for_status()
        body = b""
        for chunk in resp.iter_content(65536):
            body += chunk
            if len(body) > MAX_URL_BYTES:
                resp.close()
                raise ValueError("Response too large (> 5 MB)")
        resp._content = body  # make .content/.text usable after streaming
        return resp
    raise ValueError("Too many redirects")


def load_url(url: str, title: Optional[str] = None) -> List[Document]:
    logger.info("Fetching URL: %s", url)
    try:
        resp = _fetch(url)
    except requests.RequestException as exc:
        raise ValueError(f"Failed to fetch {url}: {exc}") from exc

    if "text/html" in resp.headers.get("content-type", ""):
        text, page_title = _html_to_text(resp.text)
        title = title or page_title or _derive_title(url)
    else:
        text = resp.text
        title = title or _derive_title(url)

    parsed = urlparse(url)
    doc_id = f"{_slug(parsed.netloc + parsed.path) or 'page'}_{hashlib.sha1(url.encode()).hexdigest()[:8]}"
    return [Document(page_content=text,
                     metadata={"source": url, "title": title, "doc_id": doc_id, "file_type": "url"})]
