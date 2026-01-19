from __future__ import annotations

import base64
import csv
import io
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import pdfplumber

try:
    from firecrawl import FirecrawlApp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FirecrawlApp = None  # type: ignore[assignment]

AUTHORIZED_SPONSOR_CACHE: set[str] | None = None
AUTHORIZED_SPONSOR_PATH = Path(__file__).resolve().parent / "2025-11-07_-_Worker_and_Temporary_Worker.csv"


# Load environment variables to mirror SAMPLE_FIRECRAWL.py behaviour
load_dotenv()


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def make_request_id(prefix: str = "req") -> str:
    seed = f"{prefix}-{time.time_ns()}-{os.getpid()}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def decode_base64_pdf(b64: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 PDF content") from e


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_chars: int = 20000) -> str:
    if not pdf_bytes or len(pdf_bytes) < 10:
        raise ValueError("Empty or invalid PDF bytes")
    
    # Validate that this is actually a PDF by checking the PDF header
    # PDF files should start with "%PDF-" (bytes: 25 50 44 46 2D)
    if not pdf_bytes.startswith(b'%PDF-'):
        # Try to find PDF header in first 1024 bytes (some PDFs have metadata before header)
        found_header = False
        for i in range(min(1024, len(pdf_bytes) - 5)):
            if pdf_bytes[i:i+5] == b'%PDF-':
                found_header = True
                pdf_bytes = pdf_bytes[i:]  # Remove any prefix before PDF header
                break
        if not found_header:
            raise ValueError(f"Invalid PDF: File does not start with PDF header. First 50 bytes: {pdf_bytes[:50]}")
    
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            try:
                txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:
                txt = ""
            if txt:
                text_parts.append(txt)
            if sum(len(p) for p in text_parts) >= max_chars:
                break
    text = "\n".join(text_parts).strip()
    if not text:
        raise ValueError("Failed to extract text from PDF; consider OCR fallback")
    return text[:max_chars]


def redact_long_text(text: str, max_len: int = 800) -> str:
    if len(text) <= max_len:
        return text
    head = text[: max_len // 2]
    tail = text[-max_len // 2 :]
    return head + "\n...\n" + tail


# Firecrawl-based scraping with compatibility fallbacks
def scrape_website_custom(url: str, api_key: Optional[str]) -> dict:
    if FirecrawlApp is None:
        return {"error": "firecrawl-sdk-missing"}

    firecrawl_api_key = api_key or os.getenv("FIRECRAWL_API_KEY") or ""

    try:
        app = FirecrawlApp(api_key=firecrawl_api_key)
    except Exception as exc:
        return {"error": f"firecrawl-init-failed: {exc}"}

    try:
        result = app.scrape(url=url)
        if isinstance(result, dict):
            return result
        return {"content": str(result)}
    except AttributeError:
        try:
            result = app.scrape_url(url)
            if isinstance(result, dict):
                return result
            return {"content": str(result)}
        except Exception as exc:
            return {"error": f"scrape-failed: {exc}"}
    except Exception as exc:
        return {"error": f"scrape-error: {exc}"}


def _normalize_company_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _load_authorized_sponsors() -> set[str]:
    global AUTHORIZED_SPONSOR_CACHE
    if AUTHORIZED_SPONSOR_CACHE is not None:
        return AUTHORIZED_SPONSOR_CACHE

    sponsors: set[str] = set()
    if not AUTHORIZED_SPONSOR_PATH.exists():
        AUTHORIZED_SPONSOR_CACHE = sponsors
        return sponsors

    with AUTHORIZED_SPONSOR_PATH.open("r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        header_variants = {"organisation name", "organization name", "\ufefforganisation name"}
        for row in reader:
            if not row:
                continue
            name = None
            for key in header_variants:
                if key in row and row[key]:
                    name = row[key]
                    break
            if not name:
                # try original casing
                for key in list(row.keys()):
                    if key.lower() in header_variants and row[key]:
                        name = row[key]
                        break
            if not name:
                continue
            sponsors.add(_normalize_company_name(name))

    AUTHORIZED_SPONSOR_CACHE = sponsors
    return sponsors


def is_authorized_sponsor(company_name: Optional[str]) -> Optional[bool]:
    if not company_name:
        return None
    normalized = _normalize_company_name(company_name)
    if not normalized:
        return None
    sponsors = _load_authorized_sponsors()
    if not sponsors:
        return None
    return normalized in sponsors
