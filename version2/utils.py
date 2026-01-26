from __future__ import annotations

import base64
import csv
import io
import os
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import pdfplumber

logger = logging.getLogger(__name__)

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
    """
    Extract text from PDF bytes. Optimized to use direct text extraction first
    (faster for text-based PDFs) before falling back to OCR.
    
    This function uses pdfplumber which extracts text directly from PDF structure,
    which is much faster than OCR for text-based PDFs.
    """
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
    
    # Optimized: Use pdfplumber's direct text extraction (fast for text-based PDFs)
    # This is much faster than OCR and works for most modern PDFs
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            try:
                # Direct text extraction (fast for text-based PDFs)
                # x_tolerance and y_tolerance help with spacing issues
                txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:
                txt = ""
            if txt:
                text_parts.append(txt)
            if sum(len(p) for p in text_parts) >= max_chars:
                break
    text = "\n".join(text_parts).strip()
    
    # If no text extracted, this might be an image-based PDF requiring OCR
    # But for now, we raise an error (OCR fallback would be handled elsewhere)
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


def convert_html_to_markdown(html_content: str) -> str:
    """
    Convert HTML content to Markdown format for better LLM processing.
    
    This function:
    - Converts HTML tags to Markdown syntax
    - Preserves structure (headers, lists, paragraphs)
    - Removes HTML noise (scripts, styles, navigation)
    - Cleans up whitespace and formatting
    
    Args:
        html_content: Raw HTML string from job posting
        
    Returns:
        Clean Markdown-formatted string optimized for LLM processing
    """
    if not html_content or not isinstance(html_content, str):
        return html_content or ""
    
    # Check if content is already plain text (no HTML tags)
    if not any(tag in html_content for tag in ['<', '>', '&lt;', '&gt;']):
        # Already plain text, return as-is
        return html_content.strip()
    
    try:
        from markdownify import markdownify as md
        
        # Convert HTML to Markdown
        # Options:
        # - heading_style: Use ATX style (# ## ###) for headers
        # - bullets: Use - for unordered lists
        # - strip: Remove HTML tags that don't have Markdown equivalents
        # - convert: Convert specific tags (default: all)
        markdown_text = md(
            html_content,
            heading_style="ATX",  # Use # for headers
            bullets="-",  # Use - for lists
            strip=['script', 'style', 'nav', 'header', 'footer', 'aside'],  # Remove noise
            convert=['p', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'strong', 'em', 'b', 'i', 'a', 'div', 'span'],
        )
        
        # Clean up extra whitespace and empty lines
        lines = []
        prev_empty = False
        for line in markdown_text.split('\n'):
            stripped = line.strip()
            if not stripped:
                if not prev_empty:
                    lines.append('')
                prev_empty = True
            else:
                lines.append(stripped)
                prev_empty = False
        
        result = '\n'.join(lines).strip()
        
        # If conversion resulted in very short text, the HTML might be minimal
        # Return original if markdown is suspiciously short
        if len(result) < 50 and len(html_content) > 200:
            logger.warning("HTML to Markdown conversion produced suspiciously short result, using original")
            # Try BeautifulSoup fallback
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                # Get text and clean up
                text = soup.get_text()
                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines()]
                lines = [line for line in lines if line]
                result = '\n'.join(lines)
            except Exception as e:
                logger.warning(f"BeautifulSoup fallback also failed: {e}, using original HTML")
                return html_content.strip()
        
        logger.info(f"Converted HTML to Markdown: {len(html_content)} chars -> {len(result)} chars")
        return result
        
    except ImportError:
        logger.warning("markdownify not installed, falling back to BeautifulSoup for HTML to Markdown conversion")
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            
            # Convert common HTML elements to Markdown-like format
            # Headers
            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                for element in soup.find_all(tag):
                    level = int(tag[1])
                    element.string = f"{'#' * level} {element.get_text().strip()}\n"
                    element.name = 'p'
            
            # Lists
            for ul in soup.find_all('ul'):
                for li in ul.find_all('li', recursive=False):
                    li.string = f"- {li.get_text().strip()}\n"
            
            for ol in soup.find_all('ol'):
                for idx, li in enumerate(ol.find_all('li', recursive=False), 1):
                    li.string = f"{idx}. {li.get_text().strip()}\n"
            
            # Bold and italic
            for tag in soup.find_all(['strong', 'b']):
                tag.string = f"**{tag.get_text().strip()}**"
            
            for tag in soup.find_all(['em', 'i']):
                tag.string = f"*{tag.get_text().strip()}*"
            
            # Links
            for a in soup.find_all('a', href=True):
                text = a.get_text().strip()
                href = a.get('href', '')
                a.string = f"[{text}]({href})"
            
            # Get text and clean up
            text = soup.get_text()
            lines = [line.strip() for line in text.splitlines()]
            lines = [line for line in lines if line]
            result = '\n'.join(lines)
            
            logger.info(f"Converted HTML to Markdown (BeautifulSoup fallback): {len(html_content)} chars -> {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"HTML to Markdown conversion failed: {e}, returning original content")
            return html_content.strip()
    
    except Exception as e:
        logger.error(f"HTML to Markdown conversion error: {e}, returning original content")
        return html_content.strip()
