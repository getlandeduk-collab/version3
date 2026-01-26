from __future__ import annotations

import asyncio
import html
import json
import os
import time
import re
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional

# Setup logging with environment variable control
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, BackgroundTasks

from fastapi.responses import JSONResponse, StreamingResponse, Response

from fastapi.middleware.cors import CORSMiddleware

from pydantic import ValidationError

from dotenv import load_dotenv

from pathlib import Path

from models import (
    MatchJobsJsonRequest,
    MatchJobsRequest,
    CandidateProfile,
    JobPosting,
    ProgressStatus,
    Settings,
    SponsorshipInfo,
    SponsorshipCheckRequest,
    FirebaseResume,
    FirebaseResumeListResponse,
    FirebaseResumeResponse,
    SavedCVResponse,
    GetUserResumesRequest,
    GetUserResumeRequest,
    GetUserResumePdfRequest,
    GetUserResumeBase64Request,
    GetUserSavedCvsRequest,
)
from utils import (
    decode_base64_pdf,
    extract_text_from_pdf_bytes,
    now_iso,
    make_request_id,
)
from experience_parser import (
    calculate_experience_breakdown,
    parse_date_range,
    classify_experience_type,
    parse_duration_string,
)

from pyngrok import ngrok, conf as ngrok_conf

# Optional imports for HTML parsing

try:

    import requests

    from bs4 import BeautifulSoup

except ImportError:

    requests = None

    BeautifulSoup = None

# Load environment from root .env and version2/.env if present

load_dotenv()  # project root

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# CRITICAL: Ensure GOOGLE_APPLICATION_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS is explicitly set from system environment

# This is needed because async context might not have access to system env vars

# Priority: GOOGLE_APPLICATION_CREDENTIALS_JSON (JSON string) > GOOGLE_APPLICATION_CREDENTIALS (file path)

# Check for JSON string first (preferred for production)

if "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in os.environ:

    # Try to get from system environment (Windows environment variables)

    import sys

    import subprocess

    try:

        # On Windows, try to get from system environment

        result = subprocess.run(

            ['powershell', '-Command', '[Environment]::GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS_JSON", "User")'],

            capture_output=True,

            text=True,

            timeout=2

        )

        if result.returncode == 0 and result.stdout.strip():

            os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = result.stdout.strip()

    except:

        pass  # Non-critical, continue anyway

# Fallback to file path method if JSON not found

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:

    # Try to get from system environment (Windows environment variables)

    import sys

    import subprocess

    try:

        # On Windows, try to get from system environment

        result = subprocess.run(

            ['powershell', '-Command', '[Environment]::GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", "User")'],

            capture_output=True,

            text=True,

            timeout=2

        )

        if result.returncode == 0 and result.stdout.strip():

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = result.stdout.strip()

    except:

        pass  # Non-critical, continue anyway

app = FastAPI(title="Intelligent Job Matching API", version="0.1.0")

# SSE Event Formatting Helper
def format_sse_event(event_type: str, data: dict) -> str:
    """
    Format data as SSE event.
    
    SSE Format:
    event: event_name
    data: {"type": "event_name", "key": "value"}
    
    """
    # Include type in data payload for frontend compatibility
    data_with_type = {"type": event_type, **data}
    return f"event: {event_type}\ndata: {json.dumps(data_with_type)}\n\n"

# Ngrok startup (optional)

NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")

NGROK_DOMAIN = os.getenv("NGROK_DOMAIN") or os.getenv("NGROK_URL") or "gobbler-fresh-sole.ngrok-free.app"

if NGROK_AUTHTOKEN and not os.getenv("DISABLE_NGROK"):

    try:

        ngrok_conf.get_default().auth_token = NGROK_AUTHTOKEN

        # Ensure no old tunnels keep port busy

        for t in ngrok.get_tunnels():

            try:

                ngrok.disconnect(t.public_url)

            except Exception:

                pass

        if NGROK_DOMAIN:

            print(f"[NGROK] Connecting to domain: {NGROK_DOMAIN}")

            ngrok.connect(addr="8000", proto="http", domain=NGROK_DOMAIN)

            # Get the public URL

            tunnels = ngrok.get_tunnels()

            if tunnels:

                print(f"[NGROK] Public URL: {tunnels[0].public_url}")

        else:

            ngrok.connect(addr="8000", proto="http")

    except Exception as e:

        # Non-fatal if ngrok fails

        print(f"[NGROK] Error: {e}")

        pass

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)

# Resume parsing cache (hash-based)
# Key: SHA256 hash of resume text, Value: parsed candidate profile dict
RESUME_PARSING_CACHE: Dict[str, Dict[str, Any]] = {}

# Startup event: Preload sponsorship CSV data
@app.on_event("startup")
async def startup_event():
    """Preload sponsorship CSV data at application startup for faster lookups."""
    try:
        from sponsorship_checker import load_sponsorship_data
        logger.info("Preloading sponsorship CSV data...")
        load_sponsorship_data()  # This will cache the data
        logger.info("Sponsorship CSV data loaded and cached successfully")
    except Exception as e:
        logger.warning(f"Failed to preload sponsorship CSV (non-fatal): {e}")
        # Non-fatal - will load on first use


def get_settings() -> Settings:

    return Settings(

        openai_api_key=os.getenv("OPENAI_API_KEY"),

        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),

        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),

        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),

        max_concurrent_scrapes=int(os.getenv("MAX_CONCURRENT_SCRAPES", "8")),

        rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),

        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),

    )


def clean_job_title(title: Optional[str]) -> Optional[str]:
    """
    Clean and normalize job title, removing patterns like "job_title:**Name:M/L developer".
    
    Args:
        title: Raw job title string
        
    Returns:
        Cleaned job title or None if invalid
    """
    if not title or not isinstance(title, str):
        return None
    
    # Remove leading/trailing whitespace
    title = title.strip()
    
    # Remove patterns like "job_title:**", "job_title:", "Title:**", "Name:**", etc.
    title = re.sub(r'^(job_title|title|name|position|role)\s*[:\*]+\s*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^\*+\s*', '', title)  # Remove leading asterisks
    title = re.sub(r'\s*\*+\s*$', '', title)  # Remove trailing asterisks
    
    # Remove common prefixes/suffixes
    title = re.sub(r'^[^:]*:\s*', '', title)  # Remove "Job Board: " or "Category: "
    title = re.sub(r'\s*[-–—|]\s*at\s+[^-]+$', '', title, flags=re.I)  # Remove " - at Company Name"
    title = re.sub(r'\s*[-–—|]\s*[^-]+(?:\.com|\.in|\.org).*$', '', title, flags=re.I)  # Remove website suffixes
    # Only remove " - Company Name" if it looks like a company name (not part of job title like "Front-End")
    # Be very conservative - only remove if it's clearly a company suffix pattern
    # Pattern: " - CompanyName" or " - Company Name Ltd" at the end, but NOT if it contains job keywords
    job_keywords = ['developer', 'engineer', 'manager', 'analyst', 'specialist', 'architect', 'designer', 'scientist', 'consultant', 'coordinator', 'officer', 'executive', 'director', 'lead', 'senior', 'junior', 'front', 'back', 'end', 'full', 'stack', 'react', 'angular', 'vue', 'node', 'python', 'java', 'c++']
    # Check if title contains job keywords - if so, don't remove anything after dash (it's part of title)
    if not any(keyword in title.lower() for keyword in job_keywords):
        # Only remove company suffixes if no job keywords found
        # Match patterns like " - Robert Half" or " - Company Ltd" at the end
        title = re.sub(r'\s*[-–—|]\s*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Ltd|Limited|Inc|Corp|LLC|LLP|PLC))?)$', '', title)
    title = re.sub(r'\s*[|]\s*', ' ', title)  # Replace pipe separators with space
    
    # Remove quotes and special characters at start/end
    title = re.sub(r'^["\'\`]+|["\'\`]+$', '', title)
    
    # Normalize whitespace
    title = re.sub(r'\s+', ' ', title)
    title = title.strip()
    
    # Validate title quality
    if not title or len(title) < 3:
        return None
    
    if len(title) > 150:
        title = title[:150].strip()
    
    # Remove if it looks like navigation or invalid
    invalid_patterns = [
        r'^(home|menu|navigation|skip to|cookie|privacy policy)',
        r'^(not specified|unknown|n/a|na|none)$',
        r'^[\*\-\s]+$',  # Only asterisks, dashes, or spaces
    ]
    for pattern in invalid_patterns:
        if re.match(pattern, title, re.IGNORECASE):
            return None
    
    return title

def clean_company_name(company: Optional[str]) -> Optional[str]:
    """
    Clean and normalize company name, removing patterns like "Name**: Company".
    
    Args:
        company: Raw company name string
        
    Returns:
        Cleaned company name or None if invalid
    """
    if not company or not isinstance(company, str):
        return None
    
    # Decode HTML entities (e.g., &amp; -> &, &lt; -> <, &gt; -> >)
    company = html.unescape(company)
    
    # Remove leading/trailing whitespace
    company = company.strip()
    
    # Remove patterns like "Name**:", "Company:", "Employer:", etc.
    company = re.sub(r'^(Name\*{0,2}:?\s*|Company:?\s*|Employer:?\s*|Organization:?\s*)', '', company, flags=re.IGNORECASE)
    company = re.sub(r'^\*+\s*', '', company)  # Remove leading asterisks
    company = re.sub(r'\s*\*+\s*$', '', company)  # Remove trailing asterisks
    
    # Remove common prefixes
    company = re.sub(r'^at\s+', '', company, flags=re.IGNORECASE)
    company = re.sub(r'^for\s+', '', company, flags=re.IGNORECASE)
    company = re.sub(r'^with\s+', '', company, flags=re.IGNORECASE)
    company = re.sub(r'^by\s+', '', company, flags=re.IGNORECASE)
    
    # Remove quotes and special characters
    company = re.sub(r'^["\'\`]+|["\'\`]+$', '', company)
    company = re.sub(r'^[\*\-\s]+|[\*\-\s]+$', '', company)
    
    # Remove truncated text indicators
    if company.endswith('...') or company.endswith('…'):
        return None  # Truncated company names are invalid
    
    # Remove if it ends mid-word (likely truncated)
    if len(company) > 50 and not company[-1].isalnum() and not company.endswith(('Ltd', 'Inc', 'LLC', 'Corp', 'Corporation', 'Group', 'Holdings')):
        # Likely truncated
        return None
    
    # Normalize whitespace
    company = re.sub(r'\s+', ' ', company)
    company = company.strip()
    
    # Validate company name quality
    if not company or len(company) < 3:
        return None
    
    # Reject if too long (likely description text)
    if len(company) > 80:
        return None
    
    # Remove if it looks invalid
    invalid_patterns = [
        r'^(not specified|unknown|n/a|na|none|company|employer|not available)$',
        r'^[\*\-\s]+$',  # Only asterisks, dashes, or spaces
        r'\b(transforming|leveraging|integrating|facilitating)\b',  # Contains verbs (likely description)
    ]
    for pattern in invalid_patterns:
        if re.search(pattern, company, re.IGNORECASE):
            return None
    
    # Reject sentence fragments that start with verbs (common extraction errors)
    company_lower = company.lower().strip()
    verb_starters = ["works", "work", "provides", "provide", "develops", "develop", "creates", "create",
                     "helps", "help", "gathers", "gather", "produces", "produce", "supports", "support",
                     "builds", "build", "connecting", "connect", "delivering", "deliver", "offering",
                     "offer", "serving", "serve", "enabling", "enable", "making", "make", "transforming",
                     "transforms", "leveraging", "leverages", "facilitating", "facilitates", "integrating",
                     "integrates", "specializing", "specializes", "working", "collaborating", "collaborates"]
    
    # Check if it starts with a verb (sentence fragment pattern)
    if any(company_lower.startswith(verb + " ") for verb in verb_starters):
        return None  # Sentence fragment, not a company name
    
    # Check if it starts with lowercase article/preposition (likely sentence fragment)
    # BUT: Allow "by [Company Name]" pattern - we'll extract the company name part
    if company_lower.startswith(("to ", "the ", "a ", "an ", "with ", "for ", "at ")):
        # Only reject if it's clearly a sentence fragment (ends with "to" or is too long)
        if company.endswith(" to") or len(company) > 50:
            return None
    # Handle "by [Company Name]" pattern - extract the company name part
    elif company_lower.startswith("by "):
        # Extract the part after "by "
        extracted = company[3:].strip()  # Remove "by " prefix
        if extracted and len(extracted) >= 2:
            # Recursively clean the extracted part
            return clean_company_name(extracted)
        return None
    
    # Check if it ends with " to" (sentence fragment pattern like "works with food retailers to")
    if company.lower().endswith(" to") or company.lower().endswith(" to "):
        return None
    
    return company

def clean_summary_text(text: Optional[str]) -> str:
    """
    Clean summary text to remove markdown formatting inconsistencies like "Name**: Value".
    
    Args:
        text: Raw summary text that may contain markdown formatting
        
    Returns:
        Cleaned summary text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove markdown code fences
    text = re.sub(r'^```[\w]*\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    
    # Remove patterns like "Name**:", "**Name**:", "Name:", etc. at the start of lines
    # This handles cases like "Name**: Clarity" or "**Company**: ABC Corp"
    text = re.sub(r'^(\*{0,2}(?:Name|Company|Title|Job Title|Position|Role|Location|Salary|Description|Summary|Employer|Organization)\*{0,2}:?\s*)', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove bold markdown (**text** or __text__)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **text** -> text
    text = re.sub(r'__([^_]+)__', r'\1', text)  # __text__ -> text
    
    # Remove italic markdown (*text* or _text_)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', text)  # *text* -> text (but not **text**)
    text = re.sub(r'(?<!_)_([^_]+)_(?!_)', r'\1', text)  # _text_ -> text (but not __text__)
    
    # Remove standalone asterisks at line starts/ends
    text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\*+$', '', text, flags=re.MULTILINE)
    
    # Remove patterns like "**:**" or "**: " at line starts
    text = re.sub(r'^\*{1,2}:?\s*', '', text, flags=re.MULTILINE)
    
    # Clean up multiple consecutive asterisks
    text = re.sub(r'\*{3,}', '', text)
    
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown list markers that might be left over
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace (multiple spaces/newlines)
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Final strip
    text = text.strip()
    
    return text

def extract_job_title_from_content(content: str, fallback_title: Optional[str] = None) -> Optional[str]:
    """
    Extract job title from scraped content using multiple strategies.
    
    Args:
        content: Scraped job content
        fallback_title: Title to use if extraction fails
        
    Returns:
        Extracted job title or fallback message
    """
    if not content:
        return clean_job_title(fallback_title) or "Job title not available in posting"
    
    content_lower = content.lower()
    
    # Pattern 1: Look for "Job Title:", "Position:", "Role:" patterns
    patterns = [
        r'(?:job\s*title|position|role|title)[:\s]+([A-Z][A-Za-z0-9\s\-\/&,\.]{5,80})',
        r'(?:we\s+are\s+hiring|looking\s+for|seeking)\s+(?:a|an)?\s*([A-Z][A-Za-z0-9\s\-\/&,\.]{5,80})',
        r'^([A-Z][A-Za-z0-9\s\-\/&,\.]{5,80})\s+(?:position|role|job|opening)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            potential_title = match.group(1).strip()
            cleaned = clean_job_title(potential_title)
            if cleaned and len(cleaned) >= 5:
                return cleaned
    
    # Pattern 2: Look for common job title keywords followed by text
    job_keywords = [
        r'(senior|junior|lead|principal)?\s*(software|web|frontend|backend|full.?stack|mobile|devops|data|ml|ai|machine learning|artificial intelligence)\s+(engineer|developer|architect|scientist|analyst)',
        r'(product|project|program|engineering|technical|software|data|business|marketing|sales|operations|hr|human resources)\s+(manager|director|lead|specialist|coordinator|assistant|officer|executive)',
        r'(senior|junior|lead|principal)?\s*(designer|developer|engineer|analyst|scientist|consultant|advisor|specialist)',
    ]
    
    for pattern in job_keywords:
        matches = re.finditer(pattern, content[:2000], re.IGNORECASE)
        for match in matches:
            potential_title = match.group(0).strip()
            cleaned = clean_job_title(potential_title)
            if cleaned and len(cleaned) >= 5:
                return cleaned
    
    # Pattern 3: Try to extract from first line or heading
    lines = content.split('\n')[:10]
    for line in lines:
        line = line.strip()
        if len(line) >= 10 and len(line) <= 100:
            # Check if it looks like a job title
            if any(keyword in line.lower() for keyword in ['engineer', 'developer', 'manager', 'analyst', 'specialist', 'director', 'executive', 'coordinator', 'officer', 'assistant']):
                cleaned = clean_job_title(line)
                if cleaned:
                    return cleaned
    
    # Fallback to provided title if available
    if fallback_title:
        cleaned = clean_job_title(fallback_title)
        if cleaned:
            return cleaned
    
    return "Job title not available in posting"

def is_valid_company_name(name: str) -> bool:
    """
    Validate if extracted text looks like a real company name.
    
    Args:
        name: Potential company name
    
    Returns:
        True if it looks like a valid company name
    """
    if not name or len(name) < 3:
        return False
    
    # Reject if it's too long (likely description text)
    if len(name) > 80:
        return False
    
    # Reject invalid company name words (expanded list)
    invalid_company_words = [
        'hirer', 'employer', 'recruiter', 'hiring', 'company', 'organization', 'organisation',
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'by', 
        'leveraging', 'transforming', 'using', 'through', 'description', 'about',
        'job', 'position', 'role', 'opportunity', 'career', 'work', 'employment',
        'posting', 'listing', 'advertisement', 'ad', 'vacancy', 'opening',
        'applicant', 'candidate', 'worker', 'employee', 'staff', 'personnel',
        'skip to main content', 'skip navigation', 'skip to content', 'main content',
        'navigation', 'menu', 'home', 'about us', 'contact us', 'privacy policy',
        'terms of service', 'cookie policy', 'accessibility', 'sitemap'
    ]
    name_lower = name.lower().strip()
    # Reject if the entire name is just an invalid word
    if name_lower in invalid_company_words:
        return False
    
    # Reject if it contains invalid words (even as part of the name) - strict check for common false positives
    for invalid_word in ['hirer', 'employer', 'recruiter', 'hiring']:
        # Match whole words only, and reject if it's a short name containing these
        if re.search(r'\b' + re.escape(invalid_word) + r'\b', name_lower):
            # Reject if it's a short name (likely false positive)
            if len(name_lower.split()) <= 2:
                return False
    
    # Reject if it contains too many common words (likely description)
    common_words = ['the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'by', 'leveraging', 'transforming', 'using', 'through']
    word_count = sum(1 for word in common_words if word in name_lower)
    if word_count >= 3:
        return False
    
    # Reject if it starts with lowercase (likely mid-sentence)
    if name[0].islower():
        return False
    
    # Reject sentence fragments that start with verbs (common extraction errors)
    verb_starters = ["works", "work", "provides", "provide", "develops", "develop", "creates", "create",
                     "helps", "help", "gathers", "gather", "produces", "produce", "supports", "support",
                     "builds", "build", "connecting", "connect", "delivering", "deliver", "offering",
                     "offer", "serving", "serve", "enabling", "enable", "making", "make", "transforming",
                     "transforms", "leveraging", "leverages", "facilitating", "facilitates", "integrating",
                     "integrates", "specializing", "specializes", "working", "collaborating", "collaborates"]
    
    # Check if it starts with a verb (sentence fragment pattern)
    if any(name_lower.startswith(verb + " ") for verb in verb_starters):
        return False  # Sentence fragment, not a company name
    
    # Check if it ends with " to" (sentence fragment pattern like "works with food retailers to")
    if name_lower.endswith(" to") or name_lower.endswith(" to "):
        return False
    
    # Reject sentence fragments that start with verbs (common extraction errors like "works with food retailers to")
    verb_starters = ["works", "work", "provides", "provide", "develops", "develop", "creates", "create",
                     "helps", "help", "gathers", "gather", "produces", "produce", "supports", "support",
                     "builds", "build", "connecting", "connect", "delivering", "deliver", "offering",
                     "offer", "serving", "serve", "enabling", "enable", "making", "make", "transforming",
                     "transforms", "leveraging", "leverages", "facilitating", "facilitates", "integrating",
                     "integrates", "specializing", "specializes", "working", "collaborating", "collaborates"]
    
    # Check if it starts with a verb (sentence fragment pattern)
    if any(name_lower.startswith(verb + " ") for verb in verb_starters):
        return False  # Sentence fragment, not a company name
    
    # Check if it ends with " to" (sentence fragment pattern like "works with food retailers to")
    if name_lower.endswith(" to") or name_lower.endswith(" to "):
        return False
    
    # Reject if it contains verbs indicating it's a description
    verb_patterns = [
        r'\b(transforming|leveraging|integrating|facilitating|building|creating|developing|providing)\b',
        r'\b(we|our|their|its)\b',
    ]
    for pattern in verb_patterns:
        if re.search(pattern, name.lower()):
            return False
    
    # Require at least one capital letter (proper noun indicator)
    if not any(c.isupper() for c in name):
        return False
    
    # Reject if it's a sentence fragment (contains sentence-ending punctuation)
    if re.search(r'[.!?]\s*$', name):
        return False
    
    return True

def extract_company_name_from_content(content: str, fallback_company: Optional[str] = None) -> Optional[str]:
    """
    [LEGACY] Extract company name from scraped content using regex patterns.
    
    NOTE: This function is kept for backward compatibility and fallback scenarios.
    Primary extraction now uses Gemini API via extract_company_and_title_from_raw_data().
    
    Args:
        content: Scraped job content
        fallback_company: Company name to use if extraction fails
        
    Returns:
        Extracted company name or fallback message
    """
    if not content:
        return clean_company_name(fallback_company) or "Company name not available in posting"
    
    content_lower = content.lower()
    
    # Pattern 1: Look for explicit company labels with proper names
    # Prioritize patterns with company suffixes (Ltd, Inc, etc.)
    priority_patterns = [
        r'(?:company|employer|organization|organisation)(?:\s+description)?[:\s]+([A-Z][A-Za-z0-9\s&.,\-\']+?(?:Ltd|Limited|Inc|LLC|Corp|Corporation|Group|Holdings|Technology|Solutions|Services|Pvt\.?\s*Ltd\.?))',
        r'([A-Z][A-Za-z0-9\s&.,\-\']+?(?:Pvt\.?\s*Ltd\.?|Private Limited|Ltd\.?|Limited|Inc\.?|LLC|Corporation|Corp\.?))',
        r'(?:at|for|with)\s+([A-Z][A-Za-z0-9\s&.,\-\']+?(?:Ltd|Limited|Inc|LLC|Corp|Corporation|Group|Holdings|Technology|Solutions|Services|Pvt\.?\s*Ltd\.?))',
    ]
    
    for pattern in priority_patterns:
        matches = re.finditer(pattern, content[:1500])
        for match in matches:
            potential_company = match.group(1).strip()
            cleaned = clean_company_name(potential_company)
            if cleaned and is_valid_company_name(cleaned):
                return cleaned
    
    # Pattern 2: Look for "Company Description" or "About" sections
    section_patterns = [
        r'(?:company\s+description|about\s+(?:the\s+)?company|about\s+us)[:\s]+([A-Z][A-Za-z0-9\s&.,\-\']{3,60}?)(?:\s+is|\s+integrate|\s+provide|\.|,)',
        r'(?:at|join|work\s+at|careers\s+at)\s+([A-Z][A-Za-z0-9\s&.,\-\']{3,50}?)(?:\s*,|\s+is|\s+we)',
    ]
    
    for pattern in section_patterns:
        matches = re.finditer(pattern, content[:1000], re.IGNORECASE)
        for match in matches:
            potential_company = match.group(1).strip()
            cleaned = clean_company_name(potential_company)
            if cleaned and is_valid_company_name(cleaned):
                return cleaned
    
    # Pattern 3: Look for "by [Company]" pattern (common in job listings - HIGH PRIORITY)
    # Examples: "3 days ago by Career poster", "posted by Career poster", "by Career poster"
    by_patterns = [
        r'(?:\d+\s+days?\s+ago\s+)?by\s+([A-Z][A-Za-z0-9\s&.,\-\']{2,50}?)(?:\s+Easy\s+Apply|\s+posted|\s+is|\s+integrate|\s+provide|\s*\n|\s*$|\.)',
        r'posted\s+by\s+([A-Z][A-Za-z0-9\s&.,\-\']{2,50}?)(?:\s+Easy\s+Apply|\s+is|\s+integrate|\s+provide|\s*\n|\s*$|\.)',
        r'(?:by|from)\s+([A-Z][A-Za-z0-9\s&.,\-\']{2,50}?)(?:\s+is|\s+integrate|\s+provide|\s*\n|\s*$)',
    ]
    
    for pattern in by_patterns:
        matches = re.finditer(pattern, content[:1000], re.IGNORECASE)  # Increased search range to 1000
        for match in matches:
            potential_company = match.group(1).strip()
            cleaned = clean_company_name(potential_company)
            if cleaned and is_valid_company_name(cleaned):
                return cleaned
    
    # Fallback to provided company if available
    if fallback_company:
        cleaned = clean_company_name(fallback_company)
        if cleaned and is_valid_company_name(cleaned):
            return cleaned
    
    return "Company name not available in posting"

def extract_json_from_response(text: str) -> Dict[str, Any]:

    """Extract JSON from agent response, handling markdown code blocks and nested content."""

    if not text:

        return {}

    original_text = text

    text = text.strip()

    # Handle phi agent response objects and other response types

    if hasattr(text, 'content'):

        text = str(text.content)

    elif hasattr(text, 'messages') and text.messages:

        # Get last message content

        last_msg = text.messages[-1]

        text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)

    else:

        text = str(text)

    text = text.strip()

    # Remove markdown code fences - more comprehensive matching

    if '```json' in text:

        # Extract content between ```json and ```

        match = re.search(r'```json\s*\n?(.*?)\n?```', text, re.DOTALL)

        if match:

            text = match.group(1).strip()

    elif '```' in text:

        # Remove any code fence markers

        lines = text.split("\n")

        start_idx = 0

        end_idx = len(lines)

        # Find first non-code-fence line

        for i, line in enumerate(lines):

            if line.strip().startswith("```"):

                start_idx = i + 1

                break

        # Find last code-fence line

        for i in range(len(lines) - 1, -1, -1):

            if lines[i].strip() == "```" or lines[i].strip().startswith("```"):

                end_idx = i

                break

        text = "\n".join(lines[start_idx:end_idx]).strip()

    # Clean up common artifacts

    text = re.sub(r'^[^{]*', '', text)  # Remove leading non-JSON text

    text = re.sub(r'[^}]*$', '', text)  # Remove trailing non-JSON text

    text = text.strip()

    # Try direct JSON parse first

    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):

            return parsed

    except json.JSONDecodeError as e:

        pass

    # Try to fix common JSON issues and parse again

    fixed_text = text

    # Fix trailing commas before closing braces/brackets

    fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)

    # Try parsing after trailing comma fix

    try:

        parsed = json.loads(fixed_text)

        if isinstance(parsed, dict):

            return parsed

    except json.JSONDecodeError:

        pass

    # Try to find the largest valid JSON object in the text

    # Find all potential JSON object boundaries

    start_positions = [m.start() for m in re.finditer(r'\{', text)]

    end_positions = [m.start() for m in re.finditer(r'\}', text)]

    # Try parsing from each opening brace

    best_match = None

    best_length = 0

    for start_pos in start_positions:

        # Find matching closing brace

        brace_count = 0

        for i in range(start_pos, len(text)):

            if text[i] == '{':

                brace_count += 1

            elif text[i] == '}':

                brace_count -= 1

                if brace_count == 0:

                    # Found matching brace

                    candidate = text[start_pos:i+1]

                    try:

                        parsed = json.loads(candidate)

                        if isinstance(parsed, dict) and len(parsed) > best_length:

                            best_match = parsed

                            best_length = len(parsed)

                    except json.JSONDecodeError:

                        pass

                    break

    if best_match:

        return best_match

    # Last resort: try to extract key-value pairs using regex

    result = {}

    # Extract quoted keys and values

    kv_pattern = r'"([^"]+)":\s*([^,}\]]+)'

    matches = re.finditer(kv_pattern, text)

    for match in matches:

        key = match.group(1)

        value = match.group(2).strip()

        # Try to parse value

        if value.startswith('"') and value.endswith('"'):

            result[key] = value[1:-1]

        elif value.startswith('['):

            # Try to parse array

            try:

                result[key] = json.loads(value)

            except:

                result[key] = value

        elif value.lower() in ('true', 'false'):

            result[key] = value.lower() == 'true'

        elif value.isdigit():

            result[key] = int(value)

        elif re.match(r'^\d+\.\d+$', value):

            result[key] = float(value)

        else:

            result[key] = value

    if result:

        print(f"⚠️  Partially parsed JSON using regex fallback. Got {len(result)} fields.")

        return result

    # If all else fails, log and return empty dict (workflow should handle this)

    print(f"⚠️  Failed to parse JSON from response")

    print(f"Response length: {len(original_text)} chars")

    print(f"Response preview: {original_text[:500]}...")

    return {}

def parse_experience_years(value: Any) -> Optional[float]:

    """Parse total years of experience from various formats."""

    if value is None:

        return None

    if isinstance(value, (int, float)):

        return float(value)

    if isinstance(value, str):

        # Extract numbers from strings like "1 year", "2-3 years", "1.5 years"

        numbers = re.findall(r'\d+\.?\d*', value)

        if numbers:

            try:

                return float(numbers[0])

            except:

                pass

    return None

def detect_portal(url: str) -> str:

    """Detect the job portal from URL domain."""

    url_lower = url.lower()

    if 'linkedin.com' in url_lower:

        return 'LinkedIn'

    elif 'internshala.com' in url_lower:

        return 'Internshala'

    elif 'indeed.com' in url_lower:

        return 'Indeed'

    elif 'glassdoor.com' in url_lower:

        return 'Glassdoor'

    elif 'monster.com' in url_lower:

        return 'Monster'

    elif 'naukri.com' in url_lower:

        return 'Naukri'

    elif 'timesjobs.com' in url_lower:

        return 'TimesJobs'

    elif 'shine.com' in url_lower:

        return 'Shine'

    elif 'hired.com' in url_lower:

        return 'Hired'

    elif 'angel.co' in url_lower or 'angelist.com' in url_lower:

        return 'AngelList'

    elif 'stackoverflow.com' in url_lower or 'stackoverflowjobs.com' in url_lower:

        return 'Stack Overflow'

    elif 'github.com' in url_lower:

        return 'GitHub Jobs'

    elif 'dice.com' in url_lower:

        return 'Dice'

    elif 'ziprecruiter.com' in url_lower:

        return 'ZipRecruiter'

    elif 'simplyhired.com' in url_lower:

        return 'SimplyHired'

    else:

        # Extract domain name as fallback

        try:

            from urllib.parse import urlparse

            parsed = urlparse(url)

            domain = parsed.netloc.replace('www.', '').split('.')[0]

            return domain.capitalize()

        except:

            return 'Unknown'

def extract_json_ld_job_title(soup: BeautifulSoup) -> Optional[str]:

    """Extract job title from JSON-LD structured data."""

    try:

        for script in soup.find_all('script', type=lambda t: t and 'json' in str(t).lower() and 'ld' in str(t).lower()):

            try:

                json_data = json.loads(script.string or '{}')

                def extract_from_obj(obj):

                    if isinstance(obj, dict):

                        obj_type = obj.get('@type', '')

                        if 'JobPosting' in str(obj_type):

                            # Try different field names

                            for field in ['title', 'jobTitle', 'name', 'jobTitleText']:

                                if field in obj and obj[field]:

                                    return str(obj[field]).strip()

                        # Recursively search nested objects

                        for value in obj.values():

                            result = extract_from_obj(value)

                            if result:

                                return result

                    elif isinstance(obj, list):

                        for item in obj:

                            result = extract_from_obj(item)

                            if result:

                                return result

                    return None

                result = extract_from_obj(json_data)

                if result:

                    return result

            except (json.JSONDecodeError, AttributeError):

                continue

    except Exception:

        pass

    return None

def save_job_applications_background(user_id: str, jobs_to_save: List[Dict[str, Any]]):
    """Background task to save job applications to Firebase (non-blocking)."""
    logger.info(f"[BACKGROUND TASK] Starting job applications save for user {user_id}, {len(jobs_to_save)} jobs")
    try:
        if not user_id:
            logger.error("[BACKGROUND TASK] user_id is empty, cannot save")
            return
        
        if not jobs_to_save or len(jobs_to_save) == 0:
            logger.warning("[BACKGROUND TASK] No jobs to save")
            return
        
        logger.info(f"[BACKGROUND TASK] Importing firebase_service...")
        from firebase_service import get_firebase_service
        logger.info(f"[BACKGROUND TASK] Getting firebase service instance...")
        firebase_service = get_firebase_service()
        logger.info(f"[BACKGROUND TASK] Calling save_job_applications_batch with {len(jobs_to_save)} jobs...")
        saved_doc_ids = firebase_service.save_job_applications_batch(user_id, jobs_to_save)
        logger.info(f"[BACKGROUND TASK] ✓ Save completed: {len(saved_doc_ids)} job applications saved for user {user_id}")
        logger.info(f"[BACKGROUND TASK] Document IDs: {saved_doc_ids}")
    except ImportError as e:
        logger.error(f"[BACKGROUND TASK] Firebase service not available: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[BACKGROUND TASK] Background save failed for job applications: {e}", exc_info=True)
        import traceback
        logger.error(f"[BACKGROUND TASK] Full traceback: {traceback.format_exc()}")

def save_sponsorship_info_background(
    user_id: str,
    request_id: str,
    sponsorship_data: Dict[str, Any],
    job_info: Optional[Dict[str, Any]] = None
):
    """Background task to save sponsorship info to Firebase (non-blocking)."""
    logger.info(f"[BACKGROUND TASK] Starting sponsorship info save for user {user_id}, request {request_id}")
    try:
        if not user_id:
            logger.error("[BACKGROUND TASK] user_id is empty, cannot save sponsorship info")
            return
        
        if not request_id:
            logger.error("[BACKGROUND TASK] request_id is empty, cannot save sponsorship info")
            return
        
        logger.info(f"[BACKGROUND TASK] Importing firebase_service...")
        from firebase_service import get_firebase_service
        logger.info(f"[BACKGROUND TASK] Getting firebase service instance...")
        firebase_service = get_firebase_service()
        logger.info(f"[BACKGROUND TASK] Calling save_sponsorship_info...")
        doc_id = firebase_service.save_sponsorship_info(
            user_id=user_id,
            request_id=request_id,
            sponsorship_data=sponsorship_data,
            job_info=job_info
        )
        logger.info(f"[BACKGROUND TASK] ✓ Save completed: sponsorship info saved with doc_id {doc_id} for user {user_id}")
    except ImportError as e:
        logger.error(f"[BACKGROUND TASK] Firebase service not available: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[BACKGROUND TASK] Background save failed for sponsorship info: {e}", exc_info=True)
        import traceback
        logger.error(f"[BACKGROUND TASK] Full traceback: {traceback.format_exc()}")

@app.post("/api/match-jobs/stream")
async def match_jobs_stream(
    json_body: Optional[str] = Form(default=None),
    pdf_file: Optional[UploadFile] = File(default=None),
    settings: Settings = Depends(get_settings),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Streaming version of match-jobs endpoint.
    Streams scoring progress as Server-Sent Events (SSE) as the LLM generates responses.
    Returns the same data as /api/match-jobs but streams progress updates.
    """
    # CRITICAL: Read file contents into memory BEFORE starting the stream
    # The UploadFile object gets closed after the request handler returns,
    # so we must read it before passing to the generator
    resume_bytes_from_file: Optional[bytes] = None
    if pdf_file is not None:
        try:
            resume_bytes_from_file = await pdf_file.read()
        except Exception as e:
            logger.error(f"Failed to read PDF file: {e}", exc_info=True)
            async def error_stream():
                yield format_sse_event("error", {
                    "message": f"Failed to read PDF file: {str(e)}"
                })
                await asyncio.sleep(0)  # Force flush
            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream"
            )
    
    async def generate_stream(resume_bytes_preloaded: Optional[bytes] = None):
        # Create response file path
        request_id = make_request_id()
        response_file_path = Path(f"responses/response_{request_id}.txt")
        response_file_path.parent.mkdir(parents=True, exist_ok=True)
        response_file = None
        
        # Capture background_tasks from outer scope for Firebase saves
        # background_tasks is accessible via closure from match_jobs_stream function
        
        try:
            from openai import OpenAI
            
            # Open file for writing in unbuffered mode for real-time streaming
            import sys
            response_file = open(response_file_path, "w", encoding="utf-8", buffering=1)  # Line buffered
            logger.info(f"Writing response to file: {response_file_path}")
            response_file.write(f"=== Job Match Analysis Response ===\n")
            response_file.write(f"Request ID: {request_id}\n")
            response_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            response_file.write(f"{'='*60}\n\n")
            response_file.flush()
            os.fsync(response_file.fileno())  # Force OS-level flush for real-time writing
            
            start_time = time.time()
            openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            
            if not openai_key:
                yield format_sse_event("error", {
                    "message": "OpenAI API key not configured"
                })
                await asyncio.sleep(0)  # Force flush
                return
            
            client = OpenAI(api_key=openai_key)
            
            # Parse request (reuse logic from match_jobs)
            # This is a simplified version - in production, extract the full parsing logic
            data: Optional[MatchJobsRequest] = None
            legacy_data: Optional[MatchJobsJsonRequest] = None
            new_format_jobs: Optional[Dict[str, Any]] = None
            jobs_string: Optional[str] = None
            user_id: Optional[str] = None
            
            if json_body:
                clean_json = json_body.strip()
                if clean_json.startswith('"') and clean_json.endswith('"'):
                    clean_json = clean_json[1:-1].replace('\\"', '"')
                payload = json.loads(clean_json)
                
                if "jobs" in payload and isinstance(payload["jobs"], dict):
                    new_format_jobs = payload["jobs"]
                    user_id = payload.get("user_id")
                elif "jobs" in payload and isinstance(payload["jobs"], str):
                    jobs_string = payload["jobs"]
                    user_id = payload.get("user_id")
                elif "resume" in payload and "jobs" in payload:
                    data = MatchJobsRequest(**payload)
                else:
                    try:
                        legacy_data = MatchJobsJsonRequest(**payload)
                    except:
                        yield format_sse_event("error", {
                            "message": "Invalid request format"
                        })
                        await asyncio.sleep(0)  # Force flush
                        return
            
            # Count jobs for progress tracking (after parsing)
            jobs_count = 1 if new_format_jobs else 0
            
            # Event 1: Start - send as status for frontend compatibility
            logger.info(f"Yielding start event: request_id={request_id}, jobs_count={jobs_count}")
            yield format_sse_event("status", {
                "message": f"Starting analysis... ({jobs_count} job(s) to analyze)"
            })
            await asyncio.sleep(0)  # Force flush
            logger.info("Start event yielded and flushed")
            
            # Get resume - use preloaded bytes if available
            resume_bytes: Optional[bytes] = None
            if data and data.resume and data.resume.content:
                resume_bytes = decode_base64_pdf(data.resume.content)
            elif legacy_data and legacy_data.pdf:
                resume_bytes = decode_base64_pdf(legacy_data.pdf)
            elif resume_bytes_preloaded is not None:
                resume_bytes = resume_bytes_preloaded
            
            if not resume_bytes:
                yield format_sse_event("error", {
                    "message": "Missing resume PDF"
                })
                await asyncio.sleep(0)  # Force flush
                return
            
            # PATCH 1: Hash resume bytes BEFORE text extraction (CRITICAL FIX)
            # This ensures same PDF file always produces same hash, regardless of OCR variations
            resume_hash = hashlib.sha256(resume_bytes).hexdigest()
            logger.info(f"Resume hash (from bytes): {resume_hash[:16]}...")
            
            # Event 2: Resume parsing progress
            yield format_sse_event("status", {
                "message": "Extracting text from PDF..."
            })
            await asyncio.sleep(0)  # Force flush
            
            # Timing: Start PDF extraction
            pdf_extraction_start = time.perf_counter()
            
            # Parse resume with streaming LLM response
            logger.info("Starting resume parsing...")
            resume_text = extract_text_from_pdf_bytes(resume_bytes)
            logger.info(f"Resume text extracted, length: {len(resume_text)}")
            
            # Timing: End PDF extraction
            pdf_extraction_duration = time.perf_counter() - pdf_extraction_start
            yield format_sse_event("timing", {
                "step": "pdf_extraction",
                "seconds": round(pdf_extraction_duration, 3)
            })
            await asyncio.sleep(0)  # Force flush
            
            # Save OCR-extracted text to separate file for debugging
            ocr_text_file_path = Path(f"responses/resume_ocr_{request_id}.txt")
            ocr_text_file_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(ocr_text_file_path, "w", encoding="utf-8") as ocr_file:
                    ocr_file.write("=== OCR-EXTRACTED RESUME TEXT ===\n")
                    ocr_file.write(f"Request ID: {request_id}\n")
                    ocr_file.write(f"Extracted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    ocr_file.write(f"Text Length: {len(resume_text)} characters\n")
                    ocr_file.write(f"Resume Hash: {resume_hash}\n")
                    ocr_file.write(f"{'='*60}\n\n")
                    ocr_file.write(resume_text)
                logger.info(f"OCR-extracted resume text saved to: {ocr_text_file_path}")
            except Exception as e:
                logger.warning(f"Failed to save OCR text to file: {e}")
            
            # Use task-specific model selection for resume parsing
            from model_config import get_model_for_task
            model_name = get_model_for_task("resume_parsing", json_mode=True).model_name
            logger.info(f"Starting streaming LLM parsing with model: {model_name} (task: resume_parsing)")
            
            # Initialize timing variable (will be set in non-cached path)
            resume_parsing_start = None
            
            if resume_hash in RESUME_PARSING_CACHE:
                logger.info(f"✅ Resume found in cache, using cached result (hash: {resume_hash[:16]}...)")
                logger.info(f"Cache validation: Resume SHA256={resume_hash[:16]}... matches cached entry")
                yield format_sse_event("status", {
                    "message": "Resume found in cache, using cached parsing result..."
                })
                await asyncio.sleep(0)  # Force flush
                
                # Timing: Cache hit
                yield format_sse_event("timing", {
                    "step": "resume_parsing",
                    "seconds": 0.001,
                    "cached": True
                })
                await asyncio.sleep(0)  # Force flush
                
                # PATCH 2: Use deepcopy instead of shallow copy (prevents cache corruption)
                import copy
                candidate_profile = copy.deepcopy(RESUME_PARSING_CACHE[resume_hash])
                
                # CRITICAL: Ensure all datetime objects are converted to strings (defensive check)
                # This prevents JSON serialization errors when candidate_profile is used later
                def convert_datetime_to_str(obj):
                    """Recursively convert datetime objects to ISO format strings"""
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, dict):
                        return {k: convert_datetime_to_str(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_datetime_to_str(item) for item in obj]
                    else:
                        return obj
                
                candidate_profile = convert_datetime_to_str(candidate_profile)
                
                # Log cached profile summary for validation
                logger.info(f"Cached profile summary: {len(candidate_profile.get('skills', []))} skills, "
                          f"{len(candidate_profile.get('experience_entries', []))} experience entries, "
                          f"{len(candidate_profile.get('projects', []))} projects")
                logger.info(f"CRITICAL: Using cached profile - same resume will always return same profile")
                
                # Send cached resume_parsed event
                candidate_profile_response = {
                    "name": candidate_profile.get("name"),
                    "email": candidate_profile.get("email"),
                    "phone": candidate_profile.get("phone"),
                    "skills": candidate_profile.get("skills", []) if isinstance(candidate_profile.get("skills"), list) else [],
                    "experience_summary": candidate_profile.get("experience_summary"),
                    "education": candidate_profile.get("education", []),
                    "certifications": candidate_profile.get("certifications", []),
                    "interests": candidate_profile.get("interests", []),
                    "raw_text_excerpt": candidate_profile.get("raw_text_excerpt")
                }
                yield format_sse_event("resume_parsed", {
                    "candidate_profile": candidate_profile_response
                })
                await asyncio.sleep(0)  # Force flush
                logger.info(f"✅ Cached resume_parsed event sent with {len(candidate_profile_response['skills'])} skills")
            else:
                # PATCH 3: Hard guard - log when resume is parsed for the first time
                logger.info(f"Resume not in cache, parsing with LLM for the first time (hash: {resume_hash[:16]}...)")
                
                # Timing: Start resume parsing
                resume_parsing_start = time.perf_counter()
            
            # Stream LLM response for resume parsing
            try:
                from openai import OpenAI
                import json as json_lib
                
                client = OpenAI(api_key=openai_key)
                
                resume_prompt = f"""Extract ALL information from this resume text without assumptions about format.

RESUME TEXT:
{resume_text}

Return JSON with:
- name: Full name or null
- email: Email address or null
- phone: Phone number or null
- skills: List of ALL skills/technologies/tools mentioned ANYWHERE in the resume (CRITICAL: must include skills from ALL sections below)
- experience_summary: 3-4 sentence professional summary (education, experience, key achievements, skills)
- experience_entries: List of {{role, company, date_range, type, description}} found anywhere
  * description: Brief description of work/projects done in this role (extract from bullet points, project descriptions)
- projects: List of {{name, description, technologies}} for standalone projects/hackathons if explicitly mentioned
- education: List of {{school, degree, dates}} found anywhere
- certifications: List of ALL certifications/licenses found
- interests: List of interests/hobbies if explicitly mentioned, else empty array

CRITICAL SKILL EXTRACTION RULES (MUST FOLLOW ALL):
1. **Skills Section**: Extract EVERY skill/technology/tool listed in any "Skills", "Technical Skills", "Tech Stack", "Technologies", "Tools" section
2. **Experience Entries**: Extract ALL technologies/tools/frameworks mentioned in EACH experience entry's description:
   - Read EVERY bullet point in experience descriptions
   - Extract technologies used (e.g., "Built REST API using Flask" → extract "Flask", "REST API")
   - Extract frameworks (e.g., "Developed ML model with TensorFlow" → extract "TensorFlow", "Machine Learning")
   - Extract tools/platforms (e.g., "Deployed on AWS using Docker" → extract "AWS", "Docker")
   - Extract languages (e.g., "Implemented in Python and Java" → extract "Python", "Java")
3. **Projects Section**: Extract ALL technologies/tools mentioned in project descriptions:
   - Read EVERY project description
   - Extract technologies, frameworks, libraries, tools used
   - Extract programming languages
   - Extract platforms (cloud, databases, etc.)
4. **Education/Coursework**: Extract technologies mentioned in coursework or education descriptions
5. **Certifications**: Extract technologies/tools mentioned in certification names or descriptions
6. **Any Other Section**: Extract skills from achievements, hackathons, competitions, publications, etc.

SKILL AGGREGATION RULES:
- The "skills" array MUST contain ALL skills found in ALL sections above
- Normalize skill variations (e.g., "Python 3", "Python3", "Python 3.9" → "Python")
- Deduplicate skills (same skill mentioned multiple times → include once)
- Include both technical AND non-technical skills (soft skills, tools, methodologies)
- DO NOT include project names, product names, or system names as skills
  * Examples of what NOT to include: "Deepfake Detection", "Voice Sales Assistant", "Classroom Tutor" (these are project names, NOT skills)
  * Only include actual technologies, tools, frameworks, languages, platforms, methodologies
- DO NOT skip skills because they seem minor or common
- Better to include too many skills than miss important ones
- CRITICAL: Be consistent - same resume text must always produce the same skills list

OTHER RULES:
- For experience_entries: Extract role, company, date_range, type, AND description (what they did, technologies used, projects built)
- For projects: Extract project names, descriptions, and technologies used (if there's a separate Projects section)
- For education: Include school name, degree, and dates (including "Expected" or "Pursuing" if present)
- CRITICAL: Extract project descriptions from experience entries - look for bullet points describing work done, technologies used, systems built

Return ONLY JSON, no explanation."""
                
                # Parse resume with LLM (with retry logic and optimized for speed)
                logger.info("Calling LLM for resume parsing...")
                
                def call_llm_sync_with_retry(max_retries=3, initial_delay=1.0):
                    """Call OpenAI API synchronously with retry logic"""
                    for attempt in range(max_retries):
                        try:
                            logger.info(f"Making OpenAI API call (attempt {attempt + 1}/{max_retries})...")
                            # Use task-specific model configuration
                            from model_config import get_model_for_task, supports_json_mode, assert_model_supports_json
                            resume_model = get_model_for_task("resume_parsing", json_mode=True)
                            use_json_mode = supports_json_mode(resume_model.model_name)
                            
                            # Runtime assertion: Ensure JSON mode is supported
                            if use_json_mode:
                                assert_model_supports_json(resume_model.model_name, "resume_parsing")
                            
                            response = client.chat.completions.create(
                                model=resume_model.model_name,
                                messages=[{"role": "user", "content": resume_prompt}],
                                temperature=0,  # MUST be 0 for deterministic output
                                seed=42,  # CRITICAL: Fixed seed for consistency - same resume must always produce same profile
                                response_format={"type": "json_object"} if use_json_mode else None,
                                timeout=30.0  # 30 second timeout per request
                            )
                            
                            # Runtime assertion: Validate JSON response
                            if use_json_mode:
                                response_text = response.choices[0].message.content.strip()
                                try:
                                    json_lib.loads(response_text)
                                except json_lib.JSONDecodeError as e:
                                    logger.error(f"CRITICAL: Resume parsing returned invalid JSON: {e}")
                                    raise ValueError(f"Resume parsing failed: Invalid JSON response from model {resume_model.model_name}")
                            logger.info("OpenAI API call completed")
                            if not response or not response.choices or len(response.choices) == 0:
                                logger.error("Empty response from OpenAI API")
                                raise Exception("Empty response from OpenAI API")
                            if not response.choices[0].message or not response.choices[0].message.content:
                                logger.error("No content in OpenAI response")
                                raise Exception("No content in OpenAI response")
                            response_text = response.choices[0].message.content.strip()
                            logger.info(f"Response received, length: {len(response_text)}")
                            if not response_text:
                                logger.error("Empty response text after stripping")
                                raise Exception("Empty response text")
                            return response_text
                        except Exception as e:
                            if attempt < max_retries - 1:
                                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                                logger.warning(f"OpenAI API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                                import time
                                time.sleep(delay)
                            else:
                                logger.error(f"OpenAI API call failed after {max_retries} attempts: {e}", exc_info=True)
                                raise
                
                # Run LLM call in thread with shorter timeout
                logger.info("Starting LLM call in background thread...")
                full_response = None
                try:
                    # Use asyncio.to_thread if available (Python 3.9+), otherwise use run_in_executor
                    if hasattr(asyncio, 'to_thread'):
                        logger.info("Using asyncio.to_thread...")
                        full_response = await asyncio.wait_for(
                            asyncio.to_thread(call_llm_sync_with_retry),
                            timeout=60.0  # 1 minute timeout (reduced from 2 minutes)
                        )
                    else:
                        logger.info("Using run_in_executor (Python < 3.9)...")
                        loop = asyncio.get_event_loop()
                        full_response = await asyncio.wait_for(
                            loop.run_in_executor(None, call_llm_sync_with_retry),
                            timeout=60.0  # 1 minute timeout
                        )
                    logger.info(f"✅ LLM response received, total length: {len(full_response)}")
                    if not full_response or len(full_response) == 0:
                        logger.error("Empty response from LLM")
                        raise Exception("Empty response from LLM")
                except asyncio.TimeoutError:
                    logger.error("LLM call timed out after 60 seconds")
                    raise Exception("Resume parsing timed out")
                except Exception as llm_error:
                    logger.error(f"Error getting LLM response: {llm_error}", exc_info=True)
                    raise llm_error
                
                # Parse the JSON response IMMEDIATELY (before any file operations)
                logger.info(f"✅ Starting JSON parsing, response length: {len(full_response)}")
                await asyncio.sleep(0)  # Yield before parsing
                
                from agents import extract_json_from_response
                candidate_profile = extract_json_from_response(full_response)
                logger.info(f"✅ JSON parsing complete, candidate_profile type: {type(candidate_profile)}")
                
                # Timing: End resume parsing (only if we started timing, i.e., non-cached path)
                if resume_parsing_start is not None:
                    resume_parsing_duration = time.perf_counter() - resume_parsing_start
                    yield format_sse_event("timing", {
                        "step": "resume_parsing",
                        "seconds": round(resume_parsing_duration, 3),
                        "cached": False
                    })
                    await asyncio.sleep(0)  # Force flush
                
                # Validate and fix candidate_profile with comprehensive fallbacks
                if not candidate_profile or not isinstance(candidate_profile, dict):
                    logger.warning("Failed to extract JSON from resume parsing response, using fallback")
                    candidate_profile = {
                        "name": "Unknown Candidate",
                        "email": None,
                        "phone": None,
                        "skills": [],
                        "experience_summary": resume_text[:500] if resume_text else "",
                        "total_years_experience": 0.0,  # For internal scoring only
                        "education": [],
                        "certifications": [],
                        "interests": []
                    }
                else:
                    # Validate and ensure all required fields exist with correct types
                    if "skills" not in candidate_profile or not isinstance(candidate_profile.get("skills"), list):
                        logger.warning("Skills field missing or invalid, initializing empty list")
                        candidate_profile["skills"] = []
                    if "experience_summary" not in candidate_profile or not isinstance(candidate_profile.get("experience_summary"), str):
                        logger.warning("experience_summary missing or invalid, using resume text excerpt")
                        candidate_profile["experience_summary"] = resume_text[:500] if resume_text else ""
                    else:
                        # Normalize experience_summary: normalize whitespace for consistency
                        candidate_profile["experience_summary"] = " ".join(candidate_profile["experience_summary"].split())
                        
                        # Process experience entries and calculate breakdown
                        experience_entries = candidate_profile.get("experience_entries", [])
                        if not isinstance(experience_entries, list):
                            experience_entries = []
                        
                        # Process and classify each entry
                        processed_entries = []
                        for entry in experience_entries:
                            if not isinstance(entry, dict):
                                continue
                            
                            # Ensure type classification
                            entry_text = f"{entry.get('role', '')} {entry.get('company', '')} {entry.get('date_range', '')}"
                            entry_type = classify_experience_type(entry_text)
                            
                            # Parse date range
                            date_range = entry.get("date_range", "")
                            # Normalize date_range for consistency (capitalization of months)
                            date_range_normalized = date_range
                            if isinstance(date_range, str):
                                months = ["January", "February", "March", "April", "May", "June", 
                                         "July", "August", "September", "October", "November", "December"]
                                for month in months:
                                    date_range_normalized = re.sub(rf'\b{month.upper()}\b', month, date_range_normalized, flags=re.IGNORECASE)
                            
                            start_date, end_date = parse_date_range(date_range_normalized) if date_range_normalized else (None, None)
                            
                            # CRITICAL: Convert datetime objects to ISO format strings for JSON serialization
                            # CRITICAL: For "Present" dates, use None instead of timestamp to avoid drift
                            start_date_str = start_date.isoformat() if start_date and isinstance(start_date, datetime) else None
                            if end_date and isinstance(end_date, datetime):
                                # Check if this is the "Present" marker date (2099-12-31)
                                if end_date.year == 2099 and end_date.month == 12 and end_date.day == 31:
                                    end_date_str = None  # Use None for "Present" to avoid timestamp drift
                                else:
                                    end_date_str = end_date.isoformat()
                            else:
                                end_date_str = None
                            
                            processed_entry = {
                                "role": entry.get("role", ""),
                                "company": entry.get("company", ""),
                                "date_range": date_range_normalized,
                                "type": entry_type,
                                "start_date": start_date_str,  # Store as string, not datetime object
                                "end_date": end_date_str  # Store as string or None for "Present"
                            }
                            # Preserve description if it exists (for requirement evaluation)
                            if "description" in entry:
                                processed_entry["description"] = entry.get("description", "")
                            processed_entries.append(processed_entry)
                        
                        # Update candidate_profile with processed entries (preserving descriptions)
                        candidate_profile["experience_entries"] = processed_entries
                        
                        # Calculate experience breakdown
                        experience_breakdown = calculate_experience_breakdown(processed_entries)
                        candidate_profile["experience_breakdown"] = experience_breakdown
                        
                        # Calculate total_years_experience from breakdown for internal scoring (but won't be in response)
                        total_duration_str = experience_breakdown.get("total", "0 months")
                        candidate_profile["total_years_experience"] = parse_duration_string(total_duration_str)
                        
                        logger.info(f"Calculated experience breakdown: {experience_breakdown}")
                        logger.info(f"Total years (for internal scoring): {candidate_profile['total_years_experience']}")
                    if "education" not in candidate_profile or not isinstance(candidate_profile.get("education"), list):
                        candidate_profile["education"] = []
                    else:
                        # Normalize education dates for consistency
                        for edu in candidate_profile["education"]:
                            if isinstance(edu, dict) and "dates" in edu:
                                dates = edu.get("dates", "")
                                if isinstance(dates, str):
                                    # Normalize date format: ensure consistent capitalization
                                    dates_normalized = dates.strip()
                                    # Normalize "Expected" prefix: ensure consistent format
                                    if "expected" in dates_normalized.lower():
                                        # Extract the date part
                                        date_part = re.sub(r'expected\s*', '', dates_normalized, flags=re.IGNORECASE).strip()
                                        dates_normalized = f"Expected {date_part}"
                                    # Normalize month capitalization
                                    months = ["January", "February", "March", "April", "May", "June", 
                                             "July", "August", "September", "October", "November", "December"]
                                    for month in months:
                                        dates_normalized = re.sub(rf'\b{month.upper()}\b', month, dates_normalized, flags=re.IGNORECASE)
                                    edu["dates"] = dates_normalized
                    if "certifications" not in candidate_profile or not isinstance(candidate_profile.get("certifications"), list):
                        candidate_profile["certifications"] = []
                    if "interests" not in candidate_profile or not isinstance(candidate_profile.get("interests"), list):
                        candidate_profile["interests"] = []
                    if "experience_entries" not in candidate_profile or not isinstance(candidate_profile.get("experience_entries"), list):
                        candidate_profile["experience_entries"] = []
                    if "projects" not in candidate_profile or not isinstance(candidate_profile.get("projects"), list):
                        candidate_profile["projects"] = []
                    if "name" not in candidate_profile or not isinstance(candidate_profile.get("name"), str):
                        candidate_profile["name"] = candidate_profile.get("name", "Unknown Candidate") or "Unknown Candidate"
                    
                    # CRITICAL: The LLM should already extract all skills from all sections
                    # We only do minimal post-processing to ensure consistency
                    # DO NOT add skills from keyword matching - this causes non-determinism
                    # The LLM with seed=42 should extract the same skills every time
                    
                    # Only normalize and deduplicate skills that the LLM extracted
                    skills = candidate_profile.get("skills", [])
                    if not isinstance(skills, list):
                        skills = []
                    
                    # Normalize skills: remove duplicates, normalize variations, sort for determinism
                    normalized_skills = []
                    seen_lower = set()
                    # Skill normalization map (common variations)
                    skill_normalizations = {
                        "artificial intelligence": "AI",
                        "ai": "AI",
                        "machine learning": "Machine Learning",
                        "deep learning": "Deep Learning",
                        "natural language processing": "Natural Language Processing",
                        "nlp": "Natural Language Processing",
                        "computer vision": "Computer Vision",
                        "data science": "Data Science"
                    }
                    
                    # Skills to exclude (project names, product names, not actual skills)
                    excluded_skills = {
                        "deepfake detection",  # Project name, not a skill
                        "voice assistant technology",  # Too generic/product name
                        "voice sales assistant",  # Project name
                        "classroom tutor",  # Project/system name
                        "automation pipelines",  # Too generic (should be "Automation" or specific tool)
                        "crm integration",  # Too generic (should be specific CRM tool)
                        "session store"  # Too generic (should be specific technology)
                    }
                    
                    for skill in skills:
                        if isinstance(skill, str):
                            skill_clean = skill.strip()
                            if skill_clean:
                                skill_lower = skill_clean.lower()
                                
                                # Exclude project names and generic terms
                                if skill_lower in excluded_skills:
                                    logger.debug(f"Excluding project name/generic term from skills: {skill_clean}")
                                    continue
                                
                                # Normalize common variations
                                if skill_lower in skill_normalizations:
                                    skill_clean = skill_normalizations[skill_lower]
                                
                                # Only add if not already seen (case-insensitive)
                                if skill_clean.lower() not in seen_lower:
                                    normalized_skills.append(skill_clean)
                                    seen_lower.add(skill_clean.lower())
                    
                    # Sort for deterministic output
                    candidate_profile["skills"] = sorted(normalized_skills)
                    logger.info(f"✅ Normalized {len(candidate_profile['skills'])} skills from LLM extraction (deterministic, seed=42)")
                    
                    # CRITICAL: Do NOT add skills from keyword matching in descriptions
                    # This causes non-determinism because:
                    # 1. Project names might be extracted as skills ("Deepfake Detection", "Voice Sales Assistant")
                    # 2. Different keyword matches might occur based on text variations
                    # 3. The LLM with seed=42 should handle all skill extraction consistently
                
                logger.info(f"✅ Resume parsing completed successfully")
                
                # CRITICAL: Ensure all datetime objects are converted to strings before caching
                # This prevents JSON serialization errors when candidate_profile is used later
                def convert_datetime_to_str(obj):
                    """Recursively convert datetime objects to ISO format strings"""
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, dict):
                        return {k: convert_datetime_to_str(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_datetime_to_str(item) for item in obj]
                    else:
                        return obj
                
                # Convert any remaining datetime objects to strings
                candidate_profile = convert_datetime_to_str(candidate_profile)
                
                # Save to cache after successful parsing (use deepcopy to prevent future mutations)
                import copy
                RESUME_PARSING_CACHE[resume_hash] = copy.deepcopy(candidate_profile)
                logger.info(f"✅ Resume parsing result saved to cache (hash: {resume_hash[:16]}...)")
                logger.info(f"Cache entry summary: {len(candidate_profile.get('skills', []))} skills, "
                          f"{len(candidate_profile.get('experience_entries', []))} experience entries, "
                          f"{len(candidate_profile.get('projects', []))} projects")
                logger.info(f"CRITICAL: Same resume (SHA256={resume_hash[:16]}...) will now always return this cached profile")
                
                # Ensure candidate_profile is a dict BEFORE sending event
                if not isinstance(candidate_profile, dict):
                    logger.warning(f"candidate_profile is not a dict, type: {type(candidate_profile)}, converting...")
                    if hasattr(candidate_profile, 'dict'):
                        candidate_profile = candidate_profile.dict()
                    elif hasattr(candidate_profile, '__dict__'):
                        candidate_profile = candidate_profile.__dict__
                    else:
                        candidate_profile = {"name": "Unknown", "skills": [], "total_years_experience": 0}
                        logger.error("Could not convert candidate_profile to dict, using defaults")
                
                # SEND resume_parsed event IMMEDIATELY (before file operations)
                logger.info(f"Yielding resume_parsed event IMMEDIATELY with full candidate profile: name={candidate_profile.get('name')}")
                # Remove total_years_experience and other float year fields from response
                candidate_profile_response = {
                    "name": candidate_profile.get("name"),
                    "email": candidate_profile.get("email"),
                    "phone": candidate_profile.get("phone"),
                    "skills": candidate_profile.get("skills", []) if isinstance(candidate_profile.get("skills"), list) else [],
                    "experience_summary": candidate_profile.get("experience_summary"),
                    "education": candidate_profile.get("education", []),
                    "certifications": candidate_profile.get("certifications", []),
                    "interests": candidate_profile.get("interests", []),
                    "raw_text_excerpt": candidate_profile.get("raw_text_excerpt")
                }
                yield format_sse_event("resume_parsed", {
                    "candidate_profile": candidate_profile_response
                })
                await asyncio.sleep(0)  # Force flush - CRITICAL for SSE
                logger.info("✅ resume_parsed event sent and flushed - continuing to file writes")
                
                # NOW do file writes in background (non-blocking)
                async def write_resume_parsing_to_file():
                    """Write resume parsing results to file asynchronously"""
                    try:
                        if response_file:
                            logger.info("Writing LLM response and parsing result to file...")
                            response_file.write(f"=== RAW LLM RESPONSE (Resume Parsing) ===\n")
                            response_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            response_file.write(f"Length: {len(full_response)} characters\n")
                            response_file.write(f"{'='*60}\n\n")
                            response_file.write(full_response)
                            response_file.write(f"\n\n")
                            response_file.write(f"=== RESUME PARSING RESULT ===\n")
                            response_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            response_file.write(f"{'='*60}\n\n")
                            response_file.write(json.dumps(candidate_profile, indent=2, ensure_ascii=False))
                            response_file.write(f"\n\n")
                            response_file.flush()
                            os.fsync(response_file.fileno())
                            logger.info("✅ Resume parsing results written to file")
                    except Exception as file_error:
                        logger.error(f"Error writing resume parsing to file: {file_error}", exc_info=True)
                
                # Start file write in background (don't await - non-blocking)
                asyncio.create_task(write_resume_parsing_to_file())
                logger.info("✅ File write task started in background")
                
            except Exception as parse_error:
                logger.error(f"Error in resume parsing: {parse_error}", exc_info=True)
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                yield format_sse_event("error", {
                    "message": f"Resume parsing failed: {str(parse_error)}"
                })
                await asyncio.sleep(0)  # Force flush
                return
            
            # Extract jobs (simplified - handle new_format_jobs case)
            logger.info(f"Starting job extraction. new_format_jobs={new_format_jobs is not None}, type={type(new_format_jobs)}")
            if new_format_jobs:
                logger.info(f"new_format_jobs keys: {list(new_format_jobs.keys()) if isinstance(new_format_jobs, dict) else 'not a dict'}")
            
            jobs = []
            try:
                if new_format_jobs:
                    raw_job_title = new_format_jobs.get("jobtitle", "")
                    job_link = new_format_jobs.get("joblink", "")
                    job_data = new_format_jobs.get("jobdata", "")
                    
                    # Convert HTML job data to Markdown for better LLM processing
                    if job_data and isinstance(job_data, str):
                        from utils import convert_html_to_markdown
                        logger.info(f"Converting HTML job data to Markdown (original length: {len(job_data)} chars)")
                        job_data = convert_html_to_markdown(job_data)
                        logger.info(f"HTML to Markdown conversion complete (markdown length: {len(job_data)} chars)")
                    
                    logger.info(f"Job data extracted - title: {raw_job_title[:50] if raw_job_title else 'None'}, data length: {len(job_data) if job_data else 0}")
                    
                    from job_extractor import extract_company_and_title_from_raw_data
                    # Extract job info (this might take a moment)
                    logger.info("Calling extract_company_and_title_from_raw_data...")
                    try:
                        # Run extraction in thread with progress monitoring
                        extraction_start = time.time()
                        last_progress = extraction_start
                        
                        def extract_job_sync():
                            """Extract job info synchronously"""
                            return extract_company_and_title_from_raw_data(
                                job_data,
                                openai_key,
                                "gpt-4o-mini"
                            )
                        
                        # Use asyncio.to_thread if available, otherwise use run_in_executor
                        if hasattr(asyncio, 'to_thread'):
                            extracted_info = await asyncio.wait_for(
                                asyncio.to_thread(extract_job_sync),
                                timeout=60.0  # 1 minute timeout
                            )
                        else:
                            loop = asyncio.get_event_loop()
                            extracted_info = await asyncio.wait_for(
                                loop.run_in_executor(None, extract_job_sync),
                                timeout=60.0  # 1 minute timeout
                            )
                        logger.info(f"Job extraction complete: company={extracted_info.get('company_name')}, title={extracted_info.get('job_title')}")
                        
                        extracted_company = extracted_info.get("company_name")
                        extracted_title = extracted_info.get("job_title")
                        
                        # Validate and clean job title with multiple fallbacks
                        final_job_title = None
                        if extracted_title:
                            cleaned_title = clean_job_title(extracted_title)
                            if cleaned_title and len(cleaned_title) >= 3:
                                final_job_title = cleaned_title
                        
                        if not final_job_title and raw_job_title:
                            cleaned_raw_title = clean_job_title(raw_job_title.strip())
                            if cleaned_raw_title and len(cleaned_raw_title) >= 3:
                                final_job_title = cleaned_raw_title
                            elif raw_job_title.strip() and len(raw_job_title.strip()) >= 3:
                                final_job_title = raw_job_title.strip()
                        
                        if not final_job_title:
                            final_job_title = "Job title not available in posting"
                        
                        # Validate and clean company name with multiple fallbacks
                        final_company = None
                        if extracted_company:
                            cleaned_company = clean_company_name(extracted_company)
                            # Additional validation: reject if it's a sentence fragment
                            if cleaned_company and len(cleaned_company) >= 2:
                                # Double-check it's not a sentence fragment using is_valid_company_name
                                if is_valid_company_name(cleaned_company):
                                    final_company = cleaned_company
                                else:
                                    logger.warning(f"Rejected extracted company name (invalid/sentence fragment): {cleaned_company[:50]}")
                                    final_company = None
                        
                        if not final_company:
                            # Try to extract from job_data if available
                            if job_data:
                                try:
                                    from sponsorship_checker import extract_company_name as extract_company_fallback
                                    fallback_company = extract_company_fallback(job_data[:2000])
                                    if fallback_company:
                                        cleaned_fallback = clean_company_name(fallback_company)
                                        if cleaned_fallback and len(cleaned_fallback) >= 2:
                                            # Validate fallback company name too
                                            if is_valid_company_name(cleaned_fallback):
                                                final_company = cleaned_fallback
                                except Exception as e:
                                    logger.debug(f"Fallback company extraction failed: {e}")
                        
                        if not final_company:
                            final_company = "Company name not available in posting"
                        
                        # Validate description
                        final_description = job_data if job_data and isinstance(job_data, str) else ""
                        if not final_description or len(final_description.strip()) == 0:
                            final_description = "Job description not available"
                        
                        # Clean job description with LLM (for Firestore storage, internal use only)
                        cleaned_job_description = None
                        if final_description and final_description != "Job description not available" and len(final_description.strip()) >= 50:
                            try:
                                from job_extractor import clean_job_description_with_llm
                                logger.info(f"Cleaning job description with LLM (raw length: {len(final_description)} chars)")
                                # Run cleaning in thread to avoid blocking
                                def clean_description_sync():
                                    # Use task-specific model for job description cleaning
                                    from model_config import get_model_for_task
                                    cleaning_model = get_model_for_task("job_extraction", json_mode=False)
                                    return clean_job_description_with_llm(
                                        final_description,
                                        openai_key,
                                        cleaning_model.model_name
                                    )
                                
                                if hasattr(asyncio, 'to_thread'):
                                    cleaned_job_description = await asyncio.wait_for(
                                        asyncio.to_thread(clean_description_sync),
                                        timeout=30.0  # 30 second timeout
                                    )
                                else:
                                    loop = asyncio.get_event_loop()
                                    cleaned_job_description = await asyncio.wait_for(
                                        loop.run_in_executor(None, clean_description_sync),
                                        timeout=30.0
                                    )
                                logger.info(f"✓ Job description cleaned (cleaned length: {len(cleaned_job_description)} chars)")
                            except asyncio.TimeoutError:
                                logger.warning("Job description cleaning timed out, using raw description")
                                cleaned_job_description = None
                            except Exception as e:
                                logger.warning(f"Failed to clean job description: {e}, using raw description")
                                cleaned_job_description = None
                        
                        # Validate URL
                        final_url = job_link if job_link and isinstance(job_link, str) and job_link.startswith(("http://", "https://")) else "https://example.com"
                        
                        logger.info(f"Final job extraction - Title: {final_job_title[:50]}, Company: {final_company[:50]}, Description length: {len(final_description)}")
                        
                        job = JobPosting(
                            url=final_url,
                            job_title=final_job_title,
                            company=final_company,
                            description=final_description,  # Raw scraped content (for scoring)
                            job_description=cleaned_job_description,  # Cleaned description (for Firestore, internal use only)
                            skills_needed=[],
                            experience_level=None,
                            salary=None
                        )
                        jobs = [job]
                        logger.info(f"Job created: {final_job_title} at {final_company}")
                        
                        # Write job extraction result to file
                        if response_file:
                            response_file.write(f"=== JOB EXTRACTION RESULT ===\n")
                            response_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            response_file.write(f"{'='*60}\n\n")
                            job_extraction_data = {
                                "job_title": final_job_title,
                                "company": final_company,
                                "url": job_link if job_link else "https://example.com",
                                "description_length": len(job_data) if job_data else 0
                            }
                            response_file.write(json.dumps(job_extraction_data, indent=2, ensure_ascii=False))
                            response_file.write(f"\n\n")
                            response_file.flush()
                            logger.info("Job extraction result written to file")
                        
                        # Event 4: Job details extracted - send as job_start for frontend
                        yield format_sse_event("job_start", {
                            "job_index": 1,
                            "total_jobs": 1,
                            "job_title": final_job_title,
                            "company": final_company
                        })
                        await asyncio.sleep(0)  # Force flush
                    except Exception as extract_error:
                        logger.error(f"Error in extract_company_and_title_from_raw_data: {extract_error}", exc_info=True)
                        yield format_sse_event("error", {
                            "message": f"Job extraction failed: {str(extract_error)}"
                        })
                        await asyncio.sleep(0)  # Force flush
                        return
                else:
                    logger.warning("new_format_jobs is None or empty, no jobs to process")
            except Exception as job_extract_error:
                logger.error(f"Error in job extraction block: {job_extract_error}", exc_info=True)
                yield format_sse_event("error", {
                    "message": f"Job extraction error: {str(job_extract_error)}"
                })
                await asyncio.sleep(0)  # Force flush
                return
            
            if not jobs:
                logger.warning("No jobs found after extraction")
                yield format_sse_event("error", {
                    "message": "No jobs found"
                })
                await asyncio.sleep(0)  # Force flush
                return
            
            logger.info(f"Successfully extracted {len(jobs)} job(s), starting scoring...")
            
            logger.info(f"Job extraction complete, {len(jobs)} job(s) found. Starting scoring...")
            
            # Stream scoring for each job
            scored_jobs = []
            for idx, job in enumerate(jobs, 1):
                logger.info(f"Scoring job {idx}: {job.job_title} at {job.company}")
                
                # Timing: Start job scoring
                job_scoring_start = time.perf_counter()
                
                # Detect role type: intern, senior, or regular full-time
                job_title_lower = (job.job_title or "").lower()
                job_desc_lower = (job.description or "").lower()[:1000]  # Check first 1000 chars for speed
                
                is_intern_role = (
                    "intern" in job_title_lower or 
                    "internship" in job_title_lower or
                    "intern" in job_desc_lower or
                    "internship" in job_desc_lower or
                    "intern duration" in job_desc_lower or
                    "3-6 months" in job_desc_lower or
                    "summer intern" in job_title_lower
                )
                
                is_senior_role = (
                    "senior" in job_title_lower or
                    "lead" in job_title_lower or
                    "principal" in job_title_lower or
                    "staff" in job_title_lower or
                    "architect" in job_title_lower or
                    "director" in job_title_lower or
                    "manager" in job_title_lower or
                    "senior" in job_desc_lower[:500] or  # Check first 500 chars
                    "lead" in job_desc_lower[:500] or
                    "principal" in job_desc_lower[:500]
                )
                
                role_type = "intern" if is_intern_role else ("senior" if is_senior_role else "regular")
                
                if is_intern_role:
                    logger.info(f"✅ Detected INTERNSHIP role for job {idx}: {job.job_title} - applying flexible evaluation rules")
                elif is_senior_role:
                    logger.info(f"✅ Detected SENIOR role for job {idx}: {job.job_title} - applying strict experience requirements")
                else:
                    logger.info(f"✅ Detected REGULAR full-time role for job {idx}: {job.job_title} - applying flexible evaluation rules")
                
                # Create scoring prompt with better handling of sparse descriptions
                # CRITICAL: Normalize job description before hashing (remove whitespace variations)
                raw_description = (job.description or "")[:3000]  # Increased to 3000 chars
                job_description_text = ' '.join(raw_description.strip().split())  # Normalize whitespace
                description_length = len(job_description_text)
                logger.debug(f"Job description normalized: {len(raw_description)} -> {len(job_description_text)} chars")
                
                # Add role type context to prompt
                role_type_context = ""
                if is_intern_role:
                    role_type_context = """
🚨 CRITICAL: THIS IS AN INTERNSHIP ROLE - APPLY FLEXIBLE EVALUATION RULES BELOW 🚨
"""
                elif is_senior_role:
                    role_type_context = """
🚨 CRITICAL: THIS IS A SENIOR ROLE - APPLY STRICT EXPERIENCE REQUIREMENTS BELOW 🚨
"""
                else:
                    role_type_context = """
🚨 CRITICAL: THIS IS A REGULAR FULL-TIME ROLE - APPLY FLEXIBLE EVALUATION RULES BELOW 🚨
"""
                
                # ============================================================
                # TWO-STAGE PIPELINE: Requirement Extraction + Evaluation
                # ============================================================
                
                # STAGE 1: Extract frozen requirements from JD ONLY
                logger.info(f"Stage 1: Extracting frozen requirements for job {idx}")
                try:
                    from requirement_extractor import extract_frozen_requirements
                    from requirement_evaluator import evaluate_candidate_against_requirements, calculate_deterministic_score
                    
                    # Generate stable job_id for caching
                    # CRITICAL: Do NOT use idx - it changes based on position and breaks caching
                    # CRITICAL: Do NOT use full LinkedIn URL - it contains dynamic query parameters
                    job_url_for_hash = None
                    if job.url and str(job.url) != "https://example.com":
                        job_url_for_hash = str(job.url)
                        # Extract stable LinkedIn job ID if possible
                        from requirement_extractor import extract_stable_job_id_from_url
                        extracted_id = extract_stable_job_id_from_url(job_url_for_hash)
                        if extracted_id:
                            stable_job_id = extracted_id
                            logger.info(f"Using extracted stable job ID: {stable_job_id} for caching")
                        else:
                            # Fallback: use title + company (stable combination)
                            title_company = f"{(job.job_title or '')}|{(job.company or '')}"
                            stable_job_id = hashlib.sha256(title_company.encode()).hexdigest()[:16]
                            logger.info(f"Using title+company hash as stable job ID: {stable_job_id[:16]}... for caching")
                    else:
                        # Fallback: use title + company (stable combination)
                        title_company = f"{(job.job_title or '')}|{(job.company or '')}"
                        stable_job_id = hashlib.sha256(title_company.encode()).hexdigest()[:16]
                        logger.info(f"Using title+company hash as stable job ID: {stable_job_id[:16]}... for caching")
                    
                    # Stage 1: Extract frozen requirements (cached, deterministic)
                    try:
                        frozen_requirements_result = extract_frozen_requirements(
                            job_description=job_description_text,
                            job_id=stable_job_id,
                            job_title=job.job_title,
                            job_url=job_url_for_hash,
                            openai_api_key=openai_key,
                            use_cache=True  # CRITICAL: Must use cache for determinism
                        )
                        logger.info(f"Job requirements extraction: use_cache=True, job_hash will be checked for cache hit")
                        
                        frozen_requirements = frozen_requirements_result.get("frozen_requirements", [])
                        logger.info(f"Stage 1 complete: Extracted {len(frozen_requirements)} frozen requirements for job {idx}")
                        
                        # Validate frozen requirements
                        if not frozen_requirements:
                            logger.warning(f"No requirements extracted for job {idx}, using fallback")
                            # Fallback: create minimal requirement set
                            frozen_requirements = [{
                                "req_id": "REQ_01",
                                "name": "Job requirements from description",
                                "category": "core",
                                "jd_evidence": job_description_text[:200]
                            }]
                            logger.warning(f"Using fallback requirement for job {idx}")
                        
                        # FIX 2: Freeze requirement list - deepcopy and lock structure
                        import copy
                        frozen_requirements = copy.deepcopy(frozen_requirements)
                        initial_requirement_count = len(frozen_requirements)
                        initial_req_ids = {req.get("req_id") for req in frozen_requirements}
                        logger.info(f"✅ Frozen {initial_requirement_count} requirements with req_ids: {sorted(list(initial_req_ids))}")
                        
                        # FIX 3: Hard assertion function - requirement count must never change
                        def assert_requirements_frozen(current_requirements, initial_count, initial_ids, context=""):
                            """Assert that requirements list has not been mutated"""
                            current_count = len(current_requirements)
                            current_ids = {req.get("req_id") for req in current_requirements}
                            
                            if current_count != initial_count:
                                error_msg = (
                                    f"CRITICAL: Requirement count changed from {initial_count} to {current_count} {context}. "
                                    f"Freezing violated - requirements list was mutated!"
                                )
                                logger.error(error_msg)
                                raise ValueError(error_msg)
                            
                            if current_ids != initial_ids:
                                missing = initial_ids - current_ids
                                extra = current_ids - initial_ids
                                error_msg = (
                                    f"CRITICAL: Requirement IDs changed {context}. "
                                    f"Missing: {missing}, Extra: {extra}. Freezing violated!"
                                )
                                logger.error(error_msg)
                                raise ValueError(error_msg)
                    except Exception as extract_error:
                        logger.error(f"Error extracting requirements for job {idx}: {extract_error}", exc_info=True)
                        # Use fallback requirements
                        frozen_requirements = [{
                            "req_id": "REQ_01",
                            "name": "Job requirements from description",
                            "category": "core",
                            "jd_evidence": job_description_text[:200] if job_description_text else "Job description not available"
                        }]
                        logger.warning(f"Using fallback requirement due to extraction error for job {idx}")
                        
                        # FIX 2: Freeze fallback requirements too
                        import copy
                        frozen_requirements = copy.deepcopy(frozen_requirements)
                        initial_requirement_count = len(frozen_requirements)
                        initial_req_ids = {req.get("req_id") for req in frozen_requirements}
                        logger.info(f"✅ Frozen {initial_requirement_count} fallback requirements with req_ids: {sorted(list(initial_req_ids))}")
                    
                    # FIX 3: Hard assertion function - requirement count must never change
                    # Define outside try block so it's available for all code paths
                    def assert_requirements_frozen(current_requirements, initial_count, initial_ids, context=""):
                        """Assert that requirements list has not been mutated"""
                        current_count = len(current_requirements)
                        current_ids = {req.get("req_id") for req in current_requirements}
                        
                        if current_count != initial_count:
                            error_msg = (
                                f"CRITICAL: Requirement count changed from {initial_count} to {current_count} {context}. "
                                f"Freezing violated - requirements list was mutated!"
                            )
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                        
                        if current_ids != initial_ids:
                            missing = initial_ids - current_ids
                            extra = current_ids - initial_ids
                            error_msg = (
                                f"CRITICAL: Requirement IDs changed {context}. "
                                f"Missing: {missing}, Extra: {extra}. Freezing violated!"
                            )
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                    
                    # STAGE 2: Evaluate candidate against frozen requirements
                    logger.info(f"Stage 2: Evaluating candidate against {len(frozen_requirements)} frozen requirements for job {idx}")
                    
                    # Validate candidate_profile before evaluation
                    if not candidate_profile:
                        raise ValueError("candidate_profile is None or empty - cannot evaluate")
                    if not isinstance(candidate_profile, dict):
                        raise ValueError(f"candidate_profile must be a dict, got {type(candidate_profile)}")
                    
                    # PATCH 3: Hard guard - ensure resume is already parsed (should never trigger during job evaluation)
                    # resume_hash is in outer scope from generate_stream function
                    try:
                        # Check if resume_hash exists in outer scope and is in cache
                        if 'resume_hash' in dir() or 'resume_hash' in globals():
                            # Try to get resume_hash from outer scope
                            import inspect
                            frame = inspect.currentframe()
                            outer_frame = frame.f_back
                            if outer_frame and 'resume_hash' in outer_frame.f_locals:
                                current_resume_hash = outer_frame.f_locals['resume_hash']
                                if current_resume_hash and current_resume_hash not in RESUME_PARSING_CACHE:
                                    error_msg = (
                                        f"CRITICAL: Resume parsing triggered during job evaluation (hash: {current_resume_hash[:16]}...). "
                                        f"This should not happen - resume should be parsed at upload time only."
                                    )
                                    logger.error(error_msg)
                    except Exception as guard_error:
                        # Guard check failed - log but don't break execution
                        logger.debug(f"Guard check failed (non-critical): {guard_error}")
                    
                    # Validate frozen_requirements before evaluation
                    if not frozen_requirements:
                        raise ValueError("frozen_requirements is empty - cannot evaluate")
                    if not isinstance(frozen_requirements, list):
                        raise ValueError(f"frozen_requirements must be a list, got {type(frozen_requirements)}")
                    
                    # FIX 3: Assert requirements are still frozen before evaluation
                    assert_requirements_frozen(frozen_requirements, initial_requirement_count, initial_req_ids, "before evaluation")
                    
                    logger.info(f"Evaluating candidate with {len(candidate_profile.get('skills', []))} skills against {len(frozen_requirements)} requirements")
                    
                    evaluation_result = evaluate_candidate_against_requirements(
                        candidate_profile=candidate_profile,
                        frozen_requirements=frozen_requirements,
                        job_id=stable_job_id,  # Use stable_job_id instead of undefined job_id
                        job_title=job.job_title,
                        openai_api_key=openai_key
                    )
                    
                    evaluation = evaluation_result.get("evaluation", [])
                    if not evaluation:
                        raise ValueError(f"Evaluation returned empty list - expected {len(frozen_requirements)} evaluations")
                    if len(evaluation) != len(frozen_requirements):
                        raise ValueError(f"Evaluation count mismatch: {len(evaluation)} evaluations vs {len(frozen_requirements)} requirements")
                    
                    # FIX 3: Assert requirements are still frozen after evaluation
                    assert_requirements_frozen(frozen_requirements, initial_requirement_count, initial_req_ids, "after evaluation")
                    
                    logger.info(f"Stage 2 complete: Evaluated {len(evaluation)} requirements for job {idx}")
                    
                    # Calculate deterministic score
                    score_result = calculate_deterministic_score(evaluation, frozen_requirements)
                    
                    # Format response to match existing structure
                    score = score_result["match_score"]
                    requirements_met = score_result["requirements_met"]
                    total_requirements = score_result["total_requirements"]  # Core requirements only (for scoring)
                    total_all_requirements = score_result.get("total_all_requirements", len(evaluation))  # All requirements
                    requirements_satisfied_list = score_result.get("requirements_satisfied", [])
                    requirements_partially_met_list = score_result.get("requirements_partially_met", [])
                    requirements_missing_list = score_result.get("requirements_missing", [])
                    
                    # CRITICAL VALIDATION: Ensure all requirements are accounted for
                    total_classified = len(requirements_satisfied_list) + len(requirements_partially_met_list) + len(requirements_missing_list)
                    if total_classified != total_all_requirements:
                        logger.error(
                            f"REQUIREMENT COUNT MISMATCH: {total_classified} classified vs {total_all_requirements} total requirements. "
                            f"MET: {len(requirements_satisfied_list)}, PARTIAL: {len(requirements_partially_met_list)}, NOT_MET: {len(requirements_missing_list)}"
                        )
                        # Fix: Use actual counts from evaluation
                        total_all_requirements = total_classified
                    
                    # Build summary - show all three counts clearly
                    requirements_partially_met_count = len(requirements_partially_met_list)
                    requirements_not_met_count = len(requirements_missing_list)
                    
                    # Build summary text that accurately reflects all statuses
                    if requirements_met > 0 and requirements_partially_met_count == 0 and requirements_not_met_count == 0:
                        # All met
                        summary = f"The candidate meets all {total_requirements} core requirements"
                    elif requirements_met == 0 and requirements_partially_met_count > 0 and requirements_not_met_count == 0:
                        # Only partial matches
                        summary = f"The candidate partially meets {requirements_partially_met_count} out of {total_requirements} core requirements"
                    elif requirements_met > 0:
                        # Mix of met and partial/not met
                        summary = f"The candidate meets {requirements_met} out of {total_requirements} core requirements"
                        if requirements_partially_met_count > 0:
                            summary += f" and partially meets {requirements_partially_met_count} requirements"
                        if requirements_not_met_count > 0:
                            summary += f". {requirements_not_met_count} requirement(s) not met"
                    else:
                        # No met, some partial/not met
                        if requirements_partially_met_count > 0:
                            summary = f"The candidate partially meets {requirements_partially_met_count} out of {total_requirements} core requirements"
                            if requirements_not_met_count > 0:
                                summary += f". {requirements_not_met_count} requirement(s) not met"
                        else:
                            # All not met
                            summary = f"The candidate does not meet any of the {total_requirements} core requirements"
                    
                    # Build reasoning
                    reasoning = f"Deterministic evaluation: {requirements_met} MET, {len(requirements_partially_met_list)} PARTIALLY_MET, {len(requirements_missing_list)} NOT_MET out of {total_requirements} core requirements. "
                    reasoning += f"Score: ({requirements_met} + 0.6 × {len(requirements_partially_met_list)}) / {total_requirements} = {score:.3f}"
                    
                    # Extract key matches (skills that are MET or PARTIALLY_MET if core)
                    # CRITICAL: Only include requirements that are actually MET or PARTIALLY_MET
                    # Create mapping of req_id to category for checking if requirement is core
                    req_category_map = {req["req_id"]: req["category"] for req in frozen_requirements}
                    
                    # Create mapping of req_id to status for validation
                    eval_status_map = {e["req_id"]: e["status"] for e in evaluation}
                    
                    # CRITICAL: Sort evaluation by req_id first for deterministic key_matches
                    evaluation_sorted = sorted(evaluation, key=lambda x: (x.get('req_id', ''), x.get('name', '')))
                    
                    key_matches = []
                    for eval_item in evaluation_sorted:
                        # CRITICAL: Only include if status is actually MET or PARTIALLY_MET
                        # Validate status exists
                        if eval_item["status"] not in ["MET", "PARTIALLY_MET", "NOT_MET"]:
                            logger.error(f"Invalid status '{eval_item['status']}' for requirement {eval_item['req_id']}")
                            continue
                        
                        # Include MET items always
                        # Include PARTIALLY_MET items if they are core requirements
                        is_core = req_category_map.get(eval_item["req_id"]) == "core"
                        should_include = (
                            eval_item["status"] == "MET" or 
                            (eval_item["status"] == "PARTIALLY_MET" and is_core)
                        )
                        
                        if should_include:
                            # Try to extract skill name from requirement name
                            req_name = eval_item["name"]
                            # If it's a technical skill requirement, extract the skill name
                            if "(" in req_name and ")" in req_name:
                                # Format: "Programming Languages (Python, Java)"
                                skill_part = req_name.split("(")[1].split(")")[0]
                                skills = [s.strip() for s in skill_part.split(",")]
                                key_matches.extend(skills)
                            else:
                                # Single skill or other requirement - use the requirement name itself
                                # Remove prefix like "Experience: " or "Education: "
                                clean_name = req_name.split(":")[-1].strip() if ":" in req_name else req_name
                                key_matches.append(clean_name)
                    
                    # Remove duplicates and limit
                    key_matches = list(dict.fromkeys(key_matches))[:10]  # Keep first 10 unique
                    
                    # VALIDATION: Filter out key_matches that don't correspond to MET/PARTIALLY_MET requirements
                    met_names = {e["name"] for e in evaluation if e["status"] == "MET"}
                    partial_names = {e["name"] for e in evaluation if e["status"] == "PARTIALLY_MET" and req_category_map.get(e["req_id"]) == "core"}
                    valid_names = met_names | partial_names
                    
                    # Filter key_matches to only include those from valid requirements
                    filtered_key_matches = []
                    for km in key_matches:
                        # Check if this key_match is part of any valid requirement name
                        is_valid = any(km in name or name in km for name in valid_names)
                        if is_valid:
                            filtered_key_matches.append(km)
                        else:
                            logger.warning(f"Removing invalid key match '{km}' - does not correspond to any MET/PARTIALLY_MET requirement")
                    
                    key_matches = filtered_key_matches
                    
                    # Final validation: Ensure key_matches only contains items from MET/PARTIALLY_MET requirements
                    if len(key_matches) != len(filtered_key_matches):
                        logger.error(f"Key matches validation failed: {len(key_matches)} original vs {len(filtered_key_matches)} filtered")
                    
                    # Build data_result matching existing format
                    data_result = {
                        "match_score": score,
                        "key_matches": key_matches,
                        "requirements_met": requirements_met,
                        "total_requirements": total_requirements,
                        "requirements_satisfied": requirements_satisfied_list,
                        "requirements_partially_met": requirements_partially_met_list,
                        "requirements_missing": requirements_missing_list,
                        "improvements_needed": [],  # Not used in two-stage pipeline
                        "reasoning": reasoning,
                        "summary": summary
                    }
                    
                    logger.info(f"Two-stage pipeline complete for job {idx}: score={score}, met={requirements_met}/{total_requirements}")
                    
                except Exception as e:
                    import traceback
                    error_traceback = traceback.format_exc()
                    logger.error(f"Error in two-stage pipeline for job {idx}: {e}", exc_info=True)
                    logger.error(f"Full traceback:\n{error_traceback}")
                    # Fallback to default values with detailed error message
                    error_details = str(e)[:200]  # Limit error message length
                    data_result = {
                        "match_score": 0.0,
                        "key_matches": [],
                        "requirements_met": 0,
                        "total_requirements": 1,
                        "requirements_satisfied": [],
                        "requirements_partially_met": [],
                        "requirements_missing": [f"Error in requirement extraction/evaluation: {error_details}"],
                        "improvements_needed": [],
                        "reasoning": f"Error: {error_details}",
                        "summary": f"Evaluation failed: {error_details}"
                    }
                
                # Skip the old prompt-based evaluation - we already have data_result
                # Continue to score processing below
                full_response = ""  # Not used in two-stage pipeline
                
                # OLD PROMPT-BASED EVALUATION REMOVED - using two-stage pipeline above
                # The following code processes data_result from two-stage pipeline
                prompt = ""  # Not used, but keep for compatibility
                use_two_stage_pipeline = True  # Flag to skip streaming
                
                # OLD PROMPT TEXT REMOVED - using two-stage pipeline instead
                # Skip to streaming section (which will also be skipped)
                
                # OLD PROMPT-BASED STREAMING EVALUATION - SKIPPED WHEN USING TWO-STAGE PIPELINE
                # Two-stage pipeline used - skip streaming
                logger.info(f"Skipping streaming evaluation for job {idx} - using two-stage pipeline result")
                
                # Continue to data_result processing below
                # All old prompt text removed - using two-stage pipeline
                # Skip old streaming code - data_result already exists from two-stage pipeline
                full_response = ""
                
                # OLD STREAMING CODE REMOVED - using two-stage pipeline instead
                # Jump directly to data_result processing section
                
                # OLD STREAMING CODE (DISABLED - using two-stage pipeline)
                # This entire block is disabled - do not execute any code here
                # The code below is kept for reference but should never execute
                if False:  # Always False - this block never executes
                    # Run the streaming in a thread, but collect chunks and yield immediately
                    import queue
                    chunk_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
                    stream_exception = None
                    
                    def collect_stream_chunks():
                        """Collect chunks from OpenAI stream and put them in queue"""
                        nonlocal stream_exception
                        try:
                            # CRITICAL: Always use temperature=0 and seed=42 for deterministic, consistent results
                            # This ensures the same job description produces the same requirements every time
                            # Use task-specific model for matching
                            from model_config import get_model_for_task, assert_model_supports_temperature
                            matching_model = get_model_for_task("matching", json_mode=False)
                            
                            # Runtime assertion: Ensure temperature=0 is supported
                            assert_model_supports_temperature(matching_model.model_name, 0, "matching")
                            
                            stream = client.chat.completions.create(
                                model=matching_model.model_name,
                                messages=[{"role": "user", "content": prompt}],
                                stream=True,
                                temperature=0,  # MUST be 0 for deterministic output
                                seed=42  # MUST be fixed seed for consistency across multiple runs
                            )
                            chunk_count = 0
                            for chunk in stream:
                                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                                    content = chunk.choices[0].delta.content
                                    chunk_queue.put(content, block=True)  # Block until space available
                                    chunk_count += 1
                            logger.info(f"OpenAI stream completed, {chunk_count} chunks received")
                            chunk_queue.put(None, block=True)  # Signal completion
                        except Exception as e:
                            logger.error(f"Error in OpenAI stream: {e}", exc_info=True)
                            stream_exception = e
                            try:
                                chunk_queue.put(("error", str(e)), block=True)
                            except:
                                pass
                    
                    # Start collecting chunks in background thread
                    loop = asyncio.get_event_loop()
                    stream_task = loop.run_in_executor(None, collect_stream_chunks)
                    
                    # Yield chunks as they arrive - poll queue with small delays
                    try:
                        while True:
                            # Check for chunks in queue (non-blocking check)
                            try:
                                chunk_data = chunk_queue.get_nowait()
                                
                                if chunk_data is None:
                                    # Stream complete
                                    break
                                elif isinstance(chunk_data, tuple) and chunk_data[0] == "error":
                                    # Error occurred
                                    yield format_sse_event("error", {
                                        "message": chunk_data[1] if len(chunk_data) > 1 else "Unknown error"
                                    })
                                    await asyncio.sleep(0)  # Force flush
                                    break
                                else:
                                    # Valid content chunk - yield immediately
                                    content = chunk_data
                                    full_response += content
                                    # Stream each token as it arrives (for summary generation later)
                                    # Note: For scoring, we collect all chunks first, then parse JSON
                                    # We'll stream summary chunks separately
                            except queue.Empty:
                                # No chunk available yet - check if stream is done
                                if stream_exception:
                                    yield format_sse_event("error", {
                                        "message": str(stream_exception)
                                    })
                                    await asyncio.sleep(0)  # Force flush
                                    break
                                
                                # Check if task is done
                                if stream_task.done():
                                    # Task finished, check if there are any remaining chunks
                                    try:
                                        # Try to get any remaining chunks
                                        while True:
                                            chunk_data = chunk_queue.get_nowait()
                                            if chunk_data is None:
                                                break
                                            elif isinstance(chunk_data, tuple) and chunk_data[0] == "error":
                                                yield format_sse_event("error", {
                                                    "message": chunk_data[1] if len(chunk_data) > 1 else "Unknown error"
                                                })
                                                await asyncio.sleep(0)  # Force flush
                                                break
                                            else:
                                                # Collect content for JSON parsing (not streaming individual tokens for scoring)
                                                content = chunk_data
                                                full_response += content
                                    except queue.Empty:
                                        pass
                                    break
                                
                                # Yield control briefly to allow other tasks and check again
                                # Use a very short sleep to check frequently for real-time streaming
                                await asyncio.sleep(0.001)  # Very short delay for near real-time streaming
                                continue
                            
                    except Exception as e:
                        logger.error(f"Error in streaming loop: {e}", exc_info=True)
                    finally:
                        # Ensure stream task completes
                        try:
                            logger.info(f"Waiting for stream task to complete for job {idx}")
                            await stream_task
                            logger.info(f"Stream task completed for job {idx}")
                        except Exception as e:
                            logger.error(f"Error waiting for stream task: {e}", exc_info=True)
                
                # OLD STREAMING CODE ENDS HERE (disabled block above)
                
                # Two-stage pipeline used - skip streaming
                logger.info(f"Skipping streaming evaluation for job {idx} - using two-stage pipeline result")
                
                # Parse the response (skip if using two-stage pipeline)
                if not use_two_stage_pipeline and not data_result:
                    logger.info(f"Parsing JSON response for job {idx}, response length: {len(full_response)}")
                    
                    # Write raw LLM response for job scoring to file
                    if response_file:
                        response_file.write(f"=== RAW LLM RESPONSE (Job Scoring - Job {idx}) ===\n")
                        response_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        response_file.write(f"Length: {len(full_response)} characters\n")
                        response_file.write(f"{'='*60}\n\n")
                        response_file.write(full_response)
                        response_file.write(f"\n\n")
                        response_file.flush()
                        logger.info(f"Raw LLM response for job scoring written to file for job {idx}")
                    
                    from agents import extract_json_from_response
                    data_result = extract_json_from_response(full_response)
                    logger.info(f"JSON parsed for job {idx}, match_score: {data_result.get('match_score') if data_result else 'None'}")
                else:
                    # Write two-stage pipeline result to file
                    if response_file:
                        response_file.write(f"=== TWO-STAGE PIPELINE RESULT (Job {idx}) ===\n")
                        response_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        response_file.write(f"{'='*60}\n\n")
                        response_file.write(json.dumps(data_result, indent=2))
                        response_file.write(f"\n\n")
                        response_file.flush()
                        logger.info(f"Two-stage pipeline result written to file for job {idx}")
                
                # Validate and fix data_result with comprehensive fallbacks
                if not data_result or not isinstance(data_result, dict):
                    logger.warning(f"Invalid data_result for job {idx}, initializing defaults")
                    data_result = {}
                
                # Ensure match_score exists and is valid
                if "match_score" not in data_result or data_result.get("match_score") is None:
                    try:
                        # Try to calculate a default score based on available data
                        data_result["match_score"] = 0.5
                    except:
                        data_result["match_score"] = 0.5
                
                # Ensure all required fields exist with correct types
                if "key_matches" not in data_result or not isinstance(data_result.get("key_matches"), list):
                    data_result["key_matches"] = []
                if "requirements_met" not in data_result or not isinstance(data_result.get("requirements_met"), (int, float)):
                    data_result["requirements_met"] = 0
                if "total_requirements" not in data_result or not isinstance(data_result.get("total_requirements"), (int, float)):
                    data_result["total_requirements"] = 1
                if "requirements_satisfied" not in data_result or not isinstance(data_result.get("requirements_satisfied"), list):
                    data_result["requirements_satisfied"] = []
                if "requirements_missing" not in data_result or not isinstance(data_result.get("requirements_missing"), list):
                    data_result["requirements_missing"] = []
                if "requirements_partially_met" not in data_result or not isinstance(data_result.get("requirements_partially_met"), list):
                    data_result["requirements_partially_met"] = []
                if "improvements_needed" not in data_result or not isinstance(data_result.get("improvements_needed"), list):
                    data_result["improvements_needed"] = []
                if "reasoning" not in data_result or not isinstance(data_result.get("reasoning"), str):
                    data_result["reasoning"] = "Score calculated based on candidate-job alignment"
                
                score = float(data_result.get("match_score", 0.5))
                
                # Normalize score to 0-1 range (in case LLM returns percentage 0-100)
                if score > 1.0:
                    logger.warning(f"Match score {score} is > 1.0, normalizing to decimal (dividing by 100)")
                    score = score / 100.0
                
                # Ensure score is between 0 and 1
                score = max(0.0, min(1.0, score))
                
                requirements_satisfied_list = data_result.get("requirements_satisfied", []) or []
                requirements_missing_list = data_result.get("requirements_missing", []) or []
                requirements_partially_met_list = data_result.get("requirements_partially_met", []) or []
                
                # GOLDEN RULE: Cap score at 0.75 if ANY REQUIRED requirement is PARTIALLY MET
                # Score = 1.00 is ONLY allowed when ALL REQUIRED requirements are MET and NO REQUIRED requirement is PARTIALLY MET
                # Count PARTIALLY MET requirements that are REQUIRED (not preferred)
                # Preferred qualifications are already excluded from total_requirements, so any partially_met here are required
                if len(requirements_partially_met_list) > 0:
                    # There are partially met REQUIRED requirements - cap score at 0.75
                    if score > 0.75:
                        logger.info(f"Job {idx}: Capping score from {score} to 0.75 because {len(requirements_partially_met_list)} REQUIRED requirement(s) are PARTIALLY MET")
                        score = 0.75
                
                # Post-processing: Filter out invented requirements (certifications, location, visa, clearance unless explicitly mentioned)
                # This prevents LLM from adding requirements not in the job description
                def is_invented_requirement(req_text):
                    """Check if requirement was invented (not explicitly in job description)"""
                    req_lower = str(req_text).lower()
                    # Keywords that should only appear if explicitly in job description
                    invented_keywords = [
                        "location", "visa", "work authorization", "security clearance", 
                        "SC clearance", "right to work", "residency", "based in",
                        "certification", "certified", "license", "licensed"
                    ]
                    for keyword in invented_keywords:
                        if keyword in req_lower:
                            # Check if keyword appears in job description
                            if keyword not in job_description_text.lower():
                                logger.warning(f"Filtering out invented requirement: {req_text} (keyword '{keyword}' not in job description)")
                                return True
                    # Special check: If requirement says "no specific X required" or "not mentioned", it's invented
                    if "no specific" in req_lower or ("not mentioned" in req_lower and "candidate" not in req_lower) or "not required" in req_lower:
                        logger.warning(f"Filtering out invented requirement: {req_text} (contains 'no specific' or 'not mentioned')")
                        return True
                    return False
                
                # Filter out invented requirements from all lists
                original_satisfied_count = len(requirements_satisfied_list)
                original_missing_count = len(requirements_missing_list)
                original_partial_count = len(requirements_partially_met_list)
                
                requirements_satisfied_list = [r for r in requirements_satisfied_list if not is_invented_requirement(r)]
                requirements_missing_list = [r for r in requirements_missing_list if not is_invented_requirement(r)]
                requirements_partially_met_list = [r for r in requirements_partially_met_list if not is_invented_requirement(r)]
                
                filtered_count = (original_satisfied_count - len(requirements_satisfied_list)) + \
                                (original_missing_count - len(requirements_missing_list)) + \
                                (original_partial_count - len(requirements_partially_met_list))
                if filtered_count > 0:
                    logger.info(f"Filtered out {filtered_count} invented requirement(s) for job {idx}")
                
                # Post-processing: Fix education classification - degree in progress should be PARTIALLY_MET, not NOT_MET
                def fix_education_classification(req_text, candidate_profile):
                    """Fix education requirements that should be PARTIALLY_MET"""
                    req_lower = str(req_text).lower()
                    if "education" in req_lower or "degree" in req_lower or "bachelor" in req_lower or "master" in req_lower:
                        # Check if candidate is pursuing degree
                        education = candidate_profile.get("education", []) or []
                        for edu in education:
                            dates = str(edu.get("dates", "")).lower()
                            degree = str(edu.get("degree", "")).lower()
                            # Check if degree is in progress (future dates, "present", "expected")
                            if any(indicator in dates for indicator in ["2027", "2028", "2029", "2030", "present", "expected", "pursuing"]):
                                if "not met" in req_lower and ("bachelor" in degree or "master" in degree or "phd" in degree):
                                    logger.warning(f"Fixing education classification: {req_text} should be PARTIALLY_MET (degree in progress)")
                                    return True
                    return False
                
                # Move incorrectly classified education requirements from missing to partially_met
                education_to_move = []
                for item in requirements_missing_list:
                    if fix_education_classification(item, candidate_profile):
                        education_to_move.append(item)
                
                for item in education_to_move:
                    requirements_missing_list.remove(item)
                    if item not in requirements_partially_met_list:
                        # Update the status in the text
                        updated_item = item.replace("NOT MET", "PARTIALLY MET").replace("not met", "PARTIALLY MET")
                        if "degree" in updated_item.lower() and "pursuing" not in updated_item.lower() and "in progress" not in updated_item.lower():
                            # Add context about degree in progress
                            if "candidate" in updated_item.lower():
                                updated_item = updated_item.replace("candidate", "candidate is pursuing").replace("has", "is pursuing")
                            else:
                                updated_item = updated_item + " (candidate is pursuing degree)"
                        requirements_partially_met_list.append(updated_item)
                        logger.info(f"Moved education requirement from missing to partially_met: {item}")
                
                # Post-processing: Fix preferred skills classification
                def fix_preferred_skill_classification(req_text, candidate_profile):
                    """Fix preferred skills that should be PARTIALLY_MET when candidate has them"""
                    req_lower = str(req_text).lower()
                    # Check if this is about a preferred skill
                    if "preferred" in req_lower or "nice to have" in req_lower:
                        # Extract skill name from requirement text and check candidate skills
                        candidate_skills = [s.lower() for s in (candidate_profile.get("skills", []) or [])]
                        # Check if any candidate skill matches (including variations)
                        for skill in candidate_skills:
                            # Check for skill name in requirement text
                            if skill in req_lower or any(variation in req_lower for variation in [skill + ".js", skill + "js", skill.replace(" ", "")]):
                                if "not met" in req_lower:
                                    logger.warning(f"Fixing preferred skill classification: {req_text} should be PARTIALLY_MET (candidate has preferred skill: {skill})")
                                    return True
                    return False
                
                # Move incorrectly classified preferred skills from missing to partially_met
                preferred_skills_to_move = []
                for item in requirements_missing_list:
                    if fix_preferred_skill_classification(item, candidate_profile):
                        preferred_skills_to_move.append(item)
                
                for item in preferred_skills_to_move:
                    requirements_missing_list.remove(item)
                    if item not in requirements_partially_met_list:
                        # Update the status in the text
                        updated_item = item.replace("NOT MET", "PARTIALLY MET").replace("not met", "PARTIALLY MET")
                        if "preferred" not in updated_item.lower() and "nice to have" not in updated_item.lower():
                            updated_item = updated_item + " (preferred skill matched)"
                        requirements_partially_met_list.append(updated_item)
                        logger.info(f"Moved preferred skill from missing to partially_met: {item}")
                
                # Post-processing: Fix senior role skill classifications
                # For senior roles, if candidate has the skill but lacks extensive experience, it should be PARTIALLY MET, not NOT MET
                def fix_senior_role_skill_classification(req_text, candidate_profile, job_title):
                    """Fix senior role skill requirements - if skill is present but experience is insufficient, mark as PARTIALLY MET"""
                    job_title_lower = (job_title or "").lower()
                    # Check if this is a senior role
                    is_senior = any(word in job_title_lower for word in ["senior", "lead", "principal", "staff", "architect", "director", "manager"])
                    if not is_senior:
                        return False
                    
                    req_lower = str(req_text).lower()
                    # Check if this is about a technical skill with experience requirement
                    experience_keywords = ["extensive experience", "strong experience", "hands-on experience", "experience in", "experience with"]
                    if not any(keyword in req_lower for keyword in experience_keywords):
                        return False
                    
                    # Extract skill names from requirement and check candidate skills
                    candidate_skills = [s.lower() for s in (candidate_profile.get("skills", []) or [])]
                    # Common technical skills to check (including variations)
                    tech_skills_map = {
                        "python": ["python"],
                        "react": ["react", "react.js", "reactjs"],
                        "javascript": ["javascript", "js", "ecmascript"],
                        "java": ["java"],
                        "typescript": ["typescript", "ts"],
                        "node": ["node", "node.js", "nodejs"],
                        "docker": ["docker"],
                        "kubernetes": ["kubernetes", "k8s"],
                        "aws": ["aws", "amazon web services"],
                        "azure": ["azure"],
                        "gcp": ["gcp", "google cloud", "google cloud platform"],
                        "sql": ["sql", "mysql", "postgresql", "postgres"],
                        "mongodb": ["mongodb", "mongo"],
                        "postgresql": ["postgresql", "postgres"],
                        "mysql": ["mysql"],
                        # ML/AI Frameworks
                        "tensorflow": ["tensorflow", "tf"],
                        "keras": ["keras"],
                        "pytorch": ["pytorch", "torch"],
                        "scikit-learn": ["scikit-learn", "scikit", "sklearn"],
                        "pandas": ["pandas"],
                        "numpy": ["numpy", "np"],
                        "opencv": ["opencv", "cv2"],
                        "machine learning": ["machine learning", "ml"],
                        "deep learning": ["deep learning", "dl"],
                        "neural networks": ["neural networks", "nn", "neural network"],
                        "computer vision": ["computer vision", "cv"],
                        "nlp": ["nlp", "natural language processing"],
                        "data science": ["data science"],
                        "data analysis": ["data analysis"]
                    }
                    
                    for skill_key, skill_variations in tech_skills_map.items():
                        # Check if requirement mentions this skill
                        if skill_key in req_lower:
                            # Check if candidate has this skill (including variations)
                            if any(var in candidate_skills for var in skill_variations):
                                # Candidate has the skill but requirement says NOT MET or "does not have experience" - should be PARTIALLY MET
                                if "not met" in req_lower or "does not have experience" in req_lower or "does not mention" in req_lower:
                                    logger.warning(f"Fixing senior role skill classification: {req_text} should be PARTIALLY MET (candidate has {skill_key} but lacks extensive experience)")
                                    return True
                    return False
                
                # Move incorrectly classified senior role skills from missing to partially_met
                senior_skills_to_move = []
                for item in requirements_missing_list:
                    if fix_senior_role_skill_classification(item, candidate_profile, job.job_title):
                        senior_skills_to_move.append(item)
                
                for item in senior_skills_to_move:
                    requirements_missing_list.remove(item)
                    if item not in requirements_partially_met_list:
                        # Update the status in the text
                        updated_item = item.replace("NOT MET", "PARTIALLY MET").replace("not met", "PARTIALLY MET")
                        if "lacks extensive experience" not in updated_item.lower() and "lacks experience" not in updated_item.lower():
                            updated_item = updated_item.replace("(NOT MET -", "(PARTIALLY MET - has skill but").replace("(not met -", "(PARTIALLY MET - has skill but")
                        requirements_partially_met_list.append(updated_item)
                        logger.info(f"Moved senior role skill from missing to partially_met: {item}")
                
                # Post-processing: Fix misclassified skills in requirements_missing
                # Check if any skills marked as NOT MET are actually in candidate's skills array
                def check_skill_in_candidate_profile(req_text, candidate_profile):
                    """Check if a skill mentioned in requirement is actually in candidate's skills array"""
                    from agents import normalize_skill
                    
                    req_lower = str(req_text).lower()
                    # Extract skill name from requirement text (format: "SkillName (NOT MET - ...)")
                    # Try to extract skill name before the first parenthesis
                    skill_name_match = req_text.split("(")[0].strip() if "(" in req_text else req_text.strip()
                    skill_name = skill_name_match.strip()
                    
                    # Get candidate skills (normalized)
                    candidate_skills = [normalize_skill(s) for s in (candidate_profile.get("skills", []) or [])]
                    candidate_skills_lower = [s.lower() for s in candidate_skills]
                    
                    # Normalize the extracted skill name
                    normalized_skill = normalize_skill(skill_name).lower()
                    
                    # Check if skill is in candidate's skills (exact match or contains)
                    if normalized_skill in candidate_skills_lower:
                        return True
                    
                    # Also check if any candidate skill contains the requirement skill or vice versa
                    for candidate_skill in candidate_skills_lower:
                        if normalized_skill in candidate_skill or candidate_skill in normalized_skill:
                            return True
                    
                    # Check experience_summary for skill mentions
                    experience_summary = (candidate_profile.get("experience_summary", "") or "").lower()
                    if normalized_skill in experience_summary:
                        return True
                    
                    return False
                
                # Move skills from missing to satisfied/partially_met if they're actually in candidate profile
                skills_to_fix = []
                for item in requirements_missing_list:
                    item_lower = str(item).lower()
                    # Check if this is a skill requirement (contains "NOT MET" and mentions a technology)
                    if "not met" in item_lower and ("candidate does not list" in item_lower or "candidate lacks" in item_lower or "does not have" in item_lower):
                        if check_skill_in_candidate_profile(item, candidate_profile):
                            skills_to_fix.append(item)
                
                for item in skills_to_fix:
                    requirements_missing_list.remove(item)
                    # Extract skill name for better message
                    skill_name = item.split("(")[0].strip() if "(" in item else item.strip()
                    # Determine if it should be MET or PARTIALLY MET
                    # If it's a required skill and candidate has it, it's MET
                    # If it's a preferred skill or requires extensive experience, it's PARTIALLY MET
                    item_lower = str(item).lower()
                    is_preferred = "preferred" in item_lower or "nice to have" in item_lower
                    requires_experience = any(keyword in item_lower for keyword in ["extensive experience", "strong experience", "hands-on experience", "years of experience"])
                    
                    if is_preferred or requires_experience:
                        # Move to partially_met
                        updated_item = item.replace("NOT MET", "PARTIALLY MET").replace("not met", "PARTIALLY MET")
                        updated_item = updated_item.replace("(NOT MET -", "(PARTIALLY MET - candidate has skill").replace("(not met -", "(PARTIALLY MET - candidate has skill")
                        if item not in requirements_partially_met_list:
                            requirements_partially_met_list.append(updated_item)
                        logger.info(f"Fixed misclassified skill: {item} → PARTIALLY MET (skill found in candidate profile)")
                    else:
                        # Move to satisfied (required skill that candidate has)
                        updated_item = item.replace("NOT MET", "MET").replace("not met", "MET")
                        updated_item = updated_item.replace("(NOT MET -", "(MET - candidate lists").replace("(not met -", "(MET - candidate lists")
                        if item not in requirements_satisfied_list:
                            requirements_satisfied_list.append(updated_item)
                        logger.info(f"Fixed misclassified skill: {item} → MET (skill found in candidate profile)")
                
                # Post-processing: Fix misclassified requirements
                # Move any items from requirements_satisfied that indicate the candidate doesn't have them
                items_to_move_to_missing = []
                for item in requirements_satisfied_list:
                    item_lower = str(item).lower()
                    if "(not mentioned in candidate profile)" in item_lower or "(candidate lacks" in item_lower or "not mentioned" in item_lower:
                        items_to_move_to_missing.append(item)
                        logger.warning(f"Found misclassified requirement in requirements_satisfied, moving to requirements_missing: {item}")
                
                # Move items from satisfied to missing
                for item in items_to_move_to_missing:
                    requirements_satisfied_list.remove(item)
                    if item not in requirements_missing_list:
                        requirements_missing_list.append(item)
                
                # Post-processing: Move PARTIALLY MET items from requirements_satisfied to requirements_partially_met
                items_to_move_to_partially_met = []
                for item in requirements_satisfied_list:
                    item_lower = str(item).lower()
                    # Check if item contains "PARTIALLY MET" or "partially met"
                    if "partially met" in item_lower:
                        items_to_move_to_partially_met.append(item)
                        logger.warning(f"Found PARTIALLY MET requirement in requirements_satisfied, moving to requirements_partially_met: {item}")
                
                # Move items from satisfied to partially_met
                for item in items_to_move_to_partially_met:
                    requirements_satisfied_list.remove(item)
                    if item not in requirements_partially_met_list:
                        requirements_partially_met_list.append(item)
                        logger.info(f"Moved PARTIALLY MET requirement from satisfied to partially_met: {item}")
                
                # Post-processing: Fix contradictory evidence text in requirements_satisfied
                # Fix items that say "MET - candidate does not list" when candidate actually has the skill
                def fix_contradictory_evidence(req_text, candidate_profile):
                    """Fix requirements that say MET but have contradictory evidence like 'does not list'"""
                    from agents import normalize_skill
                    
                    req_lower = str(req_text).lower()
                    # Check if it says MET but has contradictory evidence
                    if "met" in req_lower and ("does not list" in req_lower or "does not have" in req_lower or "lacks" in req_lower):
                        # Extract skill name
                        skill_name_match = req_text.split("(")[0].strip() if "(" in req_text else req_text.strip()
                        skill_name = skill_name_match.strip()
                        
                        # Get candidate skills (normalized)
                        candidate_skills = [normalize_skill(s) for s in (candidate_profile.get("skills", []) or [])]
                        candidate_skills_lower = [s.lower() for s in candidate_skills]
                        
                        # Normalize the extracted skill name
                        normalized_skill = normalize_skill(skill_name).lower()
                        
                        # Use tech_skills_map to check for variations
                        tech_skills_map = {
                            "python": ["python"],
                            "react": ["react", "react.js", "reactjs"],
                            "javascript": ["javascript", "js", "ecmascript"],
                            "java": ["java"],
                            "typescript": ["typescript", "ts"],
                            "node": ["node", "node.js", "nodejs"],
                            "docker": ["docker"],
                            "kubernetes": ["kubernetes", "k8s"],
                            "aws": ["aws", "amazon web services"],
                            "azure": ["azure"],
                            "gcp": ["gcp", "google cloud", "google cloud platform"],
                            "sql": ["sql", "mysql", "postgresql", "postgres"],
                            "mongodb": ["mongodb", "mongo"],
                            "postgresql": ["postgresql", "postgres"],
                            "mysql": ["mysql"],
                            "tensorflow": ["tensorflow", "tf"],
                            "keras": ["keras"],
                            "pytorch": ["pytorch", "torch"],
                            "scikit-learn": ["scikit-learn", "scikit", "sklearn"],
                            "pandas": ["pandas"],
                            "numpy": ["numpy", "np"],
                            "opencv": ["opencv", "cv2"],
                            "machine learning": ["machine learning", "ml"],
                            "deep learning": ["deep learning", "dl"],
                            "computer vision": ["computer vision", "cv"],
                            "nlp": ["nlp", "natural language processing"],
                            "data science": ["data science"],
                            "data analysis": ["data analysis"]
                        }
                        
                        # Check if skill is actually in candidate's skills (exact match)
                        if normalized_skill in candidate_skills_lower:
                            # Find the actual skill name from candidate's skills
                            for cs in candidate_skills:
                                if normalize_skill(cs).lower() == normalized_skill:
                                    return cs  # Return the actual skill name found
                        
                        # Check for variations using tech_skills_map
                        for skill_key, skill_variations in tech_skills_map.items():
                            if skill_key in normalized_skill or normalized_skill in skill_key:
                                # Check if any variation is in candidate skills
                                for var in skill_variations:
                                    if var in candidate_skills_lower:
                                        # Find the actual skill name
                                        for cs in candidate_skills:
                                            if normalize_skill(cs).lower() == var:
                                                return cs
                        
                        # Check for substring matches
                        for candidate_skill in candidate_skills_lower:
                            if normalized_skill in candidate_skill or candidate_skill in normalized_skill:
                                # Find the actual skill name
                                for cs in candidate_skills:
                                    if normalize_skill(cs).lower() == candidate_skill:
                                        return cs
                        
                        # Check experience_summary
                        experience_summary = (candidate_profile.get("experience_summary", "") or "").lower()
                        if normalized_skill in experience_summary:
                            return skill_name  # Skill mentioned in experience
                    
                    return None
                
                # Fix contradictory evidence in requirements_satisfied
                items_to_fix_evidence = []
                for item in requirements_satisfied_list:
                    fixed_skill = fix_contradictory_evidence(item, candidate_profile)
                    if fixed_skill:
                        items_to_fix_evidence.append((item, fixed_skill))
                
                for item, actual_skill in items_to_fix_evidence:
                    # Replace the contradictory evidence with correct evidence
                    if "does not list" in item.lower():
                        # Extract the requirement name (before the parenthesis)
                        req_name = item.split("(")[0].strip() if "(" in item else item.strip()
                        # Create new item with correct evidence
                        updated_item = f"{req_name} (MET - candidate lists {actual_skill})"
                        requirements_satisfied_list.remove(item)
                        if updated_item not in requirements_satisfied_list:
                            requirements_satisfied_list.append(updated_item)
                        logger.info(f"Fixed contradictory evidence: {item} → {updated_item}")
                    elif "does not have" in item.lower() or "lacks" in item.lower():
                        req_name = item.split("(")[0].strip() if "(" in item else item.strip()
                        updated_item = f"{req_name} (MET - candidate has {actual_skill})"
                        requirements_satisfied_list.remove(item)
                        if updated_item not in requirements_satisfied_list:
                            requirements_satisfied_list.append(updated_item)
                        logger.info(f"Fixed contradictory evidence: {item} → {updated_item}")
                
                # Post-processing: Remove duplicate requirements
                # Check for duplicates between satisfied and partially_met (same requirement shouldn't be in both)
                def extract_requirement_name(req_text):
                    """Extract the core requirement name (before status indicator)"""
                    # Remove status indicators like "(MET - ...)", "(PARTIALLY MET - ...)", "(NOT MET - ...)"
                    req_text = str(req_text)
                    # Remove everything after first parenthesis if it contains status
                    if "(" in req_text:
                        before_paren = req_text.split("(")[0].strip()
                        # Check if it's a status indicator
                        if any(status in req_text.lower() for status in ["met -", "partially met -", "not met -"]):
                            return before_paren.lower()
                    return req_text.lower()
                
                # Remove duplicates: if same requirement is in both satisfied and partially_met, keep only in partially_met
                satisfied_names = {extract_requirement_name(item): item for item in requirements_satisfied_list}
                partially_met_names = {extract_requirement_name(item): item for item in requirements_partially_met_list}
                
                duplicates_to_remove = []
                for name, item in satisfied_names.items():
                    if name in partially_met_names:
                        # Same requirement in both - remove from satisfied, keep in partially_met
                        duplicates_to_remove.append(item)
                        logger.warning(f"Found duplicate requirement in both satisfied and partially_met: {name}. Keeping in partially_met, removing from satisfied.")
                
                for item in duplicates_to_remove:
                    if item in requirements_satisfied_list:
                        requirements_satisfied_list.remove(item)
                
                # Also check for duplicates within the same list (same requirement with different wording)
                # Remove exact duplicates
                requirements_satisfied_list = list(dict.fromkeys(requirements_satisfied_list))  # Preserves order
                requirements_partially_met_list = list(dict.fromkeys(requirements_partially_met_list))
                requirements_missing_list = list(dict.fromkeys(requirements_missing_list))
                
                # SCORING: MET counts fully, PARTIALLY_MET counts as 50% (0.5) toward match score
                # Formula: (MET + 0.5 × PARTIALLY_MET) / total_requirements
                requirements_met = len(requirements_satisfied_list)  # Count of fully met requirements
                total_requirements = int(data_result.get("total_requirements", 0))
                
                # Update requirements_met count if items were moved
                if items_to_move_to_missing or items_to_move_to_partially_met:
                    # Recalculate requirements_met - only fully satisfied count
                    requirements_met = len(requirements_satisfied_list)
                    logger.info(f"Adjusted requirements_met from {data_result.get('requirements_met', 0)} to {requirements_met} after fixing misclassifications (moved {len(items_to_move_to_missing)} to missing, {len(items_to_move_to_partially_met)} to partially_met)")
                
                # VALIDATION: Ensure total_requirements matches the sum of satisfied + missing + partially met
                actual_total = len(requirements_satisfied_list) + len(requirements_missing_list) + len(requirements_partially_met_list)
                if total_requirements != actual_total and actual_total > 0:
                    logger.warning(f"Total requirements mismatch for job {idx}: declared={total_requirements}, actual={actual_total}. Adjusting total_requirements to match actual count.")
                    total_requirements = actual_total
                    # Also update requirements_met to match calculated value
                    requirements_met_raw = len(requirements_satisfied_list) + (0.5 * len(requirements_partially_met_list))
                    requirements_met = int(round(requirements_met_raw))
                
                # VALIDATION: Ensure requirements_met matches the count of fully satisfied requirements
                # PARTIALLY_MET does NOT count toward requirements_met
                expected_met = len(requirements_satisfied_list)
                if requirements_met != expected_met:
                    logger.warning(f"Requirements met mismatch for job {idx}: declared={requirements_met}, expected={expected_met}. Adjusting requirements_met to match satisfied count.")
                    requirements_met = expected_met
                
                # Fallback: If no requirements extracted AND description is empty/sparse, use minimal fallback
                # NOTE: This fallback should be VERY conservative - only use if description is truly empty
                if total_requirements == 0 and len(requirements_satisfied_list) == 0 and len(requirements_missing_list) == 0 and len(requirements_partially_met_list) == 0:
                    # Only use fallback if description is missing or very short (less than 50 chars)
                    description_available = job_description_text and len(job_description_text.strip()) > 50
                    
                    if not description_available:
                        logger.warning(f"No requirements extracted and description is empty/sparse for job {idx}. Using minimal role-type-only fallback.")
                        # VERY CONSERVATIVE: Only extract the most basic role type from title, nothing else
                        job_title_lower = (job.job_title or "").lower()
                        candidate_skills_lower = [s.lower() for s in (candidate_profile.get("skills", []) or [])]
                        
                        # Extract ONLY the most basic role type (e.g., "Software Engineer" → "Software development role")
                        # DO NOT add specific skills, tools, or experience requirements
                        role_type_requirement = None
                        if any(word in job_title_lower for word in ["developer", "engineer", "programmer", "coder"]):
                            role_type_requirement = "Software development role"
                        elif any(word in job_title_lower for word in ["analyst", "data"]):
                            role_type_requirement = "Data analysis role"
                        elif any(word in job_title_lower for word in ["manager", "lead", "director"]):
                            role_type_requirement = "Management role"
                        elif any(word in job_title_lower for word in ["designer", "design"]):
                            role_type_requirement = "Design role"
                        elif any(word in job_title_lower for word in ["marketing", "sales"]):
                            role_type_requirement = "Marketing/Sales role"
                        else:
                            role_type_requirement = "Role type from job title"
                        
                        # Only add this ONE minimal requirement
                        if role_type_requirement:
                            # Check if candidate has any relevant skills (very loose match)
                            matched = False
                            for skill in candidate_skills_lower:
                                # Very basic keyword matching - only for role type, not specific skills
                                if any(keyword in skill for keyword in role_type_requirement.lower().split() if len(keyword) > 4):
                                    matched = True
                                    break
                            
                            if matched:
                                requirements_satisfied_list.append(f"{role_type_requirement} (candidate has relevant background)")
                            else:
                                requirements_missing_list.append(f"{role_type_requirement} (role type from title)")
                            
                            # Update counts
                            total_requirements = 1
                            requirements_met = 1 if matched else 0
                            
                            logger.info(f"Used minimal fallback: {role_type_requirement} (matched: {matched})")
                    else:
                        # Description exists but LLM didn't extract requirements - this shouldn't happen with new prompt
                        # Log warning but don't add generic requirements
                        logger.warning(f"Description available ({len(job_description_text)} chars) but no requirements extracted. This may indicate the description is too sparse or LLM extraction failed.")
                        # Set minimal defaults to avoid division by zero
                        total_requirements = 1
                        requirements_met = 0
                
                # Final validation: Ensure consistency
                if total_requirements == 0:
                    # Recalculate from actual lists
                    total_requirements = len(requirements_satisfied_list) + len(requirements_missing_list) + len(requirements_partially_met_list)
                    if total_requirements == 0:
                        # Only set to 1 if description is truly empty
                        if not job_description_text or len(job_description_text.strip()) <= 50:
                            total_requirements = 1
                        else:
                            # Description exists but no requirements extracted - this is a problem
                            logger.error(f"CRITICAL: Description available ({len(job_description_text)} chars) but no requirements extracted for job {idx}. This indicates extraction failure.")
                            total_requirements = 1  # Set to 1 to avoid division by zero, but log error
                    requirements_met_raw = len(requirements_satisfied_list) + (0.5 * len(requirements_partially_met_list))
                    requirements_met = int(round(requirements_met_raw))
                
                # Final consistency check: total should equal satisfied + missing + partially met
                actual_total = len(requirements_satisfied_list) + len(requirements_missing_list) + len(requirements_partially_met_list)
                if total_requirements != actual_total and actual_total > 0:
                    logger.warning(f"Final validation: Total requirements mismatch for job {idx}: declared={total_requirements}, actual={actual_total}. Correcting to actual count.")
                    total_requirements = actual_total
                
                # Final consistency check: requirements_met should equal satisfied count only
                # PARTIALLY_MET does NOT count toward requirements_met
                expected_met = len(requirements_satisfied_list)
                if requirements_met != expected_met:
                    logger.warning(f"Final validation: Requirements met mismatch for job {idx}: declared={requirements_met}, expected={expected_met}. Correcting to satisfied count only.")
                    requirements_met = expected_met
                
                if requirements_met > total_requirements:
                    logger.warning(f"Requirements met ({requirements_met}) exceeds total requirements ({total_requirements}) for job {idx}. Capping to total.")
                    requirements_met = total_requirements
                
                # ALWAYS recalculate score from actual requirements_met / total_requirements to ensure mathematical accuracy
                # PARTIALLY MET requirements count as 50% (0.5) toward the score
                # Formula: (MET + 0.5 * PARTIALLY_MET) / total_requirements
                if total_requirements > 0:
                    partially_met_count = len(requirements_partially_met_list)
                    # Calculate score with 50% weight for partially met requirements
                    calculated_score = round((requirements_met + 0.5 * partially_met_count) / total_requirements, 3)
                    # Always use calculated score for accuracy (LLM may miscalculate)
                    if abs(calculated_score - score) > 0.001:  # Use calculated if difference > 0.1%
                        logger.info(f"Job {idx}: Recalculating score from {score} to {calculated_score} based on actual counts (met={requirements_met}, partially_met={partially_met_count}, total={total_requirements})")
                        score = calculated_score
                    # Remove the 0.75 cap since partially met requirements now contribute to score
                    # Score can exceed 0.75 if there are many partially met requirements
                
                # Log final validation summary
                logger.info(f"Job {idx} requirements validation: total={total_requirements}, satisfied={len(requirements_satisfied_list)}, partially_met={len(requirements_partially_met_list)}, missing={len(requirements_missing_list)}, met={requirements_met}, final_score={score}")
                
                # Extract reasoning separately (needed for logging)
                reasoning = data_result.get("reasoning", "Score calculated based on candidate-job alignment")
                
                # Generate summary using ACTUAL counts after all post-processing
                # Use actual counts from lists, not from LLM response (which may be incorrect)
                actual_satisfied_count = len(requirements_satisfied_list)
                actual_partially_met_count = len(requirements_partially_met_list)
                actual_missing_count = len(requirements_missing_list)
                actual_total = total_requirements
                
                # Build summary with correct counts
                summary_parts = []
                if actual_total > 0:
                    summary_parts.append(f"The candidate meets {actual_satisfied_count} out of {actual_total} requirements")
                    if actual_partially_met_count > 0:
                        summary_parts.append(f"and partially meets {actual_partially_met_count} requirements")
                    else:
                        summary_parts.append("and does not partially meet any requirements")
                    
                    # Add critical gaps if any
                    if actual_missing_count > 0:
                        # Extract key missing requirements (limit to 3-4 most important)
                        critical_gaps = []
                        for req in requirements_missing_list[:4]:
                            # Extract requirement name (before status)
                            req_name = req.split("(")[0].strip() if "(" in req else req.strip()
                            # Skip generic requirements
                            if req_name and len(req_name) > 5 and req_name.lower() not in ["experience", "education"]:
                                critical_gaps.append(req_name)
                        
                        if critical_gaps:
                            gaps_str = ", ".join(critical_gaps)
                            summary_parts.append(f"Critical gaps: {gaps_str}")
                else:
                    summary_parts.append("Match analysis completed.")
                
                summary_text = ". ".join(summary_parts) + "."
                
                # Trim summary if too long
                if len(summary_text) > 500:
                    summary_text = summary_text[:497] + "..."
                
                scored_jobs.append({
                    "job": job,
                    "match_score": score,
                    "key_matches": data_result.get("key_matches", []) or [],
                    "requirements_met": requirements_met,
                    "total_requirements": total_requirements,
                    "requirements_satisfied": requirements_satisfied_list,
                    "requirements_partially_met": requirements_partially_met_list,
                    "requirements_missing": requirements_missing_list,
                    "improvements_needed": data_result.get("improvements_needed", []) or [],
                    "reasoning": data_result.get("reasoning", "Score calculated based on candidate-job alignment"),
                    "summary": summary_text,
                })
                
                # Timing: End job scoring
                job_scoring_duration = time.perf_counter() - job_scoring_start
                yield format_sse_event("timing", {
                    "step": "job_scoring",
                    "job_index": idx,
                    "seconds": round(job_scoring_duration, 3)
                })
                await asyncio.sleep(0)  # Force flush
                
                # Event 6: Job scored - send complete job data as job_scored event
                logger.info(f"Yielding job_scored event for job {idx}: score={score}")
                
                # Use the summary already generated above, ensure it's not too long
                if len(summary_text) > 500:
                    summary_text = summary_text[:497] + "..."
                key_matches = data_result.get("key_matches", []) or []
                
                # Write job scoring result to file
                if response_file:
                    scoring_result = {
                        "job_index": idx,
                        "job_title": job.job_title,
                        "company": job.company,
                        "match_score": score,
                        "key_matches": key_matches[:5],
                        "requirements_met": requirements_met,
                        "total_requirements": total_requirements,
                        "requirements_satisfied": requirements_satisfied_list,
                        "requirements_partially_met": requirements_partially_met_list,
                        "requirements_missing": requirements_missing_list,
                        "improvements_needed": data_result.get("improvements_needed", []) or [],
                        "summary": summary_text,
                        "reasoning": reasoning
                    }
                    response_file.write(f"=== JOB SCORING RESULT (Job {idx}) ===\n")
                    response_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    response_file.write(f"{'='*60}\n\n")
                    response_file.write(json.dumps(scoring_result, indent=2, ensure_ascii=False))
                    response_file.write(f"\n\n")
                    response_file.flush()
                    logger.info(f"Job scoring result written to file for job {idx}")
                
                # Stream complete job scoring result with all data
                yield format_sse_event("job_scored", {
                    "rank": idx,
                    "job_url": str(job.url),
                    "job_title": job.job_title or "Unknown",
                    "company": job.company or "Unknown",
                    "match_score": round(score, 3),
                    "summary": summary_text,
                    "key_matches": key_matches,  # Send all key matches, not just first 5
                    "requirements_satisfied": requirements_satisfied_list,
                    "requirements_partially_met": requirements_partially_met_list,
                    "requirements_missing": requirements_missing_list,
                    "improvements_needed": data_result.get("improvements_needed", []) or [],
                    "location": None,  # Location extraction happens later if needed
                    "scraped_summary": None
                })
                await asyncio.sleep(0)  # Force flush - CRITICAL for SSE
                logger.info(f"job_scored event with full data yielded successfully for job {idx}")
            
            # Sort by score (descending)
            scored_jobs.sort(key=lambda x: x["match_score"], reverse=True)
            # Return all scored jobs (no score threshold filter)
            top_matches = scored_jobs[:10]  # Limit to top 10, but no minimum score requirement
            
            if not top_matches:
                yield format_sse_event("error", {
                    "message": "No jobs to process"
                })
                await asyncio.sleep(0)  # Force flush
                return
            
            # Create matched jobs (simplified summarization)
            matched_jobs_list = []
            for idx, entry in enumerate(top_matches, 1):
                job = entry["job"]
                score = entry["match_score"]
                
                # Use LLM-generated summary if available, otherwise create fallback
                summary_text = entry.get("summary", "")
                if not summary_text:
                    # Fallback: create summary from reasoning
                    summary_parts = []
                    if entry.get("reasoning"):
                        summary_parts.append(entry["reasoning"][:200])
                    if entry.get("key_matches"):
                        matches_str = ', '.join(entry["key_matches"][:3])
                        summary_parts.append(f"Key matches: {matches_str}.")
                    summary_text = " ".join(summary_parts) if summary_parts else "Match analysis completed."
                if len(summary_text) > 500:
                    summary_text = summary_text[:497] + "..."
                
                matched_job = {
                    "rank": idx,
                    "job_url": str(job.url),
                    "job_title": job.job_title or "Unknown",
                    "company": job.company or "Unknown",
                    "match_score": round(score, 3),
                    "summary": summary_text,
                    "key_matches": entry["key_matches"],
                    "requirements_satisfied": entry["requirements_satisfied"],
                    "requirements_partially_met": entry.get("requirements_partially_met", []),
                    "requirements_missing": entry["requirements_missing"],
                    "improvements_needed": entry["improvements_needed"],
                    "location": None,
                    "scraped_summary": None,
                    "job_description": job.job_description if hasattr(job, 'job_description') else None  # Internal use only, NOT in API response
                }
                matched_jobs_list.append(matched_job)
            
            # Check sponsorship
            sponsorship_info = None
            # Always check for sponsorship mentions and SC clearance in job description
            from sponsorship_checker import check_sponsorship, check_sponsorship_in_job_description, check_sc_clearance_requirement
            job_description = jobs[0].description if jobs and jobs[0].description else None
            sponsorship_mentioned = check_sponsorship_in_job_description(job_description)
            sc_clearance_required = check_sc_clearance_requirement(job_description)
            
            if jobs and jobs[0].company:
                logger.info(f"Checking sponsorship for company: {jobs[0].company}")
                
                # Timing: Start sponsorship check
                sponsorship_check_start = time.perf_counter()
                
                cleaned_name = clean_company_name(jobs[0].company)
                if cleaned_name:
                    try:
                        sponsorship_result = await asyncio.to_thread(
                            check_sponsorship, cleaned_name, jobs[0].description, openai_key
                        )
                        if sponsorship_result:
                            sponsorship_info = {
                                "company_name": sponsorship_result.get("company_name"),
                                "sponsors_workers": sponsorship_result.get("sponsors_workers", False),
                                "visa_types": sponsorship_result.get("visa_types"),
                                "summary": sponsorship_result.get("summary", "No sponsorship information available"),
                                "sponsorship_mentioned_in_job": sponsorship_result.get("sponsorship_mentioned_in_job"),
                                "sc_clearance_required": sponsorship_result.get("sc_clearance_required")
                            }
                            
                            # Timing: End sponsorship check
                            sponsorship_check_duration = time.perf_counter() - sponsorship_check_start
                            yield format_sse_event("timing", {
                                "step": "sponsorship_check",
                                "seconds": round(sponsorship_check_duration, 3)
                            })
                            await asyncio.sleep(0)  # Force flush
                            
                            # Event 8: Sponsorship result - stream complete sponsorship data
                            logger.info(f"Yielding sponsorship_checked event for company: {sponsorship_info.get('company_name')}")
                            
                            # Write sponsorship check result to file
                            if response_file:
                                response_file.write(f"=== SPONSORSHIP CHECK RESULT ===\n")
                                response_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                                response_file.write(f"{'='*60}\n\n")
                                response_file.write(json.dumps(sponsorship_info, indent=2, ensure_ascii=False))
                                response_file.write(f"\n\n")
                                response_file.flush()
                                logger.info("Sponsorship check result written to file")
                            
                            # Stream complete sponsorship result with all data
                            yield format_sse_event("sponsorship_checked", {
                                "company_name": sponsorship_info.get("company_name"),
                                "sponsors_workers": sponsorship_info.get("sponsors_workers", False),
                                "visa_types": sponsorship_info.get("visa_types"),
                                "summary": sponsorship_info.get("summary", "No sponsorship information available"),
                                "sponsorship_mentioned_in_job": sponsorship_info.get("sponsorship_mentioned_in_job"),
                                "sc_clearance_required": sponsorship_info.get("sc_clearance_required")
                            })
                            await asyncio.sleep(0)  # Force flush - CRITICAL for SSE
                            logger.info("sponsorship_checked event with full data yielded successfully")
                    except Exception as e:
                        logger.error(f"Sponsorship check failed: {e}", exc_info=True)
                        logger.info(f"Yielding sponsorship_checked event (error case) for company: {cleaned_name}")
                        
                        # Use already checked sponsorship and SC clearance values
                        
                        summary_parts = [f"Error checking sponsorship: {str(e)}"]
                        if sponsorship_mentioned['mentioned']:
                            summary_parts.append('Sponsorship details are mentioned in the job description.')
                        else:
                            summary_parts.append('Sponsorship details are not mentioned in the job description.')
                        if sc_clearance_required['required']:
                            summary_parts.append('SC clearance is required.')
                        
                        # Write sponsorship error to file
                        if response_file:
                            error_sponsorship_info = {
                                "company_name": cleaned_name,
                                "sponsors_workers": False,
                                "visa_types": None,
                                "summary": ' '.join(summary_parts),
                                "sponsorship_mentioned_in_job": sponsorship_mentioned['mentioned'],
                                "sc_clearance_required": sc_clearance_required['required']
                            }
                            response_file.write(f"=== SPONSORSHIP CHECK RESULT (ERROR) ===\n")
                            response_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            response_file.write(f"{'='*60}\n\n")
                            response_file.write(json.dumps(error_sponsorship_info, indent=2, ensure_ascii=False))
                            response_file.write(f"\n\n")
                            response_file.flush()
                            logger.info("Sponsorship check error written to file")
                        
                        # Stream sponsorship error result with full data structure
                        error_sponsorship_data = {
                            "company_name": cleaned_name,
                            "sponsors_workers": False,
                            "visa_types": None,
                            "summary": ' '.join(summary_parts),
                            "sponsorship_mentioned_in_job": sponsorship_mentioned['mentioned'],
                            "sc_clearance_required": sc_clearance_required['required']
                        }
                        yield format_sse_event("sponsorship_checked", error_sponsorship_data)
                        await asyncio.sleep(0)  # Force flush - CRITICAL for SSE
                        logger.info("sponsorship_checked event (error case) with full data yielded successfully")
            else:
                # No company found, but still check for sponsorship mentions and SC clearance
                if job_description:
                    summary_parts = []
                    if sponsorship_mentioned['mentioned']:
                        summary_parts.append('Sponsorship details are mentioned in the job description.')
                    else:
                        summary_parts.append('Sponsorship details are not mentioned in the job description.')
                    if sc_clearance_required['required']:
                        summary_parts.append('SC clearance is required.')
                    
                    if summary_parts:
                        sponsorship_info = {
                            "company_name": None,
                            "sponsors_workers": False,
                            "visa_types": None,
                            "summary": ' '.join(summary_parts),
                            "sponsorship_mentioned_in_job": sponsorship_mentioned['mentioned'],
                            "sc_clearance_required": sc_clearance_required['required']
                        }
            
            # Event 9: Summary generation (stream summaries if needed)
            # For now, summaries are already created, but we can stream them if needed
            
            # Save job applications to Firebase (background task)
            if user_id and matched_jobs_list:
                try:
                    logger.info(f"[STREAM] Preparing to save {len(matched_jobs_list)} jobs to Firebase for user {user_id}")
                    from job_extractor import extract_jobs_from_response
                    api_response_format = {
                        "matched_jobs": matched_jobs_list
                    }
                    logger.info(f"[STREAM] Extracting jobs from response format...")
                    jobs_to_save = extract_jobs_from_response(api_response_format)
                    logger.info(f"[STREAM] Extracted {len(jobs_to_save)} jobs to save")
                    
                    if jobs_to_save:
                        logger.info(f"[STREAM] Scheduling background save of {len(jobs_to_save)} job applications for user {user_id}")
                        logger.info(f"[STREAM] First job sample: {jobs_to_save[0] if jobs_to_save else 'N/A'}")
                        # Use BackgroundTasks only (removed duplicate asyncio.create_task to prevent double saves)
                        background_tasks.add_task(
                            save_job_applications_background,
                            user_id,
                            jobs_to_save
                        )
                        logger.info(f"[STREAM] ✓ Background task scheduled successfully")
                    else:
                        logger.warning(f"[STREAM] No job applications extracted to save (matched_jobs_list had {len(matched_jobs_list)} jobs)")
                except Exception as e:
                    logger.error(f"[STREAM] Error preparing background save: {e}", exc_info=True)
                    import traceback
                    logger.error(f"[STREAM] Full traceback: {traceback.format_exc()}")
                    # Non-fatal - continue with response
            
            # Save sponsorship info to Firebase (background task)
            if sponsorship_info and user_id:
                try:
                    logger.info(f"[STREAM] Preparing to save sponsorship info to Firebase for user {user_id}")
                    sponsorship_dict = {
                        "company_name": sponsorship_info.get("company_name"),
                        "sponsors_workers": sponsorship_info.get("sponsors_workers", False),
                        "visa_types": sponsorship_info.get("visa_types"),
                        "summary": sponsorship_info.get("summary", "No sponsorship information available")
                    }
                    
                    job_info = None
                    if matched_jobs_list and len(matched_jobs_list) > 0:
                        top_job = matched_jobs_list[0]
                        portal = "Unknown"
                        job_url_str = str(top_job.get("job_url", ""))
                        if "linkedin.com" in job_url_str.lower():
                            portal = "LinkedIn"
                        elif "indeed.com" in job_url_str.lower():
                            portal = "Indeed"
                        elif "glassdoor.com" in job_url_str.lower():
                            portal = "Glassdoor"
                        
                        job_info = {
                            "job_title": top_job.get("job_title"),
                            "job_url": job_url_str,
                            "company": top_job.get("company"),
                            "portal": portal
                        }
                    
                    logger.info(f"[STREAM] Scheduling background save of sponsorship info for {sponsorship_dict.get('company_name')}")
                    logger.info(f"[STREAM] Sponsorship data: {sponsorship_dict}")
                    # Use BackgroundTasks only (removed duplicate asyncio.create_task to prevent double saves)
                    background_tasks.add_task(
                        save_sponsorship_info_background,
                        user_id,
                        request_id,
                        sponsorship_dict,
                        job_info
                    )
                    logger.info(f"[STREAM] ✓ Sponsorship background task scheduled successfully")
                except Exception as e:
                    logger.error(f"[STREAM] Error scheduling sponsorship save: {e}", exc_info=True)
                    import traceback
                    logger.error(f"[STREAM] Full traceback: {traceback.format_exc()}")
            
            # Event 10: Complete
            processing_time = f"{time.time() - start_time:.1f}s"
            logger.info(f"Preparing complete event. Matched jobs: {len(matched_jobs_list)}, Processing time: {processing_time}")
            
            # Filter out job_description from API response (internal use only)
            # Create clean matched_jobs list without job_description field
            clean_matched_jobs = []
            for job in matched_jobs_list:
                clean_job = {k: v for k, v in job.items() if k != "job_description"}
                clean_matched_jobs.append(clean_job)
            
            # Remove total_years_experience and all float year fields from final response
            fields_to_remove = [
                "total_years_experience",
                "full_time_years",
                "internship_years",
                "freelance_years",
                "part_time_years",
                "contract_years",
                "academic_years",
                "project_years",
                "total_weighted_years",
                "experience_breakdown"
            ]
            candidate_profile_final = {k: v for k, v in candidate_profile.items() if k not in fields_to_remove}
            
            final_response = {
                "candidate_profile": candidate_profile_final,
                "matched_jobs": clean_matched_jobs,  # job_description excluded from API response
                "processing_time": processing_time,
                "jobs_analyzed": len(jobs),
                "request_id": request_id,
                "sponsorship": sponsorship_info
            }
            
            logger.info("Yielding complete event")
            final_response_data = {
                "candidate_profile": candidate_profile_final,
                "matched_jobs": clean_matched_jobs,  # job_description excluded from API response
                "processing_time": processing_time,
                "jobs_analyzed": len(jobs),
                "request_id": request_id,
                "sponsorship": sponsorship_info
            }
            
            # Write complete response to file
            if response_file:
                response_file.write(f"\n{'='*60}\n")
                response_file.write(f"=== COMPLETE RESPONSE ===\n")
                response_file.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                response_file.write(f"Processing Time: {processing_time}\n")
                response_file.write(f"{'='*60}\n\n")
                response_file.write(json.dumps(final_response_data, indent=2, ensure_ascii=False))
                response_file.write(f"\n\n{'='*60}\n")
                response_file.write(f"Response saved to: {response_file_path}\n")
                response_file.flush()
                response_file.close()
                logger.info(f"Complete response saved to: {response_file_path}")
            
            yield format_sse_event("complete", {
                "response": final_response_data
            })
            await asyncio.sleep(0)  # Force flush - CRITICAL for SSE
            logger.info("complete event yielded successfully")
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            logger.error(f"Streaming error: {error_msg}", exc_info=True)
            
            # Write error to file
            if response_file:
                response_file.write(f"\n{'='*60}\n")
                response_file.write(f"=== ERROR ===\n")
                response_file.write(f"Error: {error_msg}\n")
                response_file.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                response_file.write(f"{'='*60}\n")
                response_file.flush()
                response_file.close()
                logger.info(f"Error response saved to: {response_file_path}")
            
            yield format_sse_event("error", {
                "message": error_msg,
                "error": str(e)
            })
            await asyncio.sleep(0)  # Force flush
        finally:
            # Ensure file is closed
            if response_file and not response_file.closed:
                response_file.close()
    
    # Create response with explicit flush settings
    response = StreamingResponse(
        generate_stream(resume_bytes_from_file),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Content-Type": "text/event-stream; charset=utf-8",
        }
    )
    
    # Ensure response is not buffered
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Cache-Control"] = "no-cache, no-transform"
    
    return response

@app.get("/")
async def root():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/cache/clear")
async def clear_requirements_cache():
    """
    Clear the frozen requirements cache.
    
    This will force all subsequent job requirement extractions to run fresh
    (useful when you've updated the extraction prompt).
    
    Returns:
        Dict with cache stats before clearing
    """
    try:
        from requirement_extractor import clear_cache, get_cache_stats
        
        # Get stats before clearing
        stats_before = get_cache_stats()
        
        # Clear the cache
        clear_cache()
        
        return {
            "status": "success",
            "message": "Requirements cache cleared",
            "cleared_entries": stats_before["cache_size"],
            "stats_before": stats_before
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@app.get("/api/cache/stats")
async def get_requirements_cache_stats():
    """
    Get statistics about the frozen requirements cache.
    
    Returns:
        Dict with cache size, max size, TTL, expired entries, and memory usage
    """
    try:
        from requirement_extractor import get_cache_stats
        stats = get_cache_stats()
        return {
            "status": "success",
            **stats
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache stats: {str(e)}"
        )


# Firebase Resume Endpoints
@app.post("/api/firebase/resumes", response_model=FirebaseResumeListResponse)
async def get_user_resumes(request: GetUserResumesRequest):
    """
    Fetch all resumes for a specific user from Firebase Firestore.
    
    Request Body:
        user_id: The user ID
        
    Returns:
        List of resume documents
    """
    try:
        from firebase_service import get_firebase_service
        firebase_service = get_firebase_service()
        resumes_data = firebase_service.get_user_resumes(request.user_id)
        
        resumes = [FirebaseResume(**resume) for resume in resumes_data]
        return FirebaseResumeListResponse(
            user_id=request.user_id,
            resumes=resumes,
            count=len(resumes)
        )
    except ValidationError as ve:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request data: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Error fetching resumes: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch resumes: {str(e)}"
        )


@app.post("/api/firebase/resumes/get", response_model=FirebaseResumeResponse)
async def get_user_resume(request: GetUserResumeRequest):
    """
    Fetch a specific resume by ID for a user from Firebase Firestore.
    
    Request Body:
        user_id: The user ID
        resume_id: The resume document ID
        
    Returns:
        The resume document
    """
    try:
        from firebase_service import get_firebase_service
        firebase_service = get_firebase_service()
        resume_data = firebase_service.get_resume_by_id(request.user_id, request.resume_id)
        
        if not resume_data:
            raise HTTPException(
                status_code=404,
                detail=f"Resume {request.resume_id} not found for user {request.user_id}"
            )
        
        return FirebaseResumeResponse(
            user_id=request.user_id,
            resume=FirebaseResume(**resume_data)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching resume: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch resume: {str(e)}"
        )


@app.post("/api/firebase/resumes/pdf")
async def get_user_resume_pdf(request: GetUserResumePdfRequest):
    """
    Fetch a resume PDF file as bytes.
    
    Request Body:
        user_id: The user ID
        resume_id: The resume document ID
        
    Returns:
        PDF file as binary response
    """
    try:
        from firebase_service import get_firebase_service
        firebase_service = get_firebase_service()
        pdf_bytes = firebase_service.get_resume_pdf_bytes(request.user_id, request.resume_id)
        
        if not pdf_bytes:
            raise HTTPException(
                status_code=404,
                detail=f"Resume PDF not found for user {request.user_id}, resume {request.resume_id}"
            )
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="resume_{request.resume_id}.pdf"'
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching resume PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch resume PDF: {str(e)}"
        )


@app.post("/api/firebase/resumes/base64")
async def get_user_resume_base64(request: GetUserResumeBase64Request):
    """
    Fetch a resume as base64 encoded string.
    
    Request Body:
        user_id: The user ID
        resume_id: The resume document ID
        
    Returns:
        JSON with base64 encoded PDF content
    """
    try:
        from firebase_service import get_firebase_service
        firebase_service = get_firebase_service()
        resume_data = firebase_service.get_resume_by_id(request.user_id, request.resume_id)
        
        if not resume_data:
            raise HTTPException(
                status_code=404,
                detail=f"Resume {request.resume_id} not found for user {request.user_id}"
            )
        
        base64_content = firebase_service.extract_pdf_base64(resume_data)
        
        if not base64_content:
            raise HTTPException(
                status_code=404,
                detail=f"Resume PDF content not found for user {request.user_id}, resume {request.resume_id}"
            )
        
        return JSONResponse({
            "user_id": request.user_id,
            "resume_id": request.resume_id,
            "base64_content": base64_content
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching resume base64: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch resume base64: {str(e)}"
        )


@app.post("/api/firebase/users/saved-cvs", response_model=SavedCVResponse)
async def get_user_saved_cvs(request: GetUserSavedCvsRequest):
    """
    Fetch savedCVs array for a user from Firebase Firestore.
    
    Request Body:
        user_id: The user ID
        
    Returns:
        SavedCVs array
    """
    try:
        from firebase_service import get_firebase_service
        firebase_service = get_firebase_service()
        saved_cvs = firebase_service.get_user_saved_cvs(request.user_id)
        
        return SavedCVResponse(
            user_id=request.user_id,
            saved_cvs=saved_cvs,
            count=len(saved_cvs)
        )
    except Exception as e:
        logger.error(f"Error fetching saved CVs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch saved CVs: {str(e)}"
        )


@app.post("/api/check-sponsorship", response_model=SponsorshipInfo)
async def check_sponsorship_endpoint(
    request: SponsorshipCheckRequest,
    settings: Settings = Depends(get_settings),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Check if a company sponsors workers for UK visas.
    
    This endpoint uses the EXACT SAME process as the match-jobs endpoint:
    1. Receives job_info (scraped job data)
    2. Pre-extracts company name using multiple strategies
    3. Uses LLM agent (summarize_scraped_data) to extract structured info including company_name
    4. Checks UK visa sponsorship database using fuzzy matching
    5. Uses AI agent to select correct company match
    6. Optionally fetches additional company info from web
    7. Builds enhanced summary combining CSV and web data
    
    Args:
        request: SponsorshipCheckRequest with job_info (scraped job data)
        settings: Application settings
    
    Returns:
        SponsorshipInfo with sponsorship details (same format as match-jobs endpoint)
    """
    try:
        logger.info("Checking company sponsorship status")
        
        # Get job_info (scraped job data) - same as match-jobs endpoint
        job_data = request.job_info
        
        if not job_data:
            return SponsorshipInfo(
                company_name=None,
                sponsors_workers=False,
                visa_types=None,
                summary="job_info field is required and cannot be empty.",
            )
        
        # STEP 1: Extract company name using OpenAI with timeout
        openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return SponsorshipInfo(
                company_name=None,
                sponsors_workers=False,
                visa_types=None,
                summary="OpenAI API key is required for company name extraction.",
            )
        
        logger.debug(f"Extracting company name from {len(job_data)} chars via OpenAI")
        
        # Use the same extraction function as match-jobs with timeout
        from job_extractor import extract_company_and_title_from_raw_data
        
        try:
            extracted_info = await asyncio.wait_for(
                asyncio.to_thread(
                    extract_company_and_title_from_raw_data,
                    job_data,  # Send raw scraped data as-is, no preprocessing
                    openai_key,
                    "gpt-4o-mini"  # Use fast and intelligent OpenAI model
                ),
                timeout=10.0  # 10 second timeout for company extraction
            )
            
            # Get extracted company name
            extracted_company = extracted_info.get("company_name")
            logger.debug(f"OpenAI extracted company: {extracted_company[:50] if extracted_company else 'None'}")
        except asyncio.TimeoutError:
            logger.warning("Company name extraction timed out, trying fallback methods")
            extracted_company = None
        except Exception as e:
            logger.warning(f"OpenAI extraction failed: {e}, trying fallback methods")
            extracted_company = None
        
        # STEP 2: Clean and validate company name (same as match-jobs)
        final_company = None
        if extracted_company:
            cleaned = clean_company_name(extracted_company)
            if cleaned and len(cleaned) >= 2 and cleaned.lower() not in ["not specified", "unknown", "none"]:
                final_company = cleaned
                logger.debug(f"Using OpenAI-extracted company: {final_company}")
        
        # Fallback: Try sponsorship_checker extract_company_name if OpenAI failed
        if not final_company:
            try:
                from sponsorship_checker import extract_company_name
                extracted = extract_company_name(job_data[:2000] if job_data else "")
                if extracted:
                    cleaned = clean_company_name(extracted)
                    if cleaned and len(cleaned) >= 2:
                        final_company = cleaned
                        logger.debug(f"Using fallback extracted company: {final_company}")
            except Exception as e:
                logger.debug(f"Fallback extraction error: {e}")
        
        if not final_company or final_company == "Company name not available in posting":
            return SponsorshipInfo(
                company_name=None,
                sponsors_workers=False,
                visa_types=None,
                summary="Company name could not be extracted from the provided job_info. The LLM agent was unable to identify a company name in the job posting data.",
            )
        
        # STEP 3: Check sponsorship using cached CSV data (same as match-jobs)
        from sponsorship_checker import check_sponsorship
        
        logger.info(f"Checking sponsorship for company: {final_company}")
        
        # Use async thread pool for check_sponsorship with timeout
        try:
            sponsorship_result = await asyncio.wait_for(
                asyncio.to_thread(
                    check_sponsorship,
                    final_company,
                    job_data,
                    openai_key
                ),
                timeout=15.0  # 15 second timeout for sponsorship check
            )
        except asyncio.TimeoutError:
            logger.warning(f"Sponsorship check timed out for {final_company}")
            return SponsorshipInfo(
                company_name=final_company,
                sponsors_workers=False,
                visa_types=None,
                summary=f"Sponsorship check timed out. Please try again or check the company name: {final_company}",
            )
        
        # Build summary from CSV sponsorship data (fast response)
        # Web search removed for faster performance - CSV data is the authoritative source
        base_summary = sponsorship_result.get('summary', 'No sponsorship information available')
        # Clean and normalize the summary
        enhanced_summary = clean_summary_text(base_summary)
        
        # Return SponsorshipInfo with CSV-based sponsorship data
        return SponsorshipInfo(
            company_name=sponsorship_result.get('company_name'),
            sponsors_workers=sponsorship_result.get('sponsors_workers', False),
            visa_types=sponsorship_result.get('visa_types'),
            summary=enhanced_summary,
            sponsorship_mentioned_in_job=sponsorship_result.get('sponsorship_mentioned_in_job'),
            sc_clearance_required=sponsorship_result.get('sc_clearance_required')
        )
        
    except FileNotFoundError as e:
        logger.error(f"Sponsorship database not available: {e}")
        return SponsorshipInfo(
            company_name=None,
            sponsors_workers=False,
            visa_types=None,
            summary=f"Sponsorship database not available: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error checking sponsorship: {e}", exc_info=True)
        return SponsorshipInfo(
            company_name=None,
            sponsors_workers=False,
            visa_types=None,
            summary=f"Error checking sponsorship: {str(e)}",
        )
