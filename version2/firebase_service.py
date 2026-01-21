"""
Firebase service for fetching resumes from Firestore.
"""
from __future__ import annotations

import os
import base64
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Setup logging with environment variable control
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    from google.cloud.firestore_v1 import FieldFilter
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    FieldFilter = None

# Load environment variables
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# CRITICAL: Also explicitly load from system environment
# This ensures GOOGLE_APPLICATION_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS is available
import os as _os

# Check for JSON string in environment variable (preferred for production)
if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in _os.environ:
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS_JSON", _os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    json_value = _os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    # Show first/last 50 chars for security (don't print full JSON)
    if len(json_value) > 100:
        preview = json_value[:50] + "..." + json_value[-50:]
    else:
        preview = json_value[:50] + "..."
    logger.debug(f"GOOGLE_APPLICATION_CREDENTIALS_JSON found (length: {len(json_value)})")
elif "GOOGLE_APPLICATION_CREDENTIALS" in _os.environ:
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", _os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    logger.debug(f"GOOGLE_APPLICATION_CREDENTIALS found: {_os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
else:
    logger.warning("Neither GOOGLE_APPLICATION_CREDENTIALS_JSON nor GOOGLE_APPLICATION_CREDENTIALS found")


class FirebaseService:
    """Service for interacting with Firebase Firestore to fetch resumes."""
    
    _app = None
    _db = None
    
    def __init__(self):
        """Initialize Firebase Admin SDK."""
        if not FIREBASE_AVAILABLE:
            raise ImportError(
                "firebase-admin is not installed. Install it with: pip install firebase-admin"
            )
        
        # Initialize Firebase Admin SDK if not already initialized
        if FirebaseService._app is None:
            self._initialize_firebase()
        
        # Initialize Firestore client if not already initialized
        if FirebaseService._db is None:
            FirebaseService._db = firestore.client()
            logger.info("Firebase initialized")
    
    def _initialize_firebase(self):
        """
        Initialize Firebase Admin SDK with credentials from environment variables.
        
        Priority:
        1. GOOGLE_APPLICATION_CREDENTIALS_JSON (JSON string directly in env var)
        2. GOOGLE_APPLICATION_CREDENTIALS (file path to JSON file)
        3. FIREBASE_PROJECT_ID (for Application Default Credentials)
        """
        try:
            # Try to initialize Firebase Admin SDK
            try:
                # First, check if Firebase is already initialized
                FirebaseService._app = firebase_admin.get_app()
                logger.debug("Firebase already initialized")
            except ValueError:
                # Not initialized yet
                logger.debug("Initializing Firebase...")
                
                # Load environment variables from .env if present
                from dotenv import load_dotenv
                load_dotenv()
                
                # METHOD 1: Try GOOGLE_APPLICATION_CREDENTIALS_JSON (JSON string in env var)
                # This is preferred for production/deployment (e.g., Render, Heroku, etc.)
                firebase_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                
                if firebase_json:
                    logger.debug("Found GOOGLE_APPLICATION_CREDENTIALS_JSON")
                    try:
                        # Convert string to dict
                        cred_dict = json.loads(firebase_json)
                        # Initialize Firebase Admin SDK with JSON dict
                        cred = credentials.Certificate(cred_dict)
                        FirebaseService._app = firebase_admin.initialize_app(cred)
                        logger.info("Firebase initialized from JSON")
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON: {str(e)}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to initialize Firebase from JSON: {str(e)}")
                
                # METHOD 2: Try GOOGLE_APPLICATION_CREDENTIALS (file path)
                # Fallback for local development
                elif not FirebaseService._app:
                    logger.debug("Trying GOOGLE_APPLICATION_CREDENTIALS file path...")
                    
                    # Try multiple methods to get GOOGLE_APPLICATION_CREDENTIALS
                    # 1. Try os.getenv (from dotenv)
                    service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                    
                    # 2. Try os.environ directly
                    if not service_account_path:
                        service_account_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                    
                    # 3. Try system environment (Windows)
                    if not service_account_path:
                        try:
                            import subprocess
                            result = subprocess.run(
                                ['powershell', '-Command', '[Environment]::GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", "User")'],
                                capture_output=True,
                                text=True,
                                timeout=2
                            )
                            if result.returncode == 0 and result.stdout.strip():
                                service_account_path = result.stdout.strip()
                                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
                        except:
                            pass
                    
                    print(f"[Firebase] Checking GOOGLE_APPLICATION_CREDENTIALS: {service_account_path}")
                    
                    if service_account_path:
                        # Handle Windows paths - normalize separators
                        # Try different path formats (same as test_firebase_simple.py)
                        paths_to_try = [
                            service_account_path,  # Original
                            service_account_path.replace('/', '\\'),  # Windows backslash
                            service_account_path.replace('\\', '/'),  # Forward slash
                            os.path.normpath(service_account_path),  # Normalized
                            os.path.abspath(service_account_path),  # Absolute
                        ]
                        
                        path_to_use = None
                        for path in paths_to_try:
                            if os.path.exists(path):
                                path_to_use = path
                                break
                        
                        if path_to_use:
                            print(f"[Firebase] Found service account file: {path_to_use}")
                            cred = credentials.Certificate(path_to_use)
                            FirebaseService._app = firebase_admin.initialize_app(cred)
                            print("[Firebase] [OK] Firebase initialized successfully from file")
                        else:
                            raise FileNotFoundError(
                                f"Service account file not found at any of these paths:\n" +
                                "\n".join(f"  - {p}" for p in paths_to_try)
                            )
                
                # METHOD 3: Try with project ID from environment (Application Default Credentials)
                if not FirebaseService._app:
                    project_id = os.getenv("VITE_FIREBASE_PROJECT_ID") or os.getenv("FIREBASE_PROJECT_ID")
                    if not project_id:
                        raise ValueError(
                            "No Firebase credentials found. Please set one of:\n"
                            "  - GOOGLE_APPLICATION_CREDENTIALS_JSON (JSON string)\n"
                            "  - GOOGLE_APPLICATION_CREDENTIALS (file path)\n"
                            "  - FIREBASE_PROJECT_ID (for Application Default Credentials)"
                        )
                    
                    print(f"[Firebase] Using project ID: {project_id}")
                    # Try to initialize with project ID (uses Application Default Credentials)
                    FirebaseService._app = firebase_admin.initialize_app(options={'projectId': project_id})
                    print("[Firebase] [OK] Firebase initialized successfully with project ID")
                
        except Exception as e:
            if isinstance(e, (RuntimeError, ValueError, FileNotFoundError)):
                raise
            raise RuntimeError(f"Failed to initialize Firebase: {str(e)}")
    
    def get_user_resumes(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all resumes for a specific user from Firestore.
        
        Args:
            user_id: The user ID to fetch resumes for
            
        Returns:
            List of resume dictionaries containing resume data
        """
        try:
            # Reference to the resumes collection for this user
            resumes_ref = FirebaseService._db.collection("users").document(user_id).collection("resumes")
            
            # Fetch all resumes
            resumes_docs = resumes_ref.stream()
            
            resumes = []
            for doc in resumes_docs:
                resume_data = doc.to_dict()
                resume_data["id"] = doc.id  # Add document ID
                resumes.append(resume_data)
            
            return resumes
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch resumes for user {user_id}: {str(e)}")
    
    def get_resume_by_id(self, user_id: str, resume_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific resume by ID for a user.
        
        Args:
            user_id: The user ID
            resume_id: The resume document ID
            
        Returns:
            Resume dictionary or None if not found
        """
        try:
            resume_ref = (
                FirebaseService._db
                .collection("users")
                .document(user_id)
                .collection("resumes")
                .document(resume_id)
            )
            
            resume_doc = resume_ref.get()
            
            if resume_doc.exists:
                resume_data = resume_doc.to_dict()
                resume_data["id"] = resume_doc.id
                return resume_data
            else:
                return None
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch resume {resume_id} for user {user_id}: {str(e)}"
            )
    
    def extract_pdf_base64(self, resume_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract base64 PDF content from resume data.
        
        Handles the PDF_BASE64: prefix format mentioned in the requirements.
        
        Args:
            resume_data: Dictionary containing resume data with content field
            
        Returns:
            Base64 string (without prefix) or None if not found
        """
        content = resume_data.get("content")
        
        if not content:
            return None
        
        # Handle content that starts with "PDF_BASE64:" prefix
        if isinstance(content, str):
            if content.startswith("PDF_BASE64:"):
                return content[len("PDF_BASE64:"):]
            # If already plain base64, return as is
            return content
        
        return None
    
    def get_resume_pdf_bytes(self, user_id: str, resume_id: str) -> Optional[bytes]:
        """
        Get resume PDF as bytes.
        
        Args:
            user_id: The user ID
            resume_id: The resume document ID
            
        Returns:
            PDF bytes or None if not found
        """
        resume_data = self.get_resume_by_id(user_id, resume_id)
        
        if not resume_data:
            return None
        
        base64_content = self.extract_pdf_base64(resume_data)
        
        if not base64_content:
            return None
        
        try:
            # Decode base64 to bytes
            return base64.b64decode(base64_content)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 PDF content: {str(e)}")
    
    def get_user_saved_cvs(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get saved CVs array for a user (as mentioned in the requirements).
        This might be stored at the user document level.
        
        Args:
            user_id: The user ID
            
        Returns:
            List of saved CV references/data
        """
        try:
            user_ref = FirebaseService._db.collection("users").document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                saved_cvs = user_data.get("savedCVs", [])
                return saved_cvs if isinstance(saved_cvs, list) else []
            else:
                return []
                
        except Exception as e:
            raise RuntimeError(f"Failed to fetch savedCVs for user {user_id}: {str(e)}")

    def _check_job_application_duplicate(self, user_id: str, job_data: Dict[str, Any]) -> Optional[str]:
        """
        Check if a job application already exists for this user.
        
        Args:
            user_id: The user ID
            job_data: Dictionary containing job application data
            
        Returns:
            Document ID of existing duplicate if found, None otherwise
        """
        try:
            db = FirebaseService._db
            if db is None:
                return None
            
            collection_ref = db.collection("users").document(user_id).collection("job_applications")
            
            # Extract fields to check for duplicates
            job_link = job_data.get("link", "").strip()
            company = job_data.get("company", "").strip()
            role = job_data.get("role", "").strip()
            
            # Strategy 1: Check by job URL (most reliable) - using filter keyword argument
            if job_link:
                if FieldFilter:
                    query = collection_ref.where(filter=FieldFilter("link", "==", job_link)).limit(1)
                else:
                    query = collection_ref.where("link", "==", job_link).limit(1)
                docs = query.stream()
                for doc in docs:
                    logger.debug(f"Duplicate found by URL: {doc.id}")
                    return doc.id
            
            # Strategy 2: Check by company + role (fallback if no URL) - using filter keyword argument
            if company and role:
                if FieldFilter:
                    query = collection_ref.where(filter=FieldFilter("company", "==", company)).where(filter=FieldFilter("role", "==", role)).limit(1)
                else:
                    query = collection_ref.where("company", "==", company).where("role", "==", role).limit(1)
                docs = query.stream()
                for doc in docs:
                    logger.debug(f"Duplicate found by company+role: {doc.id}")
                    return doc.id
            
            return None
            
        except Exception as e:
            logger.debug(f"Error checking for duplicate: {e}")
            # Don't fail on duplicate check errors, just log and continue
            return None

    def save_job_application(self, user_id: str, job_data: Dict[str, Any]) -> str:
        """
        Save a job application to Firestore.
        Uses the EXACT same approach as test_firebase_simple.py
        Checks for duplicates before saving. If a duplicate is found, the existing
        document is updated with the latest data (preserving original createdAt).
        
        Args:
            user_id: The user ID
            job_data: Dictionary containing job application data
            
        Returns:
            The document ID of the saved/updated application
        """
        try:
            # Ensure Firebase is initialized (trust singleton pattern - no repeated checks)
            if FirebaseService._app is None:
                logger.warning("Firebase app is None, re-initializing...")
                self._initialize_firebase()
            
            # Use Firestore client directly (trust initialization)
            if FirebaseService._db is None:
                FirebaseService._db = firestore.client()
            
            db = FirebaseService._db
            
            # Check for duplicates before saving
            existing_doc_id = self._check_job_application_duplicate(user_id, job_data)
            
            # Prepare document data with defaults
            # Use datetime.now() directly EXACTLY like test_firebase_simple.py (lines 88-98)
            document_data = {
                "appliedDate": job_data.get("appliedDate", datetime.now()),
                "company": job_data.get("company", ""),
                "interviewDate": job_data.get("interviewDate", ""),
                "summary": job_data.get("summary", ""),  # Match score summary
                "link": job_data.get("link", ""),
                "notes": job_data.get("notes", ""),
                "portal": job_data.get("portal", "Unknown"),
                "role": job_data.get("role", ""),
                "status": job_data.get("status", "Applied"),
                "visaRequired": job_data.get("visaRequired", "No"),
                "requirements_met": job_data.get("requirements_met", []),  # List of requirements that are met
                "requirements_not_met": job_data.get("requirements_not_met", []),  # List of requirements that are not met
                "improvements_needed": job_data.get("improvements_needed", []),  # List of suggested improvements
                "job_description": job_data.get("job_description")  # Cleaned job description (internal use only, NOT in API response)
            }
            
            if existing_doc_id:
                # Duplicate found - replace the existing document with latest data
                collection_ref = db.collection("users").document(user_id).collection("job_applications")
                doc_ref = collection_ref.document(existing_doc_id)
                
                # Get existing document to preserve createdAt if it exists
                existing_doc = doc_ref.get()
                if existing_doc.exists:
                    existing_data = existing_doc.to_dict()
                    if "createdAt" in existing_data:
                        document_data["createdAt"] = existing_data["createdAt"]
                    else:
                        document_data["createdAt"] = datetime.now()
                else:
                    document_data["createdAt"] = datetime.now()
                
                # Always set updatedAt to now when replacing
                document_data["updatedAt"] = datetime.now()
                
                # Replace the entire document with latest data (using set with merge=False)
                doc_ref.set(document_data)
                logger.info(f"✓ Replaced duplicate job with latest data: {job_data.get('role', 'N/A')} at {job_data.get('company', 'N/A')} (ID: {existing_doc_id})")
                return existing_doc_id
            
            # Add createdAt for new documents
            document_data["createdAt"] = datetime.now()
            
            logger.debug(f"Preparing document: {document_data.get('role', 'N/A')} at {document_data.get('company', 'N/A')}")
            print(f"[Firebase] [DEBUG] Full document_data: {json.dumps({k: str(v) if isinstance(v, datetime) else v for k, v in document_data.items()}, indent=2)}")
            
            # Reference to job_applications subcollection
            # EXACT same approach as test_firebase_simple.py line 83
            print(f"[Firebase] [DEBUG] Creating collection reference...")
            collection_ref = db.collection("users").document(user_id).collection("job_applications")
            print(f"[Firebase] [OK] Collection reference created: users/{user_id}/job_applications")
            print(f"[Firebase] [DEBUG] Collection ref type: {type(collection_ref)}")
            
            # Use add() method which returns (timestamp, document_reference)
            # EXACT same approach as test_firebase_simple.py line 109
            print(f"[Firebase] [SAVE] Calling collection_ref.add(document_data)...")
            print(f"[Firebase] [DEBUG] About to save document...")
            
            result = collection_ref.add(document_data)
            
            print(f"[Firebase] [DEBUG] add() returned: {result}")
            print(f"[Firebase] [DEBUG] Result type: {type(result)}")
            
            # Handle return value - EXACT same logic as test_firebase_simple.py (lines 112-117)
            if isinstance(result, tuple):
                update_time, doc_ref = result
                doc_id = doc_ref.id
            else:
                doc_ref = result
                doc_id = doc_ref.id if hasattr(doc_ref, 'id') else str(doc_ref)
            
            logger.info(f"✓ Saved job: {document_data.get('role', 'N/A')} at {document_data.get('company', 'N/A')} (ID: {doc_id})")
            return doc_id
                
        except Exception as e:
            error_msg = f"Failed to save job application for user {user_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def save_job_applications_batch(self, user_id: str, jobs_data: List[Dict[str, Any]]) -> List[str]:
        """
        Save multiple job applications to Firestore in a batch.
        Duplicates are automatically replaced with the latest data (existing documents are updated).
        
        Args:
            user_id: The user ID
            jobs_data: List of dictionaries containing job application data
            
        Returns:
            List of document IDs of saved/updated applications
        """
        try:
            logger.info(f"Saving {len(jobs_data)} jobs to Firebase")
            document_ids = []
            duplicates_count = 0
            new_count = 0
            
            for idx, job_data in enumerate(jobs_data, 1):
                try:
                    # Save directly - duplicate check happens inside save_job_application
                    # (removed duplicate check here to avoid checking twice - 10% faster)
                        doc_id = self.save_job_application(user_id, job_data)
                        document_ids.append(doc_id)
                    # Note: save_job_application handles duplicates internally
                        new_count += 1
                except Exception as job_error:
                    logger.error(f"Failed to save job {idx}: {str(job_error)}", exc_info=True)
                    # Continue with other jobs even if one fails
                    continue
            
            logger.info(f"✓ Saved: {new_count} jobs (duplicates handled internally)")
            return document_ids
            
        except Exception as e:
            logger.error(f"Batch save failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to save job applications batch for user {user_id}: {str(e)}")

    def _normalize_company_name(self, name: str) -> str:
        """
        Normalize company name for duplicate checking.
        Removes legal suffixes, extra whitespace, and converts to lowercase.
        
        Args:
            name: Company name to normalize
            
        Returns:
            Normalized company name
        """
        if not name or not isinstance(name, str):
            return ""
        
        import re
        
        # Remove quotes and extra whitespace
        normalized = name.strip().strip('"').strip("'").strip()
        
        # Convert to lowercase for case-insensitive comparison
        normalized = normalized.lower()
        
        # Remove common legal suffixes (case-insensitive)
        suffixes = [
            r'\s+inc\.?$', r'\s+incorporated$',
            r'\s+ltd\.?$', r'\s+limited$',
            r'\s+llc\.?$', r'\s+ll\.?c\.?$',
            r'\s+corp\.?$', r'\s+corporation$',
            r'\s+plc\.?$', r'\s+public limited company$',
            r'\s+llp\.?$', r'\s+limited liability partnership$',
            r'\s+p\.?c\.?$', r'\s+professional corporation$',
            r'\s+co\.?$', r'\s+company$',
            r'\s+group$', r'\s+holdings?$',
        ]
        
        for suffix_pattern in suffixes:
            normalized = re.sub(suffix_pattern, '', normalized, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _check_sponsorship_duplicate(self, user_id: str, company_name: str, request_id: Optional[str] = None) -> Optional[str]:
        """
        Check if a sponsorship check already exists for this user and company.
        Uses multiple strategies to catch duplicates even with name variations.
        
        Args:
            user_id: The user ID
            company_name: The company name
            request_id: Optional request ID to check (if provided, checks for exact match)
            
        Returns:
            Document ID of existing duplicate if found, None otherwise
        """
        try:
            db = FirebaseService._db
            if db is None:
                return None
                
            collection_ref = db.collection("sponsorship_checks").document(user_id).collection("checks")
            
            # Strategy 1: Check by request_id (most specific - same match-jobs request)
            if request_id:
                if FieldFilter:
                    query = collection_ref.where(filter=FieldFilter("requestId", "==", request_id)).limit(1)
                else:
                    query = collection_ref.where("requestId", "==", request_id).limit(1)
                docs = query.stream()
                for doc in docs:
                    logger.debug(f"Duplicate sponsorship found by request_id: {doc.id}")
                    return doc.id
            
            # Strategy 2: Check by exact company name match (case-sensitive) - using filter keyword argument
            if company_name and company_name.strip():
                company_name_clean = company_name.strip()
                if FieldFilter:
                    query = collection_ref.where(filter=FieldFilter("companyName", "==", company_name_clean)).limit(1)
                else:
                    query = collection_ref.where("companyName", "==", company_name_clean).limit(1)
                docs = query.stream()
                for doc in docs:
                    logger.debug(f"Duplicate sponsorship found by exact company name: {doc.id}")
                    return doc.id
            
            # Strategy 3 & 4: Check by normalized and case-insensitive company name (single pass)
            # Handles variations like "Google" vs "Google LLC", "Microsoft" vs "microsoft", etc.
            if company_name and company_name.strip():
                normalized_name = self._normalize_company_name(company_name)
                company_name_lower = company_name.strip().lower()
                
                if normalized_name or company_name_lower:
                    # Get all documents for this user and check in a single pass
                    all_docs = collection_ref.stream()
                    for doc in all_docs:
                        doc_data = doc.to_dict()
                        existing_company = doc_data.get("companyName", "")
                        if existing_company:
                            # Check normalized match first (more robust)
                            if normalized_name:
                                existing_normalized = self._normalize_company_name(existing_company)
                                if existing_normalized == normalized_name:
                                    print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] Found duplicate sponsorship by normalized company name: {doc.id}")
                                    print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] Original: '{company_name}' -> Normalized: '{normalized_name}'")
                                    print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] Existing: '{existing_company}' -> Normalized: '{existing_normalized}'")
                                    return doc.id
                            
                            # Fallback: Check case-insensitive match
                            if company_name_lower and existing_company.strip().lower() == company_name_lower:
                                print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] Found duplicate sponsorship by case-insensitive company name: {doc.id}")
                                return doc.id
            
            return None
        
        except Exception as e:
            print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] Error checking for duplicate: {e}")
            import traceback
            print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] Traceback: {traceback.format_exc()}")
            # Don't fail on duplicate check errors, just log and continue
            return None
    
    def save_sponsorship_info(self, user_id: str, request_id: str, sponsorship_data: Dict[str, Any], job_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Save sponsorship information to Firestore.
        Checks for duplicates before saving.
        
        Args:
            user_id: The user ID
            request_id: The request ID (unique per match-jobs call)
            sponsorship_data: Dictionary containing sponsorship data
            job_info: Optional job information dictionary
            
        Returns:
            The document ID of the saved sponsorship (or existing duplicate)
        """
        try:
            # Ensure Firebase is initialized (trust singleton pattern)
            if FirebaseService._app is None:
                logger.warning("Firebase app is None, re-initializing...")
                self._initialize_firebase()
            
            # Use Firestore client directly (trust initialization)
            if FirebaseService._db is None:
                FirebaseService._db = firestore.client()
            
            db = FirebaseService._db
            
            # Extract company name for duplicate check
            company_name = job_info.get("company") if job_info else sponsorship_data.get("company_name", "")
            portal = job_info.get("portal", "") if job_info else ""
            job_url = job_info.get("job_url", "") if job_info else ""
            
            # Prepare document data with FLAT structure (same as job_applications)
            # This ensures consistency and avoids any potential nested object serialization issues
            # Note: createdAt will be preserved from existing doc if duplicate found
            document_data = {
                "requestId": request_id,
                "companyName": company_name or "",
                "portal": portal if portal else "",
                "website": job_url if job_url else "",
                "sponsorsWorkers": bool(sponsorship_data.get("sponsors_workers", False)),
                "visaTypes": sponsorship_data.get("visa_types", "") or "",
                "summary": sponsorship_data.get("summary", "") or "",
                "createdAt": datetime.now(),  # Will be overwritten if duplicate found
            }
            
            # Check for duplicates before saving
            existing_doc_id = self._check_sponsorship_duplicate(user_id, company_name, request_id)
            if existing_doc_id:
                logger.debug(f"Duplicate sponsorship check found: {existing_doc_id}, replacing with latest data")
                # Replace the existing document with latest data
                collection_ref = db.collection("sponsorship_checks").document(user_id).collection("checks")
                doc_ref = collection_ref.document(existing_doc_id)
                
                # Get existing document to preserve createdAt if it exists
                existing_doc = doc_ref.get()
                if existing_doc.exists:
                    existing_data = existing_doc.to_dict()
                    if "createdAt" in existing_data:
                        document_data["createdAt"] = existing_data["createdAt"]
                    else:
                        document_data["createdAt"] = datetime.now()
                else:
                    document_data["createdAt"] = datetime.now()
                
                # Always set updatedAt to now when replacing
                document_data["updatedAt"] = datetime.now()
                
                # Replace the entire document with latest data
                doc_ref.set(document_data)
                logger.info(f"✓ Replaced duplicate sponsorship info with latest data: {company_name} (ID: {existing_doc_id})")
                return existing_doc_id
            
            # Ensure user_id document exists
            user_doc_ref = db.collection("sponsorship_checks").document(user_id)
            user_doc = user_doc_ref.get()
            if not user_doc.exists:
                user_doc_ref.set({"userId": user_id, "createdAt": datetime.now()}, merge=True)
            
            # Create subcollection reference
            collection_ref = db.collection("sponsorship_checks").document(user_id).collection("checks")
            
            # Save the document
            try:
                result = collection_ref.add(document_data)
            except Exception as save_error:
                error_str = str(save_error).lower()
                if "permission" in error_str or "denied" in error_str or "403" in error_str:
                    logger.error("Permission denied! Check Firestore security rules.")
                logger.error(f"Save operation failed: {save_error}", exc_info=True)
                raise RuntimeError(f"Failed to save sponsorship document: {save_error}")
            
            # Handle return value
            if isinstance(result, tuple):
                update_time, doc_ref = result
                doc_id = doc_ref.id
            else:
                doc_ref = result
                doc_id = doc_ref.id if hasattr(doc_ref, 'id') else str(doc_ref)
            
            if not doc_id:
                raise RuntimeError("Document ID is None or empty after save operation")
            
            logger.info(f"✓ Saved sponsorship info: {company_name} (ID: {doc_id})")
            return doc_id
                
        except Exception as e:
            error_msg = f"Failed to save sponsorship info for user {user_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)


# Singleton instance
_firebase_service: Optional[FirebaseService] = None


def get_firebase_service() -> FirebaseService:
    """Get or create the Firebase service instance."""
    global _firebase_service
    if _firebase_service is None:
        _firebase_service = FirebaseService()
    return _firebase_service