"""
Extract and format job data from API response for Firebase storage.
Uses the exact same structure as test_firebase_simple.py
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import json
import logging

# Setup logging with environment variable control
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def detect_portal(url: str) -> str:
    """
    Detect the job portal from URL domain.
    Same logic as in app.py
    """
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


def extract_jobs_from_response(api_response: dict) -> List[dict]:
    """
    Extract and format jobs from API response for Firebase.
    
    Maps API response fields to Firebase format exactly matching test_firebase_simple.py structure.
    
    Args:
        api_response: The JSON response from POST /api/match-jobs endpoint
            Expected structure:
            {
                "matched_jobs": [
                    {
                        "rank": 1,
                        "job_url": "...",
                        "job_title": "...",
                        "company": "...",
                        "match_score": 0.85,
                        "summary": "...",
                        "key_matches": [...],
                        "requirements_met": 7,
                        "total_requirements": 8,
                        "scraped_summary": "..."
                    }
                ]
            }
    
    Returns:
        List of job_data dictionaries ready for Firebase save_job_applications_batch()
        Format matches test_firebase_simple.py exactly:
        {
            "appliedDate": datetime.now(),
            "company": "...",
            "createdAt": datetime.now(),
            "interviewDate": "",
            "summary": "...",
            "link": "...",
            "notes": "...",
            "portal": "...",
            "role": "...",
            "status": "Matched",
            "visaRequired": "No"
        }
    """
    jobs_data = []
    matched_jobs = api_response.get("matched_jobs", [])
    
    logger.info(f"Extracting {len(matched_jobs)} jobs from API response")
    
    for idx, job in enumerate(matched_jobs, 1):
        try:
            # Extract basic fields
            company = job.get("company", "").strip() or ""
            job_title = job.get("job_title", "").strip() or ""
            job_url = job.get("job_url", "").strip() or ""
            
            # Detect portal from URL
            portal = detect_portal(job_url) if job_url else "Unknown"
            
            # Extract match information
            match_score = job.get("match_score", 0.0)
            requirements_met = job.get("requirements_met", 0)
            total_requirements = job.get("total_requirements", 0)
            requirements_satisfied = job.get("requirements_satisfied", []) or []
            requirements_missing = job.get("requirements_missing", []) or []
            improvements_needed = job.get("improvements_needed", []) or []
            key_matches = job.get("key_matches", [])
            summary = (job.get("summary") or "").strip() or ""
            scraped_summary = (job.get("scraped_summary") or "").strip() or ""
            
            # Create summary combining match info, summary, and key matches
            summary_parts = []
            
            # Add match score info
            if match_score > 0:
                summary_parts.append(f"Match Score: {match_score:.1%}")
            
            # Add requirements info
            if total_requirements > 0:
                req_percentage = (requirements_met / total_requirements) * 100
                summary_parts.append(
                    f"Requirements Met: {requirements_met}/{total_requirements} ({req_percentage:.0f}%)"
                )
            
            # Add summary (prefer scraped_summary if available, otherwise summary)
            description_text = scraped_summary if scraped_summary else summary
            if description_text:
                # Truncate description if too long (max 5000 chars, but prefer complete sentences)
                max_length = 5000
                if len(description_text) > max_length:
                    # Find the last complete sentence before max_length
                    truncated = description_text[:max_length]
                    # Try to find the last sentence ending
                    last_period = truncated.rfind('.')
                    last_exclamation = truncated.rfind('!')
                    last_question = truncated.rfind('?')
                    last_sentence_end = max(last_period, last_exclamation, last_question)
                    
                    if last_sentence_end > max_length * 0.7:  # Only use if we're keeping at least 70% of max
                        description_text = description_text[:last_sentence_end + 1]
                    else:
                        # If no good sentence boundary, try word boundary
                        last_space = truncated.rfind(' ')
                        if last_space > max_length * 0.7:
                            description_text = description_text[:last_space] + "..."
                        else:
                            description_text = truncated + "..."
                
                # Format summary with label
                if description_text.strip():
                    summary_parts.append(f"**Summary:** {description_text}")
            
            # Add key matches
            if key_matches:
                matches_str = ", ".join(key_matches[:10])  # Limit to first 10 matches
                summary_parts.append(f"Key Matches: {matches_str}")
            
            match_summary = "\n\n".join(summary_parts) or ""
            
            # Create notes from summary (truncate to 500 chars)
            notes = summary[:500] if summary else ""
            
            # Extract cleaned job description (if available, for Firestore storage)
            cleaned_job_description = job.get("job_description")  # Internal use only, NOT in API response
            
            # Prepare job_data in EXACT format from test_firebase_simple.py
            # Using datetime.now() objects, not formatted strings
            job_data = {
                "appliedDate": datetime.now(),  # EXACT same as test_firebase_simple.py line 88
                "company": company,
                "createdAt": datetime.now(),  # EXACT same as test_firebase_simple.py line 90
                "interviewDate": "",  # EXACT same as test_firebase_simple.py line 91
                "summary": match_summary,  # Combined match info + summary
                "link": job_url,  # EXACT same as test_firebase_simple.py line 93
                "notes": notes,  # Summary truncated to 500 chars
                "portal": portal,  # EXACT same as test_firebase_simple.py line 95
                "role": job_title,  # EXACT same as test_firebase_simple.py line 96
                "status": "Matched",  # Use "Matched" for auto-matched jobs (not "Applied")
                "visaRequired": "No",  # EXACT same as test_firebase_simple.py line 98
                "requirements_met": requirements_satisfied,  # List of requirements that are met
                "requirements_not_met": requirements_missing,  # List of requirements that are not met
                "improvements_needed": improvements_needed,  # List of suggested improvements
                "job_description": cleaned_job_description  # Cleaned description (internal use only, NOT in API response)
            }
            
            jobs_data.append(job_data)
            
            logger.debug(f"Job {idx}: {job_title} at {company} ({portal}) - Match: {match_score:.1%}")
            
        except Exception as e:
            logger.error(f"Failed to extract job {idx}: {str(e)}", exc_info=True)
            continue
    
    if len(jobs_data) != len(matched_jobs):
        logger.warning(f"Only extracted {len(jobs_data)}/{len(matched_jobs)} jobs")
    else:
        logger.info(f"✓ Extracted {len(jobs_data)} jobs successfully")
    return jobs_data


def clean_job_description_with_llm(
    raw_scraped_content: str,
    openai_api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini"
) -> str:
    """
    Clean and format job description from raw scraped content using LLM.
    
    INPUT:
    - Raw scraped job content (plain text)
    
    LLM TASK:
    - Read the full scraped content
    - Produce a clean, coherent job description using ONLY the given text
    - Remove UI noise, repetition, and irrelevant sections
    - Preserve responsibilities, qualifications, role context, and domain language
    - Do NOT infer or add missing requirements
    - Do NOT summarize aggressively
    - Return a single plain-text string
    
    LLM SETTINGS:
    - temperature = 0
    - top_p = 0.1
    - seed = 42
    
    Args:
        raw_scraped_content: Raw scraped job content (plain text)
        openai_api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
        model_name: OpenAI model to use (default: "gpt-4o-mini")
    
    Returns:
        Cleaned job description string (plain text, no markdown)
    """
    if not raw_scraped_content or len(raw_scraped_content.strip()) < 50:
        logger.warning("Raw scraped content too short, returning as-is")
        return raw_scraped_content.strip() if raw_scraped_content else ""
    
    # Set OpenAI API key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif not os.getenv("OPENAI_API_KEY"):
        logger.warning("OpenAI API key not available, returning raw content")
        return raw_scraped_content.strip()
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""You are cleaning raw scraped job posting content. Your task is to produce a clean, coherent job description.

RAW SCRAPED CONTENT:
{raw_scraped_content}

TASK:
1. Read the ENTIRE scraped content above
2. Produce a clean, coherent job description using ONLY the text provided
3. Remove UI noise (navigation links, "Skip to content", cookie notices, etc.)
4. Remove repetition and redundant sections
5. Remove irrelevant sections (ads, related jobs, footer content, etc.)
6. Preserve ALL important information:
   - Job responsibilities and duties
   - Required qualifications and skills
   - Experience requirements
   - Education requirements
   - Role context and domain language
   - Company information (if relevant to the role)
   - Location, salary, benefits (if mentioned)
7. Maintain the original structure and flow where possible
8. Keep domain-specific terminology and technical language

CRITICAL RULES:
- Use ONLY the text provided - do NOT infer or add missing information
- Do NOT summarize aggressively - preserve details
- Do NOT add requirements that aren't in the original text
- Do NOT remove important technical details or qualifications
- Keep the description comprehensive and informative
- Return plain text only (no markdown formatting, no code blocks)

OUTPUT:
Return a single clean, coherent job description as plain text. Do not include any explanations, notes, or meta-commentary."""
        
        logger.info(f"Cleaning job description with LLM (content length: {len(raw_scraped_content)} chars)")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.1,
            seed=42
        )
        
        cleaned_description = response.choices[0].message.content.strip()
        
        # Remove any markdown code blocks if present
        if cleaned_description.startswith("```"):
            # Extract content from code block
            lines = cleaned_description.split("\n")
            if len(lines) > 1:
                # Remove first line (```) and last line (```)
                cleaned_description = "\n".join(lines[1:-1]).strip()
        
        logger.info(f"✓ Job description cleaned (original: {len(raw_scraped_content)} chars, cleaned: {len(cleaned_description)} chars)")
        return cleaned_description
        
    except Exception as e:
        logger.error(f"Failed to clean job description with LLM: {e}", exc_info=True)
        # Return original content on error
        return raw_scraped_content.strip()


def extract_company_and_title_from_raw_data(
    raw_scraped_data: str,
    openai_api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini"  # Fast and intelligent OpenAI model
) -> Dict[str, Optional[str]]:
    """
    Extract company name and job title from raw unstructured scraped data using OpenAI.
    
    This function sends the raw scraped data directly to OpenAI API without any
    preprocessing, parsing, or extraction. OpenAI is asked to read the text
    naturally and identify the company name and job title.
    
    Args:
        raw_scraped_data: Raw unstructured scraped text from job posting
        openai_api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
        model_name: OpenAI model to use (default: "gpt-4o-mini" - fast and intelligent)
    
    Returns:
        Dictionary with structure:
        {
            "company_name": str or None,
            "job_title": str or None
        }
    
    Example:
        >>> raw_data = "Skip to main content... Junior Data Analyst... MillerKnoll..."
        >>> result = extract_company_and_title_from_raw_data(raw_data)
        >>> print(result)
        {'company_name': 'MillerKnoll', 'job_title': 'Junior Data Analyst'}
    """
    # Set OpenAI API key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be provided or set as environment variable")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        import re
        
        # Create OpenAI client with fast model and temperature=0 to prevent hallucination
        # Note: No max_tokens limit to ensure complete extraction (company/title extraction is usually short anyway)
        model = ChatOpenAI(model=model_name, temperature=0)
        
        # Create prompt that asks OpenAI to read naturally and extract only company and title
        prompt = f"""I'm going to give you raw scraped text from a job posting. This text is unstructured and may contain job titles, company names, locations, and other information all mixed together.

Your task: Read through the entire text using your language understanding capabilities and tell me what the company name is and what the job title is.

CRITICAL RULES FOR COMPANY NAME EXTRACTION:
- The company name must be a PROPER NOUN (a specific organization/company name)
- Company names are typically short (2-5 words), capitalized, and may end with suffixes like Ltd, Limited, Inc, LLC, Corp, Corporation, Group, Holdings, Technology, Solutions, Services
- PRIORITY: Look for company names in these common patterns:
  * "X days ago by [Company Name]" or "by [Company Name]" (e.g., "3 days ago by Career poster" → "Career poster")
  * "posted by [Company Name]" or "Posted by [Company Name]"
  * "Posted by [Company Name]" or "Posted By [Company Name]"
  * "Career poster" or "[Company Name] poster" (recruiter/company name)
  * "[Company Name]" followed by job details
- DO NOT extract sentence fragments, phrases, or descriptions that describe what a company does
- DO NOT extract text that starts with verbs like "works", "provides", "develops", "creates", "helps", "gathers", "produces", "supports", "builds"
- DO NOT extract text that is part of a sentence describing company activities
- Examples of INVALID company names (reject these):
  * "works with food retailers, processors and banks to" (sentence fragment)
  * "developing new features across" (sentence fragment)
  * "building software that helps" (sentence fragment)
  * "gather farm data" (verb phrase)
  * "produce advanced insights" (verb phrase)
- Examples of VALID company names (extract these):
  * "Career poster" or "Career Poster" (found in "3 days ago by Career poster")
  * "FarmMetrics" or "FarmMetrics Technology"
  * "Reed.co.uk" (company name)
  * "Microsoft Corporation" (company name with suffix)

CRITICAL RULES FOR JOB TITLE EXTRACTION:
- The job title must be a specific role name, typically capitalized
- Examples: "Full Stack PHP/Angular Developer", "Software Engineer", "Data Analyst"

If you cannot find a clear, valid company name (proper noun, not a sentence fragment), return null for company_name.

Here's the raw scraped data:
{raw_scraped_data}

Please respond with ONLY a JSON object in this exact format:
{{
    "company_name": "the actual company/organization name (proper noun only), or null if not found or if you only find sentence fragments",
    "job_title": "the job title you identified, or null if not found"
}}

Return ONLY the JSON object, nothing else. No explanations, no markdown, just the JSON."""
        
        # Send to OpenAI
        response = model.invoke([HumanMessage(content=prompt)])
        
        # Extract response text
        if hasattr(response, 'content'):
            response_text = str(response.content)
        elif hasattr(response, 'messages') and response.messages:
            last_msg = response.messages[-1]
            response_text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
        else:
            response_text = str(response)
        
        response_text = response_text.strip()
        
        # Clean response text (remove markdown code blocks if present)
        if '```json' in response_text:
            # Extract JSON from markdown code block
            match = re.search(r'```json\s*\n?(.*?)\n?```', response_text, re.DOTALL)
            if match:
                response_text = match.group(1).strip()
        elif '```' in response_text:
            # Remove any code fence markers
            response_text = re.sub(r'^```[a-zA-Z]*\s*', '', response_text, flags=re.MULTILINE)
            response_text = re.sub(r'\s*```$', '', response_text, flags=re.MULTILINE)
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            
            # Validate and clean the result
            company_name = result.get("company_name")
            job_title = result.get("job_title")
            
            # Convert empty strings to None
            if company_name == "" or company_name is None:
                company_name = None
            else:
                company_name = str(company_name).strip()
                # Validate company name - reject sentence fragments
                if company_name:
                    # Check if it starts with a verb (common sentence fragment pattern)
                    verb_starters = ["works", "work", "provides", "provide", "develops", "develop", "creates", "create",
                                    "helps", "help", "gathers", "gather", "produces", "produce", "supports", "support",
                                    "builds", "build", "connecting", "connect", "delivering", "deliver", "offering",
                                    "offer", "serving", "serve", "enabling", "enable", "making", "make", "creating",
                                    "enabling", "transforming", "transforms", "leveraging", "leverages", "facilitating",
                                    "facilitates", "integrating", "integrates", "specializing", "specializes"]
                    company_lower = company_name.lower().strip()
                    # Handle "by [Company Name]" pattern - extract the company name part
                    if company_lower.startswith("by "):
                        # Extract the part after "by "
                        extracted = company_name[3:].strip()  # Remove "by " prefix
                        if extracted and len(extracted) >= 2:
                            company_name = extracted
                            company_lower = company_name.lower().strip()  # Update lowercase after extraction
                            logger.debug(f"Extracted company name from 'by' pattern: {company_name}")
                    # Check if it starts with a verb followed by common prepositions/articles
                    if any(company_lower.startswith(verb + " ") for verb in verb_starters):
                        logger.warning(f"Rejected company name that starts with verb (sentence fragment): {company_name[:50]}")
                        company_name = None
                    # Check if it looks like a sentence fragment (contains "to " at the end, starts with lowercase verb)
                    elif company_lower.startswith(("to ", "the ", "a ", "an ")) or company_name.endswith(" to"):
                        logger.warning(f"Rejected company name that looks like sentence fragment: {company_name[:50]}")
                        company_name = None
                    # Check if it's too long (likely description)
                    elif len(company_name) > 80:
                        logger.warning(f"Rejected company name that's too long (likely description): {company_name[:50]}")
                        company_name = None
            
            if job_title == "" or job_title is None:
                job_title = None
            else:
                job_title = str(job_title).strip()
            
            return {
                "company_name": company_name,
                "job_title": job_title
            }
            
        except json.JSONDecodeError as e:
            # Try to extract JSON from text if direct parse fails
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    return {
                        "company_name": result.get("company_name") or None,
                        "job_title": result.get("job_title") or None
                    }
                except json.JSONDecodeError:
                    pass
            
            # If all parsing fails, return None values
            logger.warning(f"Failed to parse JSON from Gemini response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return {
                "company_name": None,
                "job_title": None
            }
            
    except Exception as e:
        logger.error(f"Failed to extract company and title: {str(e)}", exc_info=True)
        return {
            "company_name": None,
            "job_title": None
        }


# Example usage
if __name__ == "__main__":
    # Example API response structure
    sample_api_response = {
        "candidate_profile": {
            "name": "John Doe",
            "skills": ["Python", "FastAPI"]
        },
        "matched_jobs": [
            {
                "rank": 1,
                "job_url": "https://internshala.com/job/detail/123456",
                "job_title": "Data Science AI & ML Research Associate Fresher Job",
                "company": "Megaminds IT Services",
                "match_score": 0.85,
                "summary": "John Doe is an excellent fit for this position. His experience with Python and machine learning frameworks aligns perfectly with the job requirements.",
                "key_matches": ["Python", "TensorFlow", "Keras", "Machine Learning"],
                "requirements_met": 7,
                "total_requirements": 8,
                "scraped_summary": "We are looking for a Data Science Research Associate with experience in AI/ML..."
            },
            {
                "rank": 2,
                "job_url": "https://linkedin.com/jobs/view/789012",
                "job_title": "Software Engineer - Python",
                "company": "Tech Corp",
                "match_score": 0.75,
                "summary": "Good match for Python development role.",
                "key_matches": ["Python", "FastAPI"],
                "requirements_met": 6,
                "total_requirements": 10,
                "scraped_summary": None
            }
        ],
        "jobs_analyzed": 2,
        "request_id": "abc123"
    }
    
    # Extract jobs
    print("="*70)
    print("EXAMPLE: Extracting jobs from API response")
    print("="*70)
    jobs = extract_jobs_from_response(sample_api_response)
    
    # Display extracted jobs
    print(f"\n[RESULT] Extracted {len(jobs)} jobs:")
    for i, job in enumerate(jobs, 1):
        print(f"\nJob {i}:")
        print(f"  Company: {job['company']}")
        print(f"  Role: {job['role']}")
        print(f"  Portal: {job['portal']}")
        print(f"  Link: {job['link']}")
        print(f"  Status: {job['status']}")
        print(f"  Notes (length): {len(job['notes'])} chars")
        print(f"  Summary (length): {len(job['summary'])} chars")
        print(f"  appliedDate: {job['appliedDate']} (type: {type(job['appliedDate'])})")
        print(f"  createdAt: {job['createdAt']} (type: {type(job['createdAt'])})")
    
    # Example: Save to Firebase
    print(f"\n[INFO] To save to Firebase, use:")
    print(f"  from firebase_service import get_firebase_service")
    print(f"  firebase_service = get_firebase_service()")
    print(f"  doc_ids = firebase_service.save_job_applications_batch(user_id, jobs)")
    print("="*70)
    
    # Example: Extract company and title from raw scraped data
    print("\n" + "="*70)
    print("EXAMPLE: Extracting company and title from raw scraped data")
    print("="*70)
    
    sample_raw_data = """Skip to main content
Junior Data Analyst in Chennai
MillerKnoll
London, England, United Kingdom
Junior Data Analyst
Apply
Join to apply for the Junior Data Analyst role at MillerKnoll"""
    
    try:
        result = extract_company_and_title_from_raw_data(sample_raw_data)
        print(f"\n[RESULT] Extracted:")
        print(f"  Company Name: {result['company_name']}")
        print(f"  Job Title: {result['job_title']}")
    except Exception as e:
        print(f"[ERROR] {str(e)}")
    
    print("="*70)

