"""
Agent-based summarization of scraped job data.
Takes scraped_data from playwright_scraper and returns structured information.
"""
from typing import Dict, Any, Optional, List
import os
import re
import json
import logging
from agents import Agent, get_model_config

# Setup logging with environment variable control
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# get_model_config is imported from agents module
# Note: We don't use JSON mode here since we parse text responses with regex


def summarize_scraped_data(
    scraped_data: Dict[str, Any],
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use an agent to summarize scraped job data into structured format.
    
    Args:
        scraped_data: Dictionary containing scraped job information from playwright_scraper
        openai_api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
    
    Returns:
        Dictionary with structured job information:
        - job_title
        - company_name
        - location
        - description
        - required_skills
        - required_experience
        - qualifications
        - responsibilities
        - salary
        - job_type
        - suggested_skills
    """
    # Set OpenAI API key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be provided or set as environment variable")
    
    # Create agent with fast model and temperature=0 to prevent hallucination
    # Using gpt-4o-mini for speed, with temperature=0 for consistency
    # Note: We don't use JSON response format here since we parse text responses with regex
    model_name = "gpt-4o-mini"  # Fast model
    from langchain_openai import ChatOpenAI
    # Create model directly without JSON response format (summarizer doesn't need JSON)
    model = ChatOpenAI(model=model_name, temperature=0)
    
    agent = Agent(
        show_tool_calls=True,
        markdown=True,
        model=model
    )
    
    # Prepare the content to analyze
    content_to_analyze = ""
    if isinstance(scraped_data, dict):
        # Reduced logging for performance - only log key info
        # Handle None values safely before slicing
        job_title = scraped_data.get('job_title') or 'N/A'
        company_name = scraped_data.get('company_name') or 'N/A'
        job_title_str = str(job_title)[:50] if job_title != 'N/A' else 'N/A'
        company_name_str = str(company_name)[:50] if company_name != 'N/A' else 'N/A'
        text_content_len = len(str(scraped_data.get('text_content', '')))
        logger.debug(f"Processing - Title: {job_title_str}, Company: {company_name_str}, Content: {text_content_len} chars")
        
        # Combine all relevant fields - prioritize raw scraped data
        content_parts = []
        
        # CRITICAL: Start with raw scraped text content (most reliable source)
        if scraped_data.get("text_content"):
            content_parts.append(f"=== RAW SCRAPED PAGE TEXT (PRIMARY SOURCE) ===\n{scraped_data['text_content']}\n")
        
        # Also include description if available (may be processed/cleaned version)
        if scraped_data.get("description"):
            content_parts.append(f"=== DESCRIPTION FIELD ===\n{scraped_data['description']}\n")
        
        # Include other structured fields
        if scraped_data.get("qualifications"):
            content_parts.append(f"=== QUALIFICATIONS ===\n{scraped_data['qualifications']}\n")
        
        if scraped_data.get("suggested_skills"):
            content_parts.append(f"=== SUGGESTED SKILLS ===\n{scraped_data['suggested_skills']}\n")
        
        # Pre-extracted fields (from Gemini extraction - these are accurate, use as-is)
        if scraped_data.get("job_title"):
            content_parts.append(f"=== JOB TITLE (ALREADY EXTRACTED BY Gemini) ===\n{scraped_data['job_title']}\n")
        
        if scraped_data.get("company_name"):
            content_parts.append(f"=== COMPANY NAME (ALREADY EXTRACTED BY Gemini) ===\n{scraped_data['company_name']}\n")
        
        if scraped_data.get("location"):
            content_parts.append(f"=== PRE-EXTRACTED LOCATION ===\n{scraped_data['location']}\n")
        
        content_to_analyze = "\n".join(content_parts) if content_parts else str(scraped_data)
    else:
        content_to_analyze = str(scraped_data)
        logger.warning(f"Received non-dict data: {type(scraped_data)}")
    
    # Check if company name and job title are already provided (from Gemini extraction)
    has_company = scraped_data.get("company_name") and scraped_data.get("company_name") not in [None, "", "Not specified", "Company name not available in posting"]
    has_title = scraped_data.get("job_title") and scraped_data.get("job_title") not in [None, "", "Not specified", "Job title not available in posting"]
    
    # Create optimized extraction prompt - skip extraction if already provided
    if has_company and has_title:
        # Both already extracted - focus only on other fields
        extraction_prompt = f"""You are extracting structured information from RAW SCRAPED JOB POSTING DATA. 
The job title and company name have already been extracted and are accurate. Use them as provided.

JOB TITLE (ALREADY EXTRACTED): {scraped_data.get("job_title")}
COMPANY NAME (ALREADY EXTRACTED): {scraped_data.get("company_name")}

RAW SCRAPED DATA PROVIDED:
{content_to_analyze}

EXTRACTION RULES:

1. **Job Title**: Use the provided title: {scraped_data.get("job_title")}
2. **Company Name**: Use the provided company: {scraped_data.get("company_name")}

3. **Complete Job Description**: Extract the full job description from the content
4. **Required Skills**: List each skill separately (e.g., Python, JavaScript, React)
5. **Required Experience**: Extract years and type (e.g., "3-5 years", "Senior level")
6. **Qualifications and Education**: Extract education requirements
7. **Responsibilities**: Extract key responsibilities and duties
8. **Salary/Compensation**: Extract if mentioned (e.g., "$100k-$150k", "£50,000-£70,000")
9. **Location**: Extract job location (city, state, country, or remote)
10. **Job Type**: Extract employment type (full-time, part-time, contract, internship, etc.)
11. **Visa Sponsorship/Scholarship**: Look for keywords like: visa sponsorship, visa support, H1B, work permit, 
    scholarship, funding, financial support, tuition assistance. Extract exact details if mentioned.

OUTPUT FORMAT:
Return structured data with all fields clearly labeled. Use clear field names like:
- Job Title: {scraped_data.get("job_title")}
- Company Name: {scraped_data.get("company_name")}
- Description: [full description]
- Required Skills: [list of skills]
- etc.

IMPORTANT: 
- Use the provided job title and company name - do NOT re-extract them
- If a field is not found, mark it as 'Not specified'
- For visa/scholarship: If mentioned, extract exact details. If not mentioned, set to 'Not specified'"""
    else:
        # Need to extract company/title - use full extraction prompt
        extraction_prompt = f"""You are extracting structured information from RAW SCRAPED JOB POSTING DATA. 
The raw data may contain extraction errors in job title and company name fields, so you must carefully analyze 
the FULL TEXT CONTENT to find the correct information.

RAW SCRAPED DATA PROVIDED:
{content_to_analyze}

CRITICAL EXTRACTION RULES:

1. **Job Title** (REQUIRED - MUST be extracted accurately):
   - Look in the FULL TEXT CONTENT for the actual job title
   - Common locations: Page title, H1/H2 headings, first few lines of content, metadata
   - Extract the COMPLETE job title (e.g., "Senior Software Engineer", not just "Engineer")
   - If truly not found after thorough analysis, return "Not specified"

2. **Company Name** (REQUIRED - MUST be extracted accurately):
   - Look in the FULL TEXT CONTENT for the actual company name
   - Common locations: "by [Company]", "Company:", "at [Company]", "from [Company]", "Employer:", "Organization:"
   - Look for company names with legal suffixes: Ltd, Limited, Inc, LLC, Corp, Corporation, Group, Holdings
   - If truly not found after thorough analysis, return "Not specified"

3. **Complete Job Description**: Extract the full job description from the content
4. **Required Skills**: List each skill separately (e.g., Python, JavaScript, React)
5. **Required Experience**: Extract years and type (e.g., "3-5 years", "Senior level")
6. **Qualifications and Education**: Extract education requirements
7. **Responsibilities**: Extract key responsibilities and duties
8. **Salary/Compensation**: Extract if mentioned (e.g., "$100k-$150k", "£50,000-£70,000")
9. **Location**: Extract job location (city, state, country, or remote)
10. **Job Type**: Extract employment type (full-time, part-time, contract, internship, etc.)
11. **Visa Sponsorship/Scholarship**: Look for keywords like: visa sponsorship, visa support, H1B, work permit, 
    scholarship, funding, financial support, tuition assistance. Extract exact details if mentioned.

OUTPUT FORMAT:
Return structured data with all fields clearly labeled. Use clear field names like:
- Job Title: [extracted title]
- Company Name: [extracted company name]
- Description: [full description]
- Required Skills: [list of skills]
- etc.

IMPORTANT: 
- If a field is not found, mark it as 'Not specified'
- For visa/scholarship: If mentioned, extract exact details. If not mentioned, set to 'Not specified'"""
    
    try:
        # Run agent
        agent_response = agent.run(extraction_prompt)
        
        # Extract response content
        response_text = ""
        if hasattr(agent_response, 'content'):
            response_text = str(agent_response.content)
        elif hasattr(agent_response, 'messages') and agent_response.messages:
            last_msg = agent_response.messages[-1]
            response_text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
        else:
            response_text = str(agent_response)
        
        # Clean markdown formatting from response text
        response_text = _clean_summary_text(response_text)
        
        # Parse the structured response
        structured_data = _parse_agent_response(response_text, scraped_data)
        
        return structured_data
        
    except Exception as e:
        # Fallback: return basic structure from scraped_data
        print(f"[RESPONSE] Error in agent summarization: {e}")
        return _create_fallback_response(scraped_data)


def _parse_agent_response(response_text: str, scraped_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse agent response to extract structured fields.
    """
    result = _empty_result()

    # Try to parse JSON if agent responded with structured JSON
    json_payload = _extract_json_payload(response_text)
    if json_payload:
        return _result_from_json(json_payload, scraped_data, result)

    # Use regex to extract fields from agent response
    # Prioritize job_title and company_name with multiple pattern variations
    patterns = {
        "job_title": [
            r'(?:Job Title|Title|Position|Role)[:\*\s]+(.+?)(?:\n|$)',
            r'1\.\s*\*\*Job title\*\*[:\s]+(.+?)(?:\n|$)',
            r'Job title[:\s]+(.+?)(?:\n|$)',
            r'Title[:\s]+(.+?)(?:\n|$)',
        ],
        "company_name": [
            r'(?:Company Name|Company|Employer|Organization|Organisation)[:\*\s]+(.+?)(?:\n|$)',
            r'2\.\s*\*\*Company name\*\*[:\s]+(.+?)(?:\n|$)',
            r'Company[:\s]+(.+?)(?:\n|$)',
            r'by\s+([A-Z][A-Za-z0-9\s&.,\-]{2,60})(?:\n|$)',
        ],
        "location": r'(?:Location)[:\s]+(.+?)(?:\n|$)',
        "required_experience": r'(?:Required Experience|Experience|Years of Experience)[:\s]+(.+?)(?:\n|$)',
        "salary": r'(?:Salary|Compensation|Pay)[:\s]+(.+?)(?:\n|$)',
        "job_type": r'(?:Job Type|Type|Employment Type)[:\s]+(.+?)(?:\n|$)',
        "visa_scholarship_info": r'(?:Visa Sponsorship|Visa Support|Scholarship|Visa/Scholarship|Visa and Scholarship)[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
    }
    
    # Extract job_title and company_name with multiple patterns (they are critical)
    for field in ["job_title", "company_name"]:
        if field in patterns:
            pattern_list = patterns[field] if isinstance(patterns[field], list) else [patterns[field]]
            for pattern in pattern_list:
                match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    extracted_value = match.group(1).strip()
                    # Clean markdown formatting from extracted values
                    cleaned_value = _clean_summary_text(extracted_value)
                    if cleaned_value and cleaned_value.lower() not in ["not specified", "unknown", "none", ""]:
                        result[field] = cleaned_value
                        print(f"[PARSER] Extracted {field}: {cleaned_value}")
                        break
    
    # Extract other fields with single patterns
    for field, pattern in patterns.items():
        if field in ["job_title", "company_name"]:
            continue  # Already handled above
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted_value = match.group(1).strip()
            # Clean markdown formatting from extracted values
            result[field] = _clean_summary_text(extracted_value)
    
    # Extract description (everything after "Description:" or "Job Description:")
    desc_match = re.search(
        r'(?:Description|Job Description|Complete Job Description)[:\s]+(.+?)(?:\n\n(?:Required|Qualifications|Responsibilities)|\Z)',
        response_text,
        re.IGNORECASE | re.DOTALL
    )
    if desc_match:
        extracted_desc = desc_match.group(1).strip()
        result["description"] = _clean_summary_text(extracted_desc)
    
    # Extract qualifications
    qual_match = re.search(
        r'(?:Qualifications|Education Requirements|Qualifications and Education)[:\s]+(.+?)(?:\n\n(?:Required|Responsibilities|Salary)|\Z)',
        response_text,
        re.IGNORECASE | re.DOTALL
    )
    if qual_match:
        extracted_qual = qual_match.group(1).strip()
        result["qualifications"] = _clean_summary_text(extracted_qual)
    
    # Extract responsibilities
    resp_match = re.search(
        r'(?:Responsibilities|Core Responsibilities|Duties)[:\s]+(.+?)(?:\n\n(?:Required|Qualifications|Salary)|\Z)',
        response_text,
        re.IGNORECASE | re.DOTALL
    )
    if resp_match:
        extracted_resp = resp_match.group(1).strip()
        result["responsibilities"] = _clean_summary_text(extracted_resp)
    
    # Extract required skills (list)
    skills_section = re.search(
        r'(?:Required Skills|Skills Needed|Skills Required)[:\s]+(.+?)(?:\n\n(?:Required|Qualifications|Responsibilities|Experience)|\Z)',
        response_text,
        re.IGNORECASE | re.DOTALL
    )
    if skills_section:
        skills_text = skills_section.group(1).strip()
        # Split by newlines, bullets, or commas
        result["required_skills"] = [
            skill.strip() for skill in re.split(r'[\n•,\-]', skills_text)
            if skill.strip() and skill.strip() != "Not specified"
        ]
    
    # Prioritize pre-extracted values from Gemini (they are more accurate)
    # Only use parsed values if pre-extracted are missing
    if scraped_data.get("job_title") and scraped_data.get("job_title") not in [None, "", "Not specified", "Job title not available in posting"]:
        result["job_title"] = scraped_data["job_title"]
    elif not result["job_title"]:
        result["job_title"] = scraped_data.get("job_title") or None
    
    if scraped_data.get("company_name") and scraped_data.get("company_name") not in [None, "", "Not specified", "Company name not available in posting"]:
        result["company_name"] = scraped_data["company_name"]
    elif not result["company_name"]:
        result["company_name"] = scraped_data.get("company_name") or None
    
    if not result["location"] and scraped_data.get("location"):
        result["location"] = scraped_data["location"]
    
    if not result["description"] and scraped_data.get("description"):
        result["description"] = scraped_data["description"]
    
    if not result["qualifications"] and scraped_data.get("qualifications"):
        result["qualifications"] = scraped_data["qualifications"]
    
    if not result["suggested_skills"] and scraped_data.get("suggested_skills"):
        # Parse suggested skills from scraped data
        skills_text = scraped_data["suggested_skills"]
        result["suggested_skills"] = [
            skill.strip() for skill in re.split(r'[\n•,\-]', skills_text)
            if skill.strip()
        ]
    
    # If description is still None, use full response as description
    if not result["description"]:
        result["description"] = response_text.strip()
    
    # Extract visa/scholarship info with broader search if not found
    if not result["visa_scholarship_info"] or result["visa_scholarship_info"] == "Not specified":
        # Look for visa/scholarship keywords in the full response
        visa_keywords = [
            r'visa sponsorship[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'visa support[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'scholarship[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'H1B[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'work permit[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'financial support[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
            r'tuition assistance[:\s]+(.+?)(?:\n\n|\n(?=[A-Z])|\Z)',
        ]
        for pattern in visa_keywords:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                result["visa_scholarship_info"] = match.group(1).strip()
                break
        
        # Also check in scraped_data text_content
        if (not result["visa_scholarship_info"] or result["visa_scholarship_info"] == "Not specified") and scraped_data.get("text_content"):
            text_content = scraped_data["text_content"].lower()
            if any(keyword in text_content for keyword in ["visa sponsorship", "visa support", "scholarship", "h1b", "work permit", "financial support", "tuition"]):
                # Extract surrounding context
                for keyword in ["visa sponsorship", "visa support", "scholarship", "h1b", "work permit"]:
                    if keyword in text_content:
                        # Find the sentence or paragraph containing the keyword
                        idx = text_content.find(keyword)
                        start = max(0, idx - 100)
                        end = min(len(text_content), idx + 200)
                        context = scraped_data["text_content"][start:end].strip()
                        result["visa_scholarship_info"] = context
                        break
            else:
                result["visa_scholarship_info"] = "Not specified"
        elif not result["visa_scholarship_info"]:
            result["visa_scholarship_info"] = "Not specified"
    
    return _finalize_result(result, scraped_data)


def _create_fallback_response(scraped_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a fallback response structure from scraped_data if agent fails.
    """
    # Check for visa/scholarship info in scraped data
    visa_scholarship_info = "Not specified"
    text_content = scraped_data.get("text_content", "").lower() if scraped_data.get("text_content") else ""
    if any(keyword in text_content for keyword in ["visa sponsorship", "visa support", "scholarship", "h1b", "work permit", "financial support", "tuition"]):
        # Extract context around visa/scholarship keywords
        for keyword in ["visa sponsorship", "visa support", "scholarship", "h1b", "work permit"]:
            if keyword in text_content:
                idx = text_content.find(keyword)
                start = max(0, idx - 100)
                end = min(len(text_content), idx + 200)
                context = scraped_data.get("text_content", "")[start:end].strip()
                visa_scholarship_info = context
                break
    
    return {
        "job_title": scraped_data.get("job_title"),
        "company_name": scraped_data.get("company_name"),
        "location": scraped_data.get("location"),
        "description": scraped_data.get("description") or scraped_data.get("text_content", "")[:2000],
        "required_skills": [],
        "required_experience": None,
        "qualifications": scraped_data.get("qualifications"),
        "responsibilities": None,
        "salary": None,
        "job_type": None,
        "suggested_skills": scraped_data.get("suggested_skills", "").split("\n") if scraped_data.get("suggested_skills") else [],
        "visa_scholarship_info": visa_scholarship_info
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "job_title": None,
        "company_name": None,
        "location": None,
        "description": None,
        "required_skills": [],
        "required_experience": None,
        "qualifications": None,
        "responsibilities": None,
        "salary": None,
        "job_type": None,
        "suggested_skills": [],
        "visa_scholarship_info": None
    }


def _extract_json_payload(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON payload from response text if present."""
    text = response_text.strip()

    # Remove leading/trailing backticks or code fences
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Look for first JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        return None

    json_candidate = json_match.group(0)

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        # Try to clean up trailing commas or quotes
        cleaned = re.sub(r",\s*([}\]])", r"\1", json_candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _result_from_json(
    payload: Dict[str, Any],
    scraped_data: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """Populate result from JSON payload. Prioritizes job_title and company_name."""
    field_mapping = {
        # CRITICAL FIELDS - multiple variations to ensure extraction
        "job_title": ["job_title", "Job title", "Title", "jobTitle", "JobTitle", "position", "Position", "role", "Role"],
        "company_name": ["company_name", "Company name", "Company", "companyName", "CompanyName", "employer", "Employer", "organization", "Organization"],
        # Other fields
        "location": ["location", "Location"],
        "description": ["description", "Complete job description", "Job description"],
        "required_skills": ["required_skills", "Required skills"],
        "required_experience": ["required_experience", "Required experience", "Experience"],
        "qualifications": ["qualifications", "Qualifications and education requirements", "Qualifications"],
        "responsibilities": ["responsibilities", "Responsibilities"],
        "salary": ["salary", "Salary/compensation", "Compensation"],
        "job_type": ["job_type", "Job type", "Type"],
        "suggested_skills": ["suggested_skills", "Suggested skills"],
        "visa_scholarship_info": ["visa_scholarship_info", "Visa sponsorship or scholarship information"],
    }

    normalized_payload = {str(k).strip(): v for k, v in payload.items()}

    # Prioritize job_title and company_name extraction
    for field in ["job_title", "company_name"]:
        if field in field_mapping:
            keys = field_mapping[field]
            for key in keys:
                if key in normalized_payload and normalized_payload[key] not in (None, ""):
                    value = normalized_payload[key]
                    if isinstance(value, str):
                        value = value.strip().strip('"')
                        if value.lower() not in ["not specified", "unknown", "none", ""]:
                            # Clean markdown formatting from string values
                            value = _clean_summary_text(value)
                            if value:  # Only set if we got a valid value after cleaning
                                result[field] = value
                                print(f"[PARSER] Extracted {field} from JSON: {value}")
                                break
    
    # Extract other fields
    for field, keys in field_mapping.items():
        if field in ["job_title", "company_name"]:
            continue  # Already handled above
        for key in keys:
            if key in normalized_payload and normalized_payload[key] not in (None, ""):
                value = normalized_payload[key]
                if isinstance(value, str):
                    value = value.strip().strip('"')
                    if value.lower() == "not specified":
                        value = "Not specified"
                    else:
                        # Clean markdown formatting from string values
                        value = _clean_summary_text(value)
                if field in {"required_skills", "suggested_skills"} and isinstance(value, str):
                    value = _split_to_list(value)
                result[field] = value
                break

    return _finalize_result(result, scraped_data)


def _split_to_list(value: str) -> List[str]:
    return [
        item.strip()
        for item in re.split(r"[\n•,\-]", value)
        if item.strip() and item.strip().lower() != "not specified"
    ]


def _clean_summary_text(text: str) -> str:
    """
    Clean summary text to remove markdown formatting inconsistencies like "Name**: Value".
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove markdown code fences
    text = re.sub(r'^```[\w]*\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    
    # Remove patterns like "Name**:", "**Name**:", "Name:", etc. at the start of lines
    text = re.sub(r'^(\*{0,2}(?:Name|Company|Title|Job Title|Position|Role|Location|Salary|Description|Summary|Employer|Organization)\*{0,2}:?\s*)', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove bold markdown (**text** or __text__)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    # Remove italic markdown (*text* or _text_)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)_([^_]+)_(?!_)', r'\1', text)
    
    # Remove standalone asterisks at line starts/ends
    text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\*+$', '', text, flags=re.MULTILINE)
    
    # Remove patterns like "**:**" or "**: " at line starts
    text = re.sub(r'^\*{1,2}:?\s*', '', text, flags=re.MULTILINE)
    
    # Clean up multiple consecutive asterisks
    text = re.sub(r'\*{3,}', '', text)
    
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def _finalize_result(result: Dict[str, Any], scraped_data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply fallbacks and normalization before returning result."""
    # Clean all string fields to remove markdown formatting
    for field in ["job_title", "company_name", "location", "description", "qualifications", 
                  "responsibilities", "salary", "job_type", "visa_scholarship_info"]:
        if isinstance(result.get(field), str):
            result[field] = _clean_summary_text(result[field])
    
    # Fallback to scraped_data if fields are missing
    if not result["job_title"] and scraped_data.get("job_title"):
        result["job_title"] = scraped_data["job_title"]

    if not result["company_name"] and scraped_data.get("company_name"):
        result["company_name"] = scraped_data["company_name"]

    if not result["location"] and scraped_data.get("location"):
        result["location"] = scraped_data["location"]

    if not result["description"] and scraped_data.get("description"):
        result["description"] = scraped_data["description"]

    if not result["qualifications"] and scraped_data.get("qualifications"):
        result["qualifications"] = scraped_data["qualifications"]

    if not result["suggested_skills"] and scraped_data.get("suggested_skills"):
        result["suggested_skills"] = _split_to_list(scraped_data["suggested_skills"])

    if isinstance(result["required_skills"], str):
        result["required_skills"] = _split_to_list(result["required_skills"])

    if isinstance(result["suggested_skills"], str):
        result["suggested_skills"] = _split_to_list(result["suggested_skills"])

    if not result["description"]:
        result["description"] = "Not specified"

    if not result["visa_scholarship_info"]:
        result["visa_scholarship_info"] = "Not specified"
    else:
        result["visa_scholarship_info"] = result["visa_scholarship_info"].strip()

    if isinstance(result["company_name"], str):
        result["company_name"] = re.sub(r'^\*+\s*|\s*\*+$', '', result["company_name"]).strip()
        result["company_name"] = result["company_name"].replace('",', '').strip()
        if result["company_name"].lower() == "not specified":
            result["company_name"] = "Not specified"

    if "is_authorized_sponsor" not in result or result["is_authorized_sponsor"] is None:
        result["is_authorized_sponsor"] = scraped_data.get("is_authorized_sponsor")

    return result

