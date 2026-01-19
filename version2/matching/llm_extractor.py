"""
LLM Extraction Module

Uses LangGraph + GPT-4 to extract structured data from job descriptions and resumes.
"""

import json
import re
import logging
from typing import Dict, Any, Optional
from agents import Agent, get_model_config
from .config import LLM_CONFIG, SKILL_NORMALIZATIONS

logger = logging.getLogger(__name__)


# Use get_model_config from agents module instead


def normalize_skill(skill: str) -> str:
    """Normalize skill name using predefined mappings."""
    skill_lower = skill.lower().strip()
    return SKILL_NORMALIZATIONS.get(skill_lower, skill.strip())


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response, handling markdown and other formatting."""
    if not text:
        return None
    
    # Remove markdown code fences
    if '```json' in text:
        match = re.search(r'```json\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif '```' in text:
        match = re.search(r'```\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    
    # Try to find JSON object
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def build_extraction_agent(model_name: str = None) -> Agent:
    """Build LangGraph agent for data extraction."""
    model_name = model_name or LLM_CONFIG["model"]
    model = get_model_config(model_name, default_temperature=LLM_CONFIG["temperature"])
    
    return Agent(
        name="Data Extractor",
        role="Extract structured data from job descriptions and resumes",
        model=model,
        instructions=[
            "Extract information in JSON format with no additional text or markdown.",
            "Return ONLY valid JSON with two keys: 'job' and 'resume'.",
            "",
            "From job description, extract:",
            "- skills_required: array of required technical skills (normalized)",
            "- skills_preferred: array of preferred skills (normalized)",
            "- years_experience: number (minimum years, use 0 if not specified)",
            "- responsibilities: array of key job responsibilities",
            "- education_required: string (None/Associate/Bachelor/Master/PhD)",
            "- education_field: string (field of study, or 'Any' if not specified)",
            "- seniority_level: string (Entry/Mid/Senior/Lead/Executive)",
            "",
            "From resume, extract:",
            "- candidate_skills: array of all skills (normalized)",
            "- candidate_years: number (total relevant years)",
            "- candidate_responsibilities: array of past responsibilities",
            "- candidate_education_level: string (highest degree)",
            "- candidate_education_field: string (field of study)",
            "- candidate_seniority: string (inferred level)",
            "",
            "Normalization rules:",
            "- Convert variations to standard names (React.js → React, SQL Server → SQL)",
            "- Extract implicit skills (built REST APIs → REST APIs)",
            "- Identify database variants as SQL",
            "",
            "CRITICAL: Return ONLY the JSON object, no explanations.",
        ],
        show_tool_calls=False,
        markdown=False,
    )


def extract_data_with_llm(
    job_description: str,
    resume: str,
    model_name: str = None,
    max_retries: int = None
) -> Dict[str, Any]:
    """
    Extract structured data from job description and resume using LLM.
    
    Args:
        job_description: Full job description text
        resume: Full resume text
        model_name: Optional model name override
        max_retries: Optional retry count override
    
    Returns:
        Dict with 'job' and 'resume' keys containing extracted data
    
    Raises:
        ValueError: If extraction fails or returns invalid data
    """
    max_retries = max_retries or LLM_CONFIG["max_retries"]
    agent = build_extraction_agent(model_name)
    
    prompt = f"""Extract the following information in JSON format with no additional text or markdown.

From the job description, extract:
- skills_required: array of required technical skills (normalized names)
- skills_preferred: array of preferred skills (normalized names)
- years_experience: number (minimum years required, use 0 if not specified)
- responsibilities: array of key job responsibilities
- education_required: string (one of: "None", "Associate", "Bachelor", "Master", "PhD")
- education_field: string (field of study, or "Any" if not specified)
- seniority_level: string (one of: "Entry", "Mid", "Senior", "Lead", "Executive")

From the resume, extract:
- candidate_skills: array of all skills mentioned (normalized names)
- candidate_years: number (total relevant years of experience)
- candidate_responsibilities: array of past responsibilities from all roles
- candidate_education_level: string (highest degree: "None", "Associate", "Bachelor", "Master", "PhD")
- candidate_education_field: string (field of study)
- candidate_seniority: string (inferred level: "Entry", "Mid", "Senior", "Lead", "Executive")

Normalization rules:
- Convert skill variations to standard names (e.g., "React.js" → "React", "SQL Server" → "SQL")
- Extract implicit skills (e.g., "built REST APIs" implies "REST APIs" skill)
- Identify PostgreSQL, MySQL, etc. as "SQL" variants

Return ONLY a valid JSON object with two keys: "job" and "resume", each containing the extracted fields.

Job Description:
{job_description}

Resume:
{resume}
"""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Extraction attempt {attempt + 1}/{max_retries}")
            
            # Get response from agent
            response = agent.run(prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                response_text = str(response.content)
            elif hasattr(response, 'messages') and response.messages:
                last_msg = response.messages[-1]
                response_text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
            else:
                response_text = str(response)
            
            logger.debug(f"Raw LLM response: {response_text[:500]}...")
            
            # Parse JSON
            extracted_data = extract_json_from_response(response_text)
            
            if not extracted_data:
                raise ValueError("Could not extract valid JSON from LLM response")
            
            # Validate structure
            if "job" not in extracted_data or "resume" not in extracted_data:
                raise ValueError("Missing 'job' or 'resume' keys in extracted data")
            
            # Normalize skills
            job_data = extracted_data["job"]
            resume_data = extracted_data["resume"]
            
            if "skills_required" in job_data:
                job_data["skills_required"] = [normalize_skill(s) for s in job_data["skills_required"]]
            if "skills_preferred" in job_data:
                job_data["skills_preferred"] = [normalize_skill(s) for s in job_data["skills_preferred"]]
            if "candidate_skills" in resume_data:
                resume_data["candidate_skills"] = [normalize_skill(s) for s in resume_data["candidate_skills"]]
            
            logger.info("Successfully extracted and normalized data")
            return extracted_data
            
        except Exception as e:
            logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to extract data after {max_retries} attempts: {e}")
    
    raise ValueError("Extraction failed")


def validate_extracted_data(data: Dict[str, Any]) -> bool:
    """
    Validate that extracted data has required fields and reasonable values.
    
    Args:
        data: Extracted data dict with 'job' and 'resume' keys
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    from .config import VALIDATION
    
    # Check job data
    job = data.get("job", {})
    required_job_fields = ["skills_required", "years_experience", "responsibilities", 
                          "education_required", "seniority_level"]
    for field in required_job_fields:
        if field not in job:
            raise ValueError(f"Missing required job field: {field}")
    
    # Check resume data
    resume = data.get("resume", {})
    required_resume_fields = ["candidate_skills", "candidate_years", "candidate_responsibilities",
                             "candidate_education_level", "candidate_seniority"]
    for field in required_resume_fields:
        if field not in resume:
            raise ValueError(f"Missing required resume field: {field}")
    
    # Validate minimum data quality
    if len(job.get("skills_required", [])) < VALIDATION["min_skills"]:
        logger.warning("Job has very few required skills")
    
    if len(resume.get("candidate_skills", [])) < VALIDATION["min_skills"]:
        logger.warning("Resume has very few skills")
    
    return True

