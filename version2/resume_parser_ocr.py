"""
Fast OCR-based resume parser using pdfplumber and regex.
Falls back to LLM parsing only if OCR extraction fails or gets insufficient data.
"""
import re
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_resume_with_ocr(resume_text: str) -> Dict[str, Any]:
    """
    Parse resume using OCR text with regex patterns.
    Fast alternative to LLM parsing (0.5-2s vs 5-10s).
    
    Args:
        resume_text: Raw text extracted from PDF using pdfplumber
        
    Returns:
        Dictionary with parsed resume fields:
        {
            "name": str,
            "email": str or None,
            "phone": str or None,
            "skills": List[str],
            "experience_summary": str,
            "total_years_experience": float,
            "education": List[Dict],
            "certifications": List[str],
            "interests": List[str]
        }
    """
    if not resume_text or len(resume_text.strip()) < 50:
        logger.warning("Resume text too short for OCR parsing, will fallback to LLM")
        return None
    
    result = {
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
        "experience_summary": "",
        "total_years_experience": 0.0,
        "education": [],
        "certifications": [],
        "interests": []
    }
    
    # Extract name (usually first line or first 2-3 capitalized words)
    name_patterns = [
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',  # First line with 2-4 capitalized words
        r'^([A-Z][A-Z\s]+)',  # All caps name
        r'Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # "Name: John Doe"
    ]
    for pattern in name_patterns:
        match = re.search(pattern, resume_text, re.MULTILINE)
        if match:
            name = match.group(1).strip()
            if len(name.split()) >= 2 and len(name) < 50:  # Reasonable name length
                result["name"] = name
                break
    
    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text)
    if email_match:
        result["email"] = email_match.group()
    
    # Extract phone (various formats)
    phone_patterns = [
        r'\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}',  # Standard US/UK format
        r'\+?\d[\d\s-]{8,}\d',  # Generic international
        r'Phone[:\s]+([+\d\s\-()]+)',  # "Phone: +1 234 567 8900"
    ]
    for pattern in phone_patterns:
        match = re.search(pattern, resume_text)
        if match:
            phone = re.sub(r'[^\d+]', '', match.group(0) if match.groups() else match.group())
            if len(phone) >= 10:
                result["phone"] = phone
                break
    
    # Extract skills (common technical skills)
    skill_keywords = [
        # Programming Languages
        'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust', 'Swift', 'Kotlin',
        'PHP', 'Ruby', 'Perl', 'Scala', 'R', 'MATLAB', 'SQL', 'HTML', 'CSS',
        # Frameworks & Libraries
        'React', 'Angular', 'Vue', 'Node.js', 'Express', 'Django', 'Flask', 'FastAPI', 'Spring',
        'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'Pandas', 'NumPy', 'OpenCV',
        # Databases
        'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Cassandra', 'DynamoDB',
        # Cloud & DevOps
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'CI/CD', 'Git', 'GitHub', 'GitLab',
        # Tools & Others
        'Linux', 'Unix', 'Windows', 'MacOS', 'JIRA', 'Confluence', 'Agile', 'Scrum',
        # AI/ML
        'Machine Learning', 'Deep Learning', 'AI', 'Data Science', 'NLP', 'Computer Vision',
        'Natural Language Processing', 'Neural Networks', 'Data Analysis'
    ]
    
    found_skills = []
    resume_lower = resume_text.lower()
    for skill in skill_keywords:
        if skill.lower() in resume_lower:
            found_skills.append(skill)
    
    result["skills"] = found_skills
    
    # Extract experience summary - look for ALL types: work, internships, projects, hackathons
    experience_patterns = [
        r'Experience[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
        r'Work\s+Experience[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
        r'Professional\s+Experience[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
        r'Internship[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
        r'Projects?[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
        r'Hackathon[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
        r'Achievements?[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
    ]
    
    experience_sections = []
    for pattern in experience_patterns:
        matches = re.finditer(pattern, resume_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            section_text = match.group(1).strip()
            if len(section_text) > 50:  # Only include substantial sections
                experience_sections.append(section_text[:500])  # Limit each section
    
    # Combine all experience sections
    if experience_sections:
        experience_text = " ".join(experience_sections[:3])  # Take first 3 sections, max 1500 chars
        experience_text = experience_text[:1500]  # Final limit
    else:
        # If no explicit experience section, look for project/internship mentions in the text
        # Try to extract relevant experience-related content
        project_keywords = ['developed', 'built', 'created', 'implemented', 'designed', 'launched', 'intern', 'hackathon', 'project']
        lines = resume_text.split('\n')
        relevant_lines = []
        for line in lines:
            if any(keyword in line.lower() for keyword in project_keywords):
                relevant_lines.append(line.strip())
                if len(relevant_lines) >= 10:  # Limit to 10 relevant lines
                    break
        if relevant_lines:
            experience_text = " ".join(relevant_lines)[:1500]
        else:
            experience_text = resume_text[:500]  # Fallback to first 500 chars
    
    result["experience_summary"] = experience_text
    
    # Calculate years of experience from ALL sources: work, internships, projects, hackathons
    years_patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)',
        r'(\d+)\+?\s*yrs?\s+(?:of\s+)?(?:experience|exp)',
        r'experience[:\s]+.*?(\d+)\+?\s*years?',
        r'(\d+)\+?\s*months?\s+(?:of\s+)?(?:experience|internship|project)',
    ]
    
    total_years = 0.0
    months_total = 0.0
    
    # First, look for explicit year/month mentions
    for pattern in years_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        if matches:
            for m in matches:
                if m.isdigit():
                    if 'month' in pattern.lower():
                        months_total += float(m)
                    else:
                        total_years += float(m)
    
    # Convert months to years
    total_years += months_total / 12.0
    
    # If no explicit years, try to infer from dates (look for ALL date ranges)
    if total_years == 0.0:
        # Look for date patterns in experience sections (work, internships, projects)
        date_patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4}|Present|Current|Now)',
            r'(\w+\s+\d{4})\s*[-–]\s*(\w+\s+\d{4}|Present|Current|Now)',
            r'(\d{1,2}[/-]\d{4})\s*[-–]\s*(\d{1,2}[/-]\d{4}|Present|Current|Now)',
        ]
        
        all_date_ranges = []
        for pattern in date_patterns:
            date_matches = re.findall(pattern, resume_text, re.IGNORECASE)
            all_date_ranges.extend(date_matches)
        
        if all_date_ranges:
            # Calculate total months/years from all date ranges
            total_months = 0
            for start, end in all_date_ranges:
                try:
                    # Extract year from start date
                    start_year_match = re.search(r'(\d{4})', str(start))
                    end_year_match = re.search(r'(\d{4})', str(end)) if end.lower() not in ['present', 'current', 'now'] else None
                    
                    if start_year_match:
                        start_year = int(start_year_match.group(1))
                        if end_year_match:
                            end_year = int(end_year_match.group(1))
                        else:
                            end_year = datetime.now().year
                        
                        # Calculate months (approximate: assume 12 months per year)
                        months = (end_year - start_year) * 12
                        total_months += months
                except:
                    pass
            
            if total_months > 0:
                total_years = total_months / 12.0
    
    # If still 0, try to estimate from context (internships, projects mentioned)
    if total_years == 0.0:
        # Look for internship/project duration mentions
        duration_patterns = [
            r'(\d+)\s*(?:month|week|day)s?\s+(?:internship|project|hackathon)',
            r'(?:internship|project|hackathon).*?(\d+)\s*(?:month|week|day)s?',
        ]
        for pattern in duration_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            if matches:
                for m in matches:
                    if m.isdigit():
                        months_total += float(m) if 'month' in pattern else float(m) / 4.0  # Approximate weeks to months
                if months_total > 0:
                    total_years = months_total / 12.0
                    break
    
    # Round to 1 decimal place
    result["total_years_experience"] = round(total_years, 1) if total_years > 0 else 0.0
    
    # Extract education - look for college/university names and degrees
    education_patterns = [
        r'Education[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
        r'Academic\s+Background[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
        r'([A-Z][A-Z\s&]+(?:COLLEGE|UNIVERSITY|INSTITUTE|SCHOOL)[\s\w&]+)',  # Look for college names in ALL CAPS
    ]
    
    education_text = ""
    for pattern in education_patterns:
        match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
        if match:
            education_text = match.group(1).strip()
            break
    
    # Also look for college names directly in the text (common pattern: ALL CAPS college name)
    if not education_text:
        college_name_pattern = r'([A-Z][A-Z\s&]+(?:COLLEGE|UNIVERSITY|INSTITUTE|SCHOOL)[\s\w&]+)'
        college_match = re.search(college_name_pattern, resume_text)
        if college_match:
            education_text = college_match.group(1)
    
    # Parse education entries
    if education_text:
        # Look for degree patterns (B.E., B.S., Bachelor, Master, etc.)
        degree_pattern = r'(B\.E\.|B\.S\.|B\.A\.|M\.S\.|M\.A\.|Bachelor|Master|PhD|Ph\.D\.|Associate)[\s\w]*?(?:in|of)?\s*([A-Z][\w\s&]+?)(?:\s*[-–]\s*(\d{4}|Expected\s+\w+\s+\d{4}))?'
        degree_matches = re.findall(degree_pattern, education_text, re.IGNORECASE)
        
        # Also look for college names (often in ALL CAPS)
        college_name_pattern = r'([A-Z][A-Z\s&]+(?:COLLEGE|UNIVERSITY|INSTITUTE|SCHOOL)[\s\w&]+)'
        college_matches = re.findall(college_name_pattern, resume_text, re.IGNORECASE)
        
        if degree_matches:
            for degree_match in degree_matches:
                degree_type = degree_match[0].strip()
                field = degree_match[1].strip() if len(degree_match) > 1 and degree_match[1] else ""
                year = degree_match[2].strip() if len(degree_match) > 2 and degree_match[2] else ""
                
                # Find university name - look before and after the degree mention
                university = ""
                if college_matches:
                    # Use the first college name found
                    university = college_matches[0].strip()
                else:
                    # Try to find university name near the degree
                    context_start = max(0, resume_text.find(degree_type) - 100)
                    context_end = min(len(resume_text), resume_text.find(degree_type) + 200)
                    context = resume_text[context_start:context_end]
                    university_match = re.search(r'([A-Z][A-Z\s&]+(?:COLLEGE|UNIVERSITY|INSTITUTE|SCHOOL)[\s\w&]+)', context)
                    if university_match:
                        university = university_match.group(1).strip()
                
                result["education"].append({
                    "school": university or field or "Not specified",
                    "degree": degree_type + (" " + field if field and field != university else ""),
                    "dates": year or "Not specified"
                })
        elif college_matches:
            # If we found a college but no degree, still add it
            for college in college_matches[:1]:  # Take first college
                result["education"].append({
                    "school": college.strip(),
                    "degree": "Not specified",
                    "dates": "Not specified"
                })
    
    # Extract certifications
    cert_patterns = [
        r'Certifications?[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
        r'Certificates?[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)',
    ]
    
    cert_text = ""
    for pattern in cert_patterns:
        match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
        if match:
            cert_text = match.group(1).strip()
            break
    
    if cert_text:
        # Split by common delimiters
        certs = re.split(r'[,\n•\-]', cert_text)
        result["certifications"] = [c.strip() for c in certs if c.strip() and len(c.strip()) > 3]
    
    # Extract interests (usually at the end)
    interests_patterns = [
        r'Interests?[:\s]+(.+?)(?:$|\n\n)',
        r'Hobbies?[:\s]+(.+?)(?:$|\n\n)',
    ]
    
    interests_text = ""
    for pattern in interests_patterns:
        match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
        if match:
            interests_text = match.group(1).strip()
            break
    
    if interests_text:
        interests = re.split(r'[,\n•\-]', interests_text)
        result["interests"] = [i.strip() for i in interests if i.strip() and len(i.strip()) > 2]
    
    # Validate we got enough data and experience_summary is not education text
    has_name = result["name"] and result["name"] != "Unknown"
    has_skills = len(result["skills"]) > 0
    has_experience = result["experience_summary"] and len(result["experience_summary"]) > 50
    
    # Check if experience_summary looks like education text (common mistake)
    experience_text = result.get("experience_summary", "").upper()
    education_keywords = ["COLLEGE", "UNIVERSITY", "INSTITUTE", "SCHOOL", "BACHELOR", "MASTER", "DEGREE", "EDUCATION", "GRADUATE"]
    looks_like_education = any(keyword in experience_text for keyword in education_keywords) and len(experience_text) < 100
    
    # Also check if it's just a school name fragment
    if looks_like_education or (has_experience and not any(word in experience_text.lower() for word in ["developed", "built", "created", "worked", "intern", "project", "hackathon", "implemented", "designed"])):
        logger.warning(f"OCR parsing extracted education text as experience_summary: '{result.get('experience_summary', '')[:100]}', will fallback to LLM")
        return None
    
    if not (has_name and (has_skills or has_experience)):
        logger.warning(f"OCR parsing got insufficient data (name={has_name}, skills={has_skills}, exp={has_experience}), will fallback to LLM")
        return None
    
    logger.info(f"OCR parsing successful: name={result['name']}, skills={len(result['skills'])}, years={result['total_years_experience']}")
    return result


def parse_resume_with_llm_fallback(
    resume_text: str,
    model_name: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse resume with LLM using OCR-extracted text.
    
    Args:
        resume_text: OCR-extracted text from PDF (already extracted using pdfplumber)
        model_name: OpenAI model name
        openai_api_key: OpenAI API key
        
    Returns:
        Parsed resume dictionary
    """
    # Parse OCR-extracted text with direct OpenAI API call (faster than agent)
    logger.info("Parsing OCR-extracted text with direct LLM API call")
    try:
        from openai import OpenAI
        import os
        
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        client = OpenAI(api_key=openai_api_key)
        
        resume_prompt = f"""You are extracting structured data from a resume. Read the ENTIRE resume text below and extract information accurately.

RESUME TEXT (OCR-extracted from PDF):
{resume_text}

CRITICAL EXTRACTION RULES:

1. **experience_summary** (MOST IMPORTANT - READ CAREFULLY):
   - Read through the resume text and identify ALL sections that describe practical work/experience:
     * Work Experience / Professional Experience / Employment sections
     * Internship sections
     * Projects sections (personal projects, team projects, hackathons)
     * Research work
     * Achievements that involve building/creating something
   - Extract the ACTUAL CONTENT from these sections - what they did, what they built, key achievements
   - Write a 2-3 sentence summary that combines ALL this experience
   - DO NOT include: school names, education details, certifications, personal info, or random text
   - DO NOT make up experience - only use what's actually written in the resume
   - Example format: "Built [project name] using [technologies]. Worked as [role] at [company/org] where [achievement]. Participated in [hackathon/project] achieving [result]."
   - If you see sections like "Voice Sales Assistant", "RAG-based decision support", "deepfake detection", etc. - these are projects/experience, include them
   - If you see "SRI KRISHNA COLLEGE OF" or similar - this is EDUCATION, NOT experience - DO NOT include it

2. **total_years_experience**: 
   - Look for date ranges in experience sections (e.g., "2023 - 2024", "Jan 2024 - Present")
   - Calculate total months/years from ALL experience periods
   - Include internships, projects with durations, hackathons
   - Convert months to years (6 months = 0.5 years)
   - Sum all periods together
   - If no explicit dates, estimate from context but be conservative

3. **education**: 
   - Find university/college names (often in ALL CAPS like "SRI KRISHNA COLLEGE OF ENGINEERING")
   - Find degree types (B.E., B.S., Bachelor, Master, etc.)
   - Find graduation dates or expected dates
   - Return as array: [{{"school": "Full college name", "degree": "B.E. Computer Science", "dates": "Expected April 2027"}}]

4. **interests**: 
   - ONLY extract if there's an explicit "Interests" or "Hobbies" section
   - Examples: "Football", "Reading", "Travel"
   - If no such section exists, return empty array: []

5. **skills**: 
   - Extract ALL technologies, tools, frameworks mentioned anywhere in the resume
   - Include programming languages, libraries, frameworks, tools

OUTPUT FORMAT (valid JSON only, no markdown):
{{
  "name": "Full name from resume",
  "email": "email if found",
  "phone": "phone if found",
  "skills": ["Python", "TensorFlow", "Java", ...],
  "experience_summary": "2-3 sentences summarizing ALL practical experience from the resume",
  "total_years_experience": 1.5,
  "education": [{{"school": "College name", "degree": "Degree type", "dates": "Date"}}],
  "certifications": ["Cert names if found"],
  "interests": ["Only if explicit interests section exists, else []"]
}}

REMEMBER: 
- experience_summary must come from ACTUAL experience/project sections in the resume text
- Do NOT include education text in experience_summary
- Do NOT include random text fragments
- Read the resume carefully and extract what's actually there

Return ONLY valid JSON, no markdown formatting."""
        
        # Direct OpenAI API call (faster than agent)
        # Note: Some models don't support temperature=0, so we omit it to use default
        response = client.chat.completions.create(
            model=model_name or "gpt-4o-mini",
            messages=[{"role": "user", "content": resume_prompt}],
            response_format={"type": "json_object"} if "gpt-4" in (model_name or "").lower() or ("o1" in (model_name or "").lower() and "gpt-5" not in (model_name or "").lower()) else None
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            resume_json = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from response
            from agents import extract_json_from_response
            resume_json = extract_json_from_response(response_text)
        
        if resume_json:
            logger.info("LLM parsing successful")
            return resume_json
        else:
            logger.warning("LLM parsing failed to extract JSON, using minimal fallback")
            return {
                "name": "Unknown Candidate",
                "email": None,
                "phone": None,
                "skills": [],
                "experience_summary": resume_text[:500],
                "total_years_experience": 1.0,
                "education": [],
                "certifications": [],
                "interests": []
            }
            
    except Exception as e:
        logger.error(f"LLM fallback failed: {e}", exc_info=True)
        # Last resort
        return {
            "name": "Unknown Candidate",
            "email": None,
            "phone": None,
            "skills": [],
            "experience_summary": resume_text[:500],
            "total_years_experience": 1.0,
            "education": [],
            "certifications": [],
            "interests": []
        }

