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
    
    # Extract skills from ALL sections comprehensively
    # Comprehensive skill keywords list (technical + non-technical)
    skill_keywords = [
        # Programming Languages
        'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust', 'Swift', 'Kotlin',
        'PHP', 'Ruby', 'Perl', 'Scala', 'R', 'MATLAB', 'SQL', 'HTML', 'CSS', 'Shell', 'Bash',
        # Web Frameworks & Libraries
        'React', 'Angular', 'Vue', 'Next.js', 'Nuxt.js', 'Node.js', 'Express', 'Django', 'Flask', 
        'FastAPI', 'Spring', 'Spring Boot', 'Laravel', 'Rails', 'ASP.NET', '.NET',
        'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'Scikit', 'Pandas', 'NumPy', 'OpenCV',
        'Matplotlib', 'Seaborn', 'Plotly', 'D3.js', 'Pillow', 'SciPy',
        # Databases
        'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Cassandra', 'DynamoDB',
        'SQLite', 'Oracle', 'SQL Server', 'Neo4j', 'Weaviate',
        # Cloud & DevOps
        'AWS', 'Azure', 'GCP', 'Google Cloud', 'Heroku', 'DigitalOcean', 'Docker', 'Kubernetes',
        'Jenkins', 'CI/CD', 'Git', 'GitHub', 'GitLab', 'Bitbucket', 'Terraform', 'Ansible',
        'Puppet', 'Chef', 'Vagrant',
        # Tools & Others
        'Linux', 'Unix', 'Windows', 'MacOS', 'JIRA', 'Confluence', 'Slack', 'Postman', 'Swagger',
        'GraphQL', 'REST API', 'Agile', 'Scrum', 'Kanban', 'Waterfall', 'DevOps', 'TDD', 'BDD',
        # AI/ML/Data Science
        'Machine Learning', 'ML', 'Deep Learning', 'AI', 'Artificial Intelligence', 'Data Science',
        'NLP', 'Natural Language Processing', 'Computer Vision', 'Neural Networks', 'Data Analysis',
        'Data Engineering', 'LLM', 'Large Language Models', 'RAG', 'Retrieval-Augmented Generation',
        'CNN', 'RNN', 'LSTM', 'Transformer', 'GPT', 'BERT',
        # BI & Visualization
        'Tableau', 'Power BI', 'Looker', 'Qlik', 'Business Intelligence',
        # Non-Technical Skills & Software
        'Microsoft Office', 'Excel', 'Word', 'PowerPoint', 'Outlook', 'Google Workspace',
        'Salesforce', 'SAP', 'Oracle', 'QuickBooks', 'HubSpot', 'Google Analytics',
        'Adobe Creative Suite', 'Photoshop', 'Illustrator', 'InDesign',
        'Epic', 'Cerner', 'Meditech', 'Bloomberg', 'Reuters',
        # Soft Skills
        'Communication', 'Leadership', 'Teamwork', 'Problem-solving', 'Time Management',
        'Project Management', 'Agile', 'Scrum', 'Customer Service', 'Presentation Skills'
    ]
    
    found_skills = []
    resume_lower = resume_text.lower()
    
    # STEP 1: Extract from explicit Skills/Tech Stack sections
    skills_section_patterns = [
        r'Skills?[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
        r'Technical\s+Skills?[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
        r'Tech\s+Stack[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
        r'Technologies?[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
        r'Tools?[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
    ]
    
    skills_section_text = ""
    for pattern in skills_section_patterns:
        match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
        if match:
            skills_section_text = match.group(1).strip()
            break
    
    # STEP 2: Extract from Experience/Projects sections
    experience_section_patterns = [
        r'Experience[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
        r'Work\s+Experience[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
        r'Projects?[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
        r'Internship[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
        r'Hackathon[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
    ]
    
    experience_section_text = ""
    for pattern in experience_section_patterns:
        matches = re.finditer(pattern, resume_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            experience_section_text += " " + match.group(1).strip()
    
    # STEP 3: Extract from Education/Coursework
    education_section_patterns = [
        r'Education[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
        r'Coursework[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
    ]
    
    education_section_text = ""
    for pattern in education_section_patterns:
        match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
        if match:
            education_section_text += " " + match.group(1).strip()
    
    # STEP 4: Extract from Certifications
    cert_section_patterns = [
        r'Certifications?[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
        r'Certificates?[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Z\s]+|$)',
    ]
    
    cert_section_text = ""
    for pattern in cert_section_patterns:
        match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
        if match:
            cert_section_text += " " + match.group(1).strip()
    
    # Combine all sections for comprehensive search
    all_sections_text = (skills_section_text + " " + experience_section_text + " " + 
                        education_section_text + " " + cert_section_text + " " + resume_text).lower()
    
    # Extract skills from all sections
    for skill in skill_keywords:
        skill_lower = skill.lower()
        # Check if skill appears in any section
        if skill_lower in all_sections_text:
            # Normalize skill name (use original case from keyword list)
            if skill not in found_skills:
                found_skills.append(skill)
    
    # Also extract skills mentioned in Skills section as individual items (comma/line separated)
    if skills_section_text:
        # Split by common delimiters and check each item
        skill_items = re.split(r'[,\n•\-•\|\/]', skills_section_text)
        for item in skill_items:
            item_clean = item.strip()
            if len(item_clean) > 2 and len(item_clean) < 50:  # Reasonable skill name length
                # Check if it matches any known skill (case-insensitive)
                item_lower = item_clean.lower()
                for known_skill in skill_keywords:
                    if known_skill.lower() == item_lower or known_skill.lower() in item_lower:
                        if known_skill not in found_skills:
                            found_skills.append(known_skill)
                # If it's a reasonable skill-like word, add it (capitalize properly)
                if item_clean and not any(char.isdigit() for char in item_clean[:3]):  # Skip if starts with numbers
                    # Check if it looks like a skill (has letters, reasonable length)
                    if re.match(r'^[A-Za-z][A-Za-z\s\.\+\#]+$', item_clean):
                        skill_normalized = item_clean.strip().title()
                        if skill_normalized not in found_skills and len(skill_normalized) > 2:
                            found_skills.append(skill_normalized)
    
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
    model_name: Optional[str] = None,
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

5. **skills** (MOST CRITICAL - USE MULTI-STEP EXTRACTION PROCESS):
   **CRITICAL**: You MUST extract EVERY skill mentioned ANYWHERE in the resume, even if mentioned only once.
   **CRITICAL**: Read the ENTIRE resume text line by line - do not skip any section.
   **CRITICAL**: If a skill is mentioned in the resume text, it MUST appear in the skills array.
   **CRITICAL**: Missing a skill that is mentioned in the resume is a critical error.
   
   STEP 1: Extract from Skills/Tech Stack Section
   - Look for explicit "Skills", "Technical Skills", "Tech Stack", "Technologies", "Tools", or similar sections
   - Extract every item listed in these sections
   - Include variations (e.g., "Python 3", "Python3" → "Python")
   
   STEP 2: Extract from Experience/Work Sections
   - Read through ALL experience descriptions line by line
   - Look for technologies mentioned in project descriptions
   - Extract technologies used in each role/project
   - Examples: "Built REST API using Flask" → extract "Flask", "REST API"
   - Examples: "Developed ML model with TensorFlow and Keras" → extract "TensorFlow", "Keras", "Machine Learning"
   - Examples: "Deployed on AWS using Docker" → extract "AWS", "Docker"
   
   STEP 3: Extract from Projects Section
   - Read through ALL project descriptions
   - Extract technologies, frameworks, tools, libraries mentioned
   - Look for tech stacks, architecture details, implementation details
   - Examples: "React frontend with Node.js backend" → extract "React", "Node.js"
   - Examples: "Computer Vision project using OpenCV" → extract "OpenCV", "Computer Vision"
   
   STEP 4: Extract from Education/Coursework
   - Look for relevant coursework that mentions technologies
   - Extract programming languages, tools, frameworks mentioned in courses
   - Examples: "Data Structures in C++" → extract "C++"
   - Examples: "Database Systems course using PostgreSQL" → extract "PostgreSQL"
   
   STEP 5: Extract from Certifications
   - Extract technologies mentioned in certification names or descriptions
   - Examples: "AWS Certified Solutions Architect" → extract "AWS"
   - Examples: "Oracle Certified Java Developer" → extract "Java", "Oracle"
   
   STEP 6: Extract from Any Other Sections
   - Check Interests/Hobbies for technical mentions
   - Check Achievements/Awards for technology references
   - Check Publications/Research for technical tools
   - Check Hackathons/Competitions for technologies used
   
   STEP 7: Extract NON-TECHNICAL SKILLS (CRITICAL FOR ALL JOB TYPES):
   - Soft Skills: Communication, Leadership, Teamwork, Problem-solving, Time Management, Organization, etc.
   - Language Skills: All languages mentioned with proficiency levels if specified
   - Industry-Specific Skills: Extract ALL domain-specific skills (Healthcare, Finance, Marketing, Sales, Legal, Education, etc.)
   - Software/Tools: Extract ALL software, tools, and systems mentioned (Microsoft Office, Salesforce, SAP, QuickBooks, etc.)
   - Certifications/Licenses: Extract ALL certifications and licenses mentioned
   - Methodologies/Frameworks: Project Management methodologies, Business frameworks, Quality standards
   
   COMPREHENSIVE SKILL CATEGORIES TO EXTRACT:
   - Programming Languages: Python, Java, JavaScript, TypeScript, C++, C#, Ruby, PHP, Swift, Kotlin, Go, Rust, Scala, R, MATLAB
   - Web Frameworks: React, Angular, Vue, Next.js, Django, Flask, FastAPI, Spring Boot, Express.js, Laravel, Rails
   - Libraries: numpy, pandas, scikit-learn, TensorFlow, PyTorch, Keras, OpenCV, Matplotlib, Seaborn
   - Databases: PostgreSQL, MySQL, MongoDB, Redis, SQLite, Oracle, SQL Server, Cassandra, DynamoDB, Elasticsearch
   - Cloud Platforms: AWS, Azure, GCP, Heroku, DigitalOcean
   - DevOps Tools: Docker, Kubernetes, Jenkins, Git, GitHub, GitLab, CI/CD, Terraform, Ansible
   - Tools: Git, Jira, Confluence, Slack, Postman, Swagger, GraphQL, REST API
   - Methodologies: Agile, Scrum, Kanban, Waterfall, DevOps, CI/CD, TDD, BDD
   - Specializations: Machine Learning, Deep Learning, AI, NLP, Computer Vision, Data Science, Data Engineering
   - Soft Skills: Communication, Leadership, Teamwork, Problem-solving, Time Management, Critical Thinking, Adaptability
   - Industry Software: Salesforce, SAP, Oracle, Microsoft Office, Google Workspace, Adobe Creative Suite, QuickBooks, Epic, Cerner
   
   CRITICAL RULES:
   - Extract EVERY skill mentioned (technical AND non-technical), even if mentioned only once
   - Do NOT skip non-technical skills - they are just as important as technical skills
   - Extract skills from ALL sections: Skills section, Experience, Projects, Education, Certifications, Achievements, Hackathons
   - Normalize variations to standard names (e.g., "Python 3.9" → "Python", "Microsoft Excel" → "Excel")
   - Include both full names and abbreviations if both are mentioned (e.g., "Customer Relationship Management" and "CRM")
   - Don't skip skills because they seem minor, common, or non-technical
   - Be exhaustive - it's better to include too many skills than miss important ones
   - Return as a complete, deduplicated array with ALL skills found across ALL sections (technical + non-technical)

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
        
        # Use task-specific model selection for OCR fallback
        from model_config import get_model_for_task, supports_json_mode
        if model_name is None:
            # Use task-specific model
            ocr_model = get_model_for_task("ocr_fallback", json_mode=True)
            model_name = ocr_model.model_name
            use_json_mode = True
        else:
            # Legacy: model_name provided, check JSON support
            use_json_mode = supports_json_mode(model_name)
        
        # Direct OpenAI API call (faster than agent)
        # CRITICAL: Always use temperature=0 for deterministic output
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": resume_prompt}],
            temperature=0,  # MUST be 0 for deterministic output
            response_format={"type": "json_object"} if use_json_mode else None
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

