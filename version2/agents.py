from __future__ import annotations

from typing import List, Dict, Any, Optional, Union
import json
import re
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
import operator

# Firecrawl tool wrapper (since FirecrawlTools from phidata doesn't exist in langchain)
class FirecrawlTool(BaseTool):
    name: str = "firecrawl_scrape"
    description: str = "Scrape content from a URL using Firecrawl API"
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        try:
            from firecrawl import FirecrawlApp
            self.client = FirecrawlApp(api_key=self.api_key) if self.api_key else None
        except ImportError:
            self.client = None
    
    def _run(self, url: str) -> str:
        """Scrape URL using Firecrawl."""
        if not self.client:
            return f"Error: Firecrawl API key not configured. URL: {url}"
        try:
            result = self.client.scrape_url(url)
            return result.get('markdown', result.get('content', str(result)))
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"
    
    async def _arun(self, url: str) -> str:
        """Async scrape URL using Firecrawl."""
        return self._run(url)


# Agent wrapper class to maintain compatibility with phidata Agent interface
class Agent:
    """Wrapper class that mimics phidata Agent interface using LangChain/LangGraph."""
    
    def __init__(
        self,
        name: str = None,
        role: str = None,
        model: ChatOpenAI = None,
        instructions: List[str] = None,
        tools: List[BaseTool] = None,
        show_tool_calls: bool = True,
        markdown: bool = True,
        response_format: Dict[str, Any] = None
    ):
        self.name = name or "Agent"
        self.role = role or ""
        self.model = model
        self.instructions = instructions or []
        self.tools = tools or []
        self.show_tool_calls = show_tool_calls
        self.markdown = markdown
        self.response_format = response_format
        
        # Bind tools to model if available
        if self.model and self.tools:
            self.model_with_tools = self.model.bind_tools(self.tools)
        else:
            self.model_with_tools = self.model
    
    def run(self, input_data: Union[str, Dict[str, Any]], stream: bool = False) -> Union[str, Any]:
        """Run the agent with input data."""
        from langchain_core.messages import ToolMessage
        
        # Build system message from instructions
        system_prompt = ""
        if self.role:
            system_prompt += f"Role: {self.role}\n\n"
        if self.instructions:
            system_prompt += "\n".join(self.instructions)
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Add input as human message
        if isinstance(input_data, dict):
            input_text = json.dumps(input_data, indent=2)
        else:
            input_text = str(input_data)
        messages.append(HumanMessage(content=input_text))
        
        # Get response from model
        if self.model_with_tools and self.tools:
            # Use tool-calling model with iterative tool execution
            max_iterations = 5  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                response = self.model_with_tools.invoke(messages)
                messages.append(response)
                
                # Check if model wants to call tools
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # Execute tools
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else getattr(tool_call, 'name', '')
                        tool_args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                        tool_call_id = tool_call.get('id', '') if isinstance(tool_call, dict) else getattr(tool_call, 'id', '')
                        
                        # Find the tool
                        tool = None
                        for t in self.tools:
                            if t.name == tool_name:
                                tool = t
                                break
                        
                        if tool:
                            try:
                                # Execute tool - handle both sync and async
                                # For tools like FirecrawlTool that take a single string argument
                                if isinstance(tool_args, dict):
                                    # If tool expects a single positional arg (like URL), extract it
                                    if len(tool_args) == 1:
                                        # Get the first (and likely only) value
                                        tool_result = tool._run(list(tool_args.values())[0])
                                    else:
                                        tool_result = tool._run(**tool_args)
                                elif isinstance(tool_args, str):
                                    tool_result = tool._run(tool_args)
                                else:
                                    tool_result = tool._run(tool_args)
                                
                                # Add tool result to messages
                                messages.append(ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_call_id
                                ))
                            except Exception as e:
                                messages.append(ToolMessage(
                                    content=f"Error executing tool {tool_name}: {str(e)}",
                                    tool_call_id=tool_call_id
                                ))
                    iteration += 1
                else:
                    # No more tool calls, return final response
                    return response.content or str(response)
            
            # If we hit max iterations, return the last response
            return messages[-1].content if messages else str(response)
        else:
            # Simple model without tools
            response = self.model.invoke(messages)
            return response.content or str(response)


def get_model_config(model_name: str, default_temperature: float = 0) -> ChatOpenAI:
    """
    Get model configuration with temperature support check.
    
    Some models (like o1, o1-mini, gpt-5-mini) don't support custom temperature.
    Only set temperature for models that support it.
    
    Args:
        model_name: Name of the model
        default_temperature: Desired temperature (only used if model supports it)
    
    Returns:
        ChatOpenAI model instance
    """
    # Models that don't support temperature customization
    models_without_temperature = [
        "o1", "o1-mini", "o1-preview", "o1-2024",
        "gpt-5-mini", "gpt-5",  # Some GPT-5 models may not support it
    ]
    
    # Check if model supports temperature
    model_lower = model_name.lower()
    supports_temperature = not any(no_temp in model_lower for no_temp in models_without_temperature)
    
    # Build model kwargs
    model_kwargs = {"model": model_name}
    
    if supports_temperature:
        model_kwargs["temperature"] = default_temperature
    
    # JSON mode support (only for certain models that support it)
    # Note: o1 models support JSON mode, but gpt-5 models may not
    # IMPORTANT: When using response_format={"type": "json_object"}, the prompt MUST contain the word "json"
    response_format = None
    if "gpt-4" in model_lower or ("o1" in model_lower and "gpt-5" not in model_lower):
        response_format = {"type": "json_object"}
        model_kwargs["model_kwargs"] = {"response_format": response_format}
    
    return ChatOpenAI(**model_kwargs)


# Removed unused build_orchestrator function - not used in production code


def build_resume_parser(model_name: str) -> Agent:
    """Agent that extracts structured information from resume text."""
    model = get_model_config(model_name, default_temperature=0)
    return Agent(
        name="Resume Parser",
        role="Extract and structure all information from resume OCR text",
        model=model,
        instructions=[
            "Parse raw OCR text from resume and extract ALL information.",
            "You MUST return ONLY valid JSON (no markdown, no code blocks, no explanations).",
            "CRITICAL: Your response must be valid JSON that can be parsed directly.",
            "IMPORTANT: Return your response as a JSON object.",
            "",
            "Extract these fields:",
            "- name: Full name of candidate",
            "- email: Email address",
            "- phone: Phone number",
            "- skills: Array of ALL technical skills, tools, frameworks, languages",
            "- experience_summary: String summary of work experience",
            "- total_years_experience: Number (calculate total years from all roles)",
            "- education: Array of objects with school, degree, dates",
            "- certifications: Array of certification names",
            "- interests: Array of interests/hobbies",
            "",
            "Example output format:",
            '{"name": "John Doe", "email": "john@example.com", "phone": "+1234567890", "skills": ["Python", "Java"], "experience_summary": "2 years in AI", "total_years_experience": 2, "education": [{"school": "MIT", "degree": "BS CS", "dates": "2020"}], "certifications": ["AWS"], "interests": ["AI", "ML"]}',
            "",
            "CRITICAL RULES:",
            "1. Return ONLY the JSON object, nothing else",
            "2. All fields must be present (use empty string or empty array if not found)",
            "3. Do not add any text before or after the JSON",
            "4. Ensure all strings are properly escaped",
        ],
        show_tool_calls=False,
        markdown=False,
    )


def build_scraper(api_key: str = None) -> Agent:
    """Agent that scrapes individual job postings."""
    # Use provided api_key, or get from environment (same as SAMPLE_FIRECRAWL.PY)
    firecrawl_api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    model = get_model_config(model_name, default_temperature=0)
    return Agent(
        name="Job Scraper",
        role="Extract complete job posting information from URLs",
        tools=[FirecrawlTool(api_key=firecrawl_api_key)],  # Pass API key
        model=model,
        instructions=[
            "Given a job URL, extract ALL available information. CRITICAL: You MUST extract the following REQUIRED fields:",
            "",
            "OUTPUT FORMAT: Return a structured response with clearly labeled sections.",
            "",
            "1. **Job title** (REQUIRED - exact title from posting):",
            "   - Look for the main job title/position name in headings, titles, or prominent text",
            "   - Examples: 'Full Stack Developer', 'Software Engineer', 'Data Scientist'",
            "   - If not found, return 'Not specified'",
            "   - Format: 'Job Title: [exact title]'",
            "",
            "2. **Company name** (REQUIRED - name of the hiring company):",
            "   - Look for company name in various formats: 'by [Company]', 'Company:', 'at [Company]', 'from [Company]'",
            "   - Check for company names near the job title or in headers",
            "   - Examples: 'Michael Page Technology', 'Google', 'Microsoft Corporation'",
            "   - If not found, return 'Not specified'",
            "   - Format: 'Company Name: [company name]'",
            "",
            "3. **Job Description:**",
            "   - Extract the complete job description",
            "   - Include all responsibilities, requirements, and details",
            "   - Format: 'Job Description: [full description]'",
            "",
            "4. **Required Skills:**",
            "   - List each skill separately",
            "   - Format: 'Required Skills: [skill1], [skill2], [skill3]'",
            "",
            "5. **Required Experience:**",
            "   - Extract years and type of experience required",
            "   - Format: 'Required Experience: [details]'",
            "",
            "6. **Qualifications:**",
            "   - List education and qualification requirements",
            "   - Format: 'Qualifications: [details]'",
            "",
            "7. **Responsibilities:**",
            "   - List key responsibilities",
            "   - Format: 'Responsibilities: [list]'",
            "",
            "8. **Salary/Compensation:**",
            "   - Extract if mentioned",
            "   - Format: 'Salary: [details]' or 'Salary: Not specified'",
            "",
            "9. **Location:**",
            "   - Extract job location",
            "   - Format: 'Location: [location]' or 'Location: Not specified'",
            "",
            "10. **Job Type:**",
            "   - Extract job type (full-time, internship, etc.)",
            "   - Format: 'Job Type: [type]' or 'Job Type: Not specified'",
            "",
            "CRITICAL RULES:",
            "- Job title and Company name are REQUIRED fields - extract them accurately",
            "- Use consistent formatting with clear labels",
            "- If a field is not found, explicitly state 'Not specified'",
            "- Do not truncate or cut off text in the middle of sentences",
            "- Ensure all extracted text is complete and properly formatted",
            "- Avoid repeating or duplicating information",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def normalize_skill(skill: str) -> str:
    """Normalize skill name for comparison (lowercase, remove special chars)."""
    if not skill:
        return ""
    # Convert to lowercase and remove special characters
    normalized = re.sub(r'[^\w\s]', '', skill.lower().strip())
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def extract_skills_from_text(text: str) -> List[str]:
    """Extract skill keywords from text - comprehensive extraction."""
    if not text:
        return []
    
    # Comprehensive technical skills patterns
    skill_patterns = [
        # Programming languages
        r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|php|swift|kotlin|go|rust|scala|r)\b',
        # Web frameworks
        r'\b(react|angular|vue|node\.?js|django|flask|spring|express|laravel|rails)\b',
        # Databases
        r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|oracle|cassandra|dynamodb)\b',
        # Cloud & DevOps
        r'\b(aws|azure|gcp|docker|kubernetes|terraform|jenkins|ci/cd|ansible|puppet)\b',
        # ML/AI/Data Science
        r'\b(machine learning|ml|deep learning|ai|artificial intelligence|tensorflow|pytorch|scikit-learn|keras)\b',
        r'\b(data science|data analysis|data engineering|data annotation|data curation)\b',
        r'\b(pandas|numpy|matplotlib|seaborn|scipy|opencv|computer vision|cv)\b',
        r'\b(nlp|natural language processing|llm|large language models)\b',
        r'\b(cnn|convolutional neural network|rnn|lstm|transformer)\b',
        # Data visualization & BI
        r'\b(tableau|power bi|looker|qlik|d3\.js|plotly)\b',
        # Frontend
        r'\b(html|css|sass|less|bootstrap|tailwind|jquery|webpack)\b',
        # Version control & collaboration
        r'\b(git|github|gitlab|bitbucket|svn|mercurial)\b',
        # Methodologies
        r'\b(agile|scrum|kanban|jira|confluence|waterfall)\b',
        # Annotation & labeling tools
        r'\b(cvat|labelbox|via|label studio|supervisely|annotation|labeling)\b',
        # Other technical
        r'\b(api|rest|graphql|microservices|etl|data pipeline|spark|hadoop)\b',
    ]
    
    found_skills = []
    text_lower = text.lower()
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        found_skills.extend(matches)
    
    # Also extract common domain terms that indicate requirements
    domain_terms = [
        r'\b(data\s+annotation)\b',
        r'\b(data\s+curation)\b',
        r'\b(data\s+cleaning)\b',
        r'\b(data\s+engineering)\b',
        r'\b(machine\s+learning)\b',
        r'\b(computer\s+vision)\b',
        r'\b(image\s+processing)\b',
        r'\b(model\s+training)\b',
    ]
    
    for pattern in domain_terms:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        found_skills.extend(matches)
    
    return list(set(found_skills))


def calculate_skills_match(candidate_skills: List[str], job_skills: List[str], job_description: str = "") -> float:
    """Calculate skills match score (0.0 to 1.0). Weight: 40%"""
    if not job_skills and not job_description:
        return 0.5  # Default if no skills info
    
    # Normalize all skills
    candidate_skills_normalized = [normalize_skill(s) for s in candidate_skills if s]
    job_skills_normalized = [normalize_skill(s) for s in job_skills if s]
    
    # Extract skills from job description if needed
    if job_description and not job_skills_normalized:
        extracted_skills = extract_skills_from_text(job_description)
        job_skills_normalized.extend([normalize_skill(s) for s in extracted_skills])
    
    if not job_skills_normalized:
        return 0.5
    
    # Count exact matches
    exact_matches = sum(1 for js in job_skills_normalized if js in candidate_skills_normalized)
    
    # Count partial matches (substring matches)
    partial_matches = 0
    for js in job_skills_normalized:
        if js not in candidate_skills_normalized:
            # Check if any candidate skill contains or is contained in job skill
            for cs in candidate_skills_normalized:
                if js in cs or cs in js:
                    partial_matches += 1
                    break
    
    total_matches = exact_matches + (partial_matches * 0.5)
    match_ratio = min(1.0, total_matches / len(job_skills_normalized))
    
    return match_ratio


def calculate_experience_match(
    candidate_years: Optional[float],
    job_experience_level: Optional[str],
    job_description: str = ""
) -> float:
    """Calculate experience match score (0.0 to 1.0). Weight: 30%"""
    if candidate_years is None:
        candidate_years = 0.0
    
    # Extract years from job description if experience_level not provided
    required_years = None
    if job_experience_level:
        # Try to extract number of years from experience_level
        years_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yr)', job_experience_level.lower())
        if years_match:
            required_years = float(years_match.group(1))
    
    if required_years is None and job_description:
        # Try to extract from description
        years_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yr)\s*(?:of|experience)',
            r'experience[:\s]+(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yr)',
        ]
        for pattern in years_patterns:
            match = re.search(pattern, job_description.lower())
            if match:
                required_years = float(match.group(1))
                break
    
    if required_years is None:
        # Default scoring based on keywords
        desc_lower = (job_description or "").lower()
        if any(word in desc_lower for word in ['senior', 'lead', 'principal', '5+', '10+']):
            required_years = 5.0
        elif any(word in desc_lower for word in ['mid', 'intermediate', '3+', '4+']):
            required_years = 3.0
        elif any(word in desc_lower for word in ['junior', 'entry', '0-2', '1-2']):
            required_years = 1.0
        else:
            required_years = 2.0  # Default assumption
    
    # Calculate match: closer to required = higher score
    if candidate_years >= required_years:
        # Overqualified: still good but slightly penalize
        ratio = min(1.0, required_years / max(candidate_years, 1.0))
        return 0.9 * ratio + 0.1  # Cap at 0.9-1.0
    else:
        # Underqualified: linear penalty
        ratio = candidate_years / max(required_years, 1.0)
        return max(0.0, ratio * 0.8)  # Cap at 0.8 for underqualified


def calculate_role_fit(
    candidate_experience_summary: Optional[str],
    job_title: Optional[str],
    job_description: str = ""
) -> float:
    """Calculate role fit score (0.0 to 1.0). Weight: 20%"""
    if not job_description and not job_title:
        return 0.5
    
    text_to_analyze = f"{job_title or ''} {job_description or ''}".lower()
    candidate_text = (candidate_experience_summary or "").lower()
    
    if not candidate_text:
        return 0.3
    
    # Extract key role-related keywords from job
    role_keywords = []
    
    # Job title keywords
    if job_title:
        title_words = re.findall(r'\b\w+\b', job_title.lower())
        role_keywords.extend([w for w in title_words if len(w) > 3])
    
    # Common role-related terms
    common_terms = [
        'developer', 'engineer', 'analyst', 'manager', 'executive', 'specialist',
        'consultant', 'architect', 'scientist', 'designer', 'administrator',
        'billing', 'finance', 'accounting', 'sales', 'marketing', 'data',
        'software', 'web', 'mobile', 'backend', 'frontend', 'full stack'
    ]
    
    for term in common_terms:
        if term in text_to_analyze:
            role_keywords.append(term)
    
    if not role_keywords:
        return 0.5
    
    # Count how many role keywords appear in candidate experience
    matches = sum(1 for keyword in role_keywords if keyword in candidate_text)
    match_ratio = matches / len(role_keywords)
    
    return min(1.0, match_ratio * 1.2)  # Slight boost for good matches


def calculate_growth_potential(
    candidate_years: Optional[float],
    candidate_skills: List[str],
    job_description: str = ""
) -> float:
    """Calculate growth potential score (0.0 to 1.0). Weight: 10%"""
    # Growth potential is higher if:
    # 1. Candidate has some but not all skills (room to grow)
    # 2. Candidate has reasonable experience (not too junior, not too senior)
    # 3. Job mentions learning/growth opportunities
    
    growth_score = 0.5  # Base score
    
    # Check if job mentions growth/learning
    if job_description:
        desc_lower = job_description.lower()
        growth_keywords = ['learn', 'growth', 'development', 'training', 'mentor', 'opportunity', 'career']
        if any(keyword in desc_lower for keyword in growth_keywords):
            growth_score += 0.2
    
    # Adjust based on experience level (sweet spot: 1-5 years)
    if candidate_years is not None:
        if 1.0 <= candidate_years <= 5.0:
            growth_score += 0.2
        elif candidate_years < 1.0:
            growth_score += 0.1
        # Senior candidates have less growth potential score
    
    return min(1.0, growth_score)


def calculate_job_match_score(
    candidate_profile: Dict[str, Any],
    job_details: Union[Dict[str, Any], str]
) -> Dict[str, Any]:
    """
    Deterministic function to calculate job-candidate match score.
    
    Args:
        candidate_profile: Dict with keys: skills, total_years_experience, experience_summary, etc.
        job_details: Dict with keys: job_title, company, description, skills_needed, requirements, etc.
                   OR a string (will try to parse JSON)
    
    Returns:
        Dict with: match_score, key_matches, requirements_met, total_requirements, reasoning, mismatch_areas
    """
    # Parse job_details if it's a string
    if isinstance(job_details, str):
        try:
            # Try to extract JSON from string
            job_data = extract_json_from_response(job_details)
            if not job_data:
                # Try direct JSON parse
                job_data = json.loads(job_details)
        except:
            # If parsing fails, try to extract from text
            job_data = {
                "description": job_details,
                "job_title": None,
                "skills_needed": [],
                "requirements": []
            }
    else:
        job_data = job_details
    
    # Extract candidate data
    candidate_skills = candidate_profile.get("skills", []) or []
    candidate_years = candidate_profile.get("total_years_experience")
    candidate_experience = candidate_profile.get("experience_summary", "") or ""
    candidate_certs = candidate_profile.get("certifications", []) or []
    candidate_education = candidate_profile.get("education", []) or []
    
    # Extract job data
    job_title = job_data.get("job_title") or job_data.get("title")
    job_description = job_data.get("description", "") or ""
    job_skills = job_data.get("skills_needed", []) or job_data.get("required_skills", []) or []
    job_requirements = job_data.get("requirements", []) or []
    job_experience_level = job_data.get("experience_level") or job_data.get("required_experience")
    
    # Calculate component scores
    skills_score = calculate_skills_match(candidate_skills, job_skills, job_description)
    experience_score = calculate_experience_match(candidate_years, job_experience_level, job_description)
    role_fit_score = calculate_role_fit(candidate_experience, job_title, job_description)
    growth_score = calculate_growth_potential(candidate_years, candidate_skills, job_description)
    
    # Weighted final score
    match_score = (
        skills_score * 0.40 +
        experience_score * 0.30 +
        role_fit_score * 0.20 +
        growth_score * 0.10
    )
    
    # Ensure score is between 0.0 and 1.0
    match_score = max(0.0, min(1.0, match_score))
    
    # Find key matches
    key_matches = []
    candidate_skills_normalized = [normalize_skill(s) for s in candidate_skills]
    job_skills_normalized = [normalize_skill(s) for s in job_skills]
    
    for js in job_skills_normalized:
        if js in candidate_skills_normalized:
            # Find original skill name
            for cs in candidate_skills:
                if normalize_skill(cs) == js:
                    key_matches.append(cs)
                    break
    
    # Count requirements met
    total_requirements = len(job_requirements) + len(job_skills_normalized)
    if total_requirements == 0:
        # Estimate from description
        total_requirements = max(5, len(job_skills_normalized))
    
    requirements_met = len(key_matches)
    if candidate_years and job_experience_level:
        # Check if experience requirement is met
        required_years_match = re.search(r'(\d+(?:\.\d+)?)', str(job_experience_level))
        if required_years_match:
            required_years = float(required_years_match.group(1))
            if candidate_years >= required_years:
                requirements_met += 1
            total_requirements += 1
    
    # Find mismatch areas
    mismatch_areas = []
    for js in job_skills_normalized:
        if js not in candidate_skills_normalized:
            # Find original skill name
            for orig_skill in (job_skills or []):
                if normalize_skill(orig_skill) == js:
                    mismatch_areas.append(orig_skill)
                    break
    
    # Generate reasoning
    if match_score >= 0.7:
        reasoning = f"Excellent match. Candidate has {len(key_matches)} key skills and {candidate_years or 0:.1f} years of experience."
    elif match_score >= 0.5:
        reasoning = f"Good match. Candidate has {len(key_matches)} matching skills with {candidate_years or 0:.1f} years of experience."
    elif match_score >= 0.3:
        reasoning = f"Weak match. Candidate has some relevant skills but may lack {len(mismatch_areas)} key requirements."
    else:
        reasoning = f"Poor match. Significant gaps in required skills and experience."
    
    return {
        "match_score": round(match_score, 3),
        "key_matches": key_matches[:10],  # Limit to top 10
        "requirements_met": requirements_met,
        "total_requirements": max(1, total_requirements),
        "reasoning": reasoning,
        "mismatch_areas": mismatch_areas[:5]  # Limit to top 5
    }


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text response (helper function)."""
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
        except:
            pass
    
    return None


def validate_agent_response(response: Union[str, Any], expected_fields: List[str] = None) -> Dict[str, Any]:
    """
    Validate and clean agent response.
    
    Args:
        response: Raw response from agent
        expected_fields: List of expected field names (optional)
    
    Returns:
        Dict with validation results and cleaned data
    """
    result = {
        "valid": False,
        "data": {},
        "errors": [],
        "warnings": []
    }
    
    # Convert response to string if needed
    if hasattr(response, 'content'):
        response_text = str(response.content)
    elif hasattr(response, 'messages') and response.messages:
        last_msg = response.messages[-1]
        response_text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
    else:
        response_text = str(response)
    
    response_text = response_text.strip()
    
    # Check for truncation indicators
    if response_text.endswith('...') or '...' in response_text[-50:]:
        result["warnings"].append("Response may be truncated")
    
    # Try to extract JSON
    parsed_json = extract_json_from_response(response_text)
    if parsed_json:
        result["data"] = parsed_json
        result["valid"] = True
        
        # Validate expected fields if provided
        if expected_fields:
            missing_fields = [field for field in expected_fields if field not in parsed_json]
            if missing_fields:
                result["warnings"].append(f"Missing fields: {', '.join(missing_fields)}")
    else:
        result["errors"].append("Could not extract valid JSON from response")
        result["data"] = {"raw_response": response_text}
    
    return result


def clean_scraper_response(response_text: str) -> Dict[str, Any]:
    """
    Clean and structure scraper agent response.
    Handles inconsistent formatting and extracts structured data.
    """
    result = {
        "job_title": None,
        "company": None,
        "description": None,
        "skills_needed": [],
        "requirements": [],
        "experience_level": None,
        "location": None,
        "salary": None,
        "job_type": None,
    }
    
    if not response_text:
        return result
    
    # Extract Job Title
    title_patterns = [
        r'Job Title:\s*(.+?)(?:\n|$)',
        r'Title:\s*(.+?)(?:\n|$)',
        r'\*\*Job title\*\*[:\s]*(.+?)(?:\n|$)',
    ]
    for pattern in title_patterns:
        match = re.search(pattern, response_text, re.MULTILINE | re.IGNORECASE)
        if match:
            result["job_title"] = match.group(1).strip()
            if result["job_title"].lower() not in ['not specified', 'none', 'n/a']:
                break
    
    # Extract Company Name
    company_patterns = [
        r'Company Name:\s*(.+?)(?:\n|$)',
        r'Company:\s*(.+?)(?:\n|$)',
        r'\*\*Company name\*\*[:\s]*(.+?)(?:\n|$)',
    ]
    for pattern in company_patterns:
        match = re.search(pattern, response_text, re.MULTILINE | re.IGNORECASE)
        if match:
            result["company"] = match.group(1).strip()
            if result["company"].lower() not in ['not specified', 'none', 'n/a']:
                break
    
    # Extract Description
    desc_patterns = [
        r'Job Description:\s*(.+?)(?:\n(?:Required|Qualifications|Responsibilities|Salary|Location|Job Type):|$)',
        r'Description:\s*(.+?)(?:\n(?:Required|Qualifications|Responsibilities|Salary|Location|Job Type):|$)',
    ]
    for pattern in desc_patterns:
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            desc = match.group(1).strip()
            # Remove truncation indicators
            if not desc.endswith('...'):
                result["description"] = desc
                break
    
    # Extract Skills
    skills_match = re.search(r'Required Skills?:\s*(.+?)(?:\n(?:Required|Qualifications|Responsibilities|Salary|Location|Job Type):|$)', response_text, re.DOTALL | re.IGNORECASE)
    if skills_match:
        skills_text = skills_match.group(1).strip()
        # Split by comma, semicolon, or newline
        result["skills_needed"] = [s.strip() for s in re.split(r'[,;\n]', skills_text) if s.strip()]
    
    # Extract Experience
    exp_match = re.search(r'Required Experience:\s*(.+?)(?:\n(?:Required|Qualifications|Responsibilities|Salary|Location|Job Type):|$)', response_text, re.DOTALL | re.IGNORECASE)
    if exp_match:
        result["experience_level"] = exp_match.group(1).strip()
    
    # Extract Location
    loc_match = re.search(r'Location:\s*(.+?)(?:\n(?:Required|Qualifications|Responsibilities|Salary|Job Type):|$)', response_text, re.MULTILINE | re.IGNORECASE)
    if loc_match:
        result["location"] = loc_match.group(1).strip()
    
    # Extract Salary
    salary_match = re.search(r'Salary:\s*(.+?)(?:\n(?:Required|Qualifications|Responsibilities|Location|Job Type):|$)', response_text, re.MULTILINE | re.IGNORECASE)
    if salary_match:
        result["salary"] = salary_match.group(1).strip()
    
    # Extract Job Type
    type_match = re.search(r'Job Type:\s*(.+?)(?:\n|$)', response_text, re.MULTILINE | re.IGNORECASE)
    if type_match:
        result["job_type"] = type_match.group(1).strip()
    
    return result


# Deterministic scorer kept for reference but not used by default
# Uncomment and use build_deterministic_scorer() if you want fully deterministic scoring
class _DeterministicScorer:
    """Wrapper class to mimic Agent interface but use deterministic scoring."""
    
    def __init__(self, model_name: str = None):
        """Initialize scorer (model_name kept for compatibility but not used)."""
        self.name = "Job Scorer (Deterministic)"
        self.model_name = model_name
    
    def run(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        Run scoring on input data.
        
        Args:
            input_data: JSON string or dict with candidate_profile and job_details
        
        Returns:
            JSON string with match results
        """
        # Parse input
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
            except:
                # Try to extract candidate_profile and job_details from prompt text
                # This handles the case where app.py sends a formatted prompt
                data = self._parse_prompt_input(input_data)
        else:
            data = input_data
        
        # Extract candidate_profile and job_details
        candidate_profile = data.get("candidate_profile", {})
        job_details = data.get("job_details", {}) or data.get("job", {})
        
        # Convert Pydantic models or objects to dicts
        if hasattr(candidate_profile, 'dict'):
            candidate_profile = candidate_profile.dict()
        elif hasattr(candidate_profile, '__dict__'):
            candidate_profile = candidate_profile.__dict__
        elif not isinstance(candidate_profile, dict):
            candidate_profile = {}
        
        if hasattr(job_details, 'dict'):
            job_details = job_details.dict()
        elif hasattr(job_details, '__dict__'):
            job_details = job_details.__dict__
        elif not isinstance(job_details, (dict, str)):
            job_details = {}
        
        # Calculate score
        result = calculate_job_match_score(candidate_profile, job_details)
        
        # Return as JSON string
        return json.dumps(result, indent=2)
    
    def _parse_prompt_input(self, prompt: str) -> Dict[str, Any]:
        """Parse prompt text to extract candidate_profile and job_details."""
        candidate_profile = {}
        job_details = {}
        
        # Extract Candidate Profile JSON
        if "Candidate Profile:" in prompt:
            profile_section = prompt.split("Candidate Profile:")[1]
            if "Job Details:" in profile_section:
                profile_section = profile_section.split("Job Details:")[0]
            
            # Try to find JSON object in the profile section
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', profile_section, re.DOTALL)
            if json_match:
                try:
                    candidate_profile = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        
        # Extract Job Details
        if "Job Details:" in prompt:
            job_section = prompt.split("Job Details:")[1]
            
            # Extract Title
            title_match = re.search(r'- Title:\s*(.+?)(?:\n|$)', job_section, re.MULTILINE)
            if title_match:
                job_details["job_title"] = title_match.group(1).strip()
            
            # Extract Company
            company_match = re.search(r'- Company:\s*(.+?)(?:\n|$)', job_section, re.MULTILINE)
            if company_match:
                job_details["company"] = company_match.group(1).strip()
            
            # Extract URL
            url_match = re.search(r'- URL:\s*(.+?)(?:\n|$)', job_section, re.MULTILINE)
            if url_match:
                job_details["url"] = url_match.group(1).strip()
            
            # Extract Description (everything after "- Description:" until next section or end)
            desc_match = re.search(r'- Description:\s*(.+?)(?:\n\n|\nCRITICAL|\nReturn|$)', job_section, re.DOTALL)
            if desc_match:
                job_details["description"] = desc_match.group(1).strip()
        
        return {
            "candidate_profile": candidate_profile,
            "job_details": job_details
        }


def build_deterministic_scorer(model_name: str = None):
    """
    Returns a fully deterministic scorer (no AI).
    Use this if you want 100% consistent scores with no variation.
    
    Note: This is NOT the default. Use build_scorer() for AI-based scoring.
    """
    return _DeterministicScorer(model_name)


def build_scorer(model_name: str) -> Agent:
    """
    Agent that evaluates job-candidate fit using AI reasoning.
    Uses strict prompting and validation to minimize hallucinations.
    """
    model = get_model_config(model_name, default_temperature=0)
    return Agent(
        name="Job Scorer",
        role="Evaluate candidate-job match accurately based on provided data only",
        model=model,
        instructions=[
            "You are a precise job matching evaluator. Analyze ONLY the information provided.",
            "",
            "INPUT FORMAT:",
            "You receive: candidate_profile (JSON) and job_details (text or JSON).",
            "",
            "SCORING METHODOLOGY:",
            "1. Skills Match (40%): Count how many job-required skills the candidate has",
            "   - Extract ALL skills/technologies mentioned in job description",
            "   - Match against candidate's skills list",
            "   - Include related skills (e.g., TensorFlow matches ML, Python matches data analysis)",
            "   - Count each matched skill in key_matches[]",
            "",
            "2. Experience Match (30%): Compare experience level",
            "   - Check if candidate meets minimum years required",
            "   - Consider domain relevance",
            "   - Add to requirements_met if experience requirement is satisfied",
            "",
            "3. Role Fit (20%): Match job responsibilities to candidate experience",
            "   - Look for keywords in candidate's experience_summary",
            "   - Consider education alignment",
            "   - Add to requirements_met if education requirement is satisfied",
            "",
            "4. Growth Potential (10%): Candidate's ability to grow in role",
            "",
            "REQUIREMENTS COUNTING:",
            "- total_requirements = ALL requirements mentioned in job (skills + experience + education + qualifications)",
            "- requirements_met = Number of requirements the candidate satisfies",
            "- Example: Job needs [Python, ML, 0-1 year exp, BE degree] = 4 total requirements",
            "  Candidate has [Python, TensorFlow, Pandas, 1.4 years, BE CSE] = 5 requirements met (Python✓, ML via TensorFlow✓, exp✓, degree✓, bonus: Pandas✓)",
            "",
            "CRITICAL RULES TO PREVENT HALLUCINATION:",
            "- Do NOT invent skills the candidate doesn't have",
            "- Do NOT assume information not in the data",
            "- Count ONLY skills explicitly listed in candidate_profile.skills[]",
            "- If job description mentions 'data annotation' and candidate has 'Python, ML, Computer Vision' - these ARE related matches",
            "- Be GENEROUS with related skills (ML/CV/Data Science experience counts for data annotation roles)",
            "- Count related skills: 'Data Annotation' job + candidate has 'TensorFlow, Pandas, OpenCV' = 3 requirements met",
            "",
            "OUTPUT FORMAT (MUST be valid JSON):",
            "Return your response as a JSON object with the following structure:",
            "{",
            '  "match_score": 0.65,',
            '  "key_matches": ["Python", "TensorFlow", "Pandas", "Computer Vision", "Data Analysis"],',
            '  "requirements_met": 8,',
            '  "total_requirements": 10,',
            '  "reasoning": "Candidate has strong ML and Python skills relevant to data annotation. Experience with TensorFlow, Keras, and computer vision directly applies to ML data curation tasks. Meets experience and education requirements."',
            "}",
            "CRITICAL: Your response must be valid JSON that can be parsed directly.",
            "",
            "REQUIREMENTS CALCULATION EXAMPLE:",
            "Job Description: 'Data Annotation for ML. Need Python, ML knowledge, 0-1 year exp, BE/BTech degree'",
            "Total Requirements: 4 (Python, ML, Experience, Degree)",
            "Candidate: Has Python, TensorFlow, Keras, Pandas, OpenCV, 1.4 years, BE CSE",
            "Requirements Met: 5+ (Python✓, ML via TensorFlow/Keras✓, Experience✓, Degree✓, Bonus: Pandas, OpenCV for data work)",
            "Key Matches: ['Python', 'TensorFlow', 'Keras', 'Pandas', 'OpenCV', 'Computer Vision']",
            "Match Score: 0.70-0.75 (strong match)",
            "",
            "SCORING GUIDELINES:",
            "- 0.0-0.3: Poor fit (missing most key skills)",
            "- 0.3-0.5: Weak fit (some transferable skills)",
            "- 0.5-0.7: Good fit (solid match with minor gaps)",
            "- 0.7-1.0: Excellent fit (strong alignment)",
            "",
            "IMPORTANT: Be realistic but not overly strict. Related skills count!",
            "",
            "REAL EXAMPLE - Data Annotator Role:",
            "Job: 'Data Annotation for AI/ML car platforms, 0-1 year, BE/MTech CSE'",
            "Candidate: Python, TensorFlow, Keras, PyTorch, OpenCV, Pandas, NumPy, Computer Vision, 1.42 years, BE CSE",
            "Analysis:",
            "- Total Requirements: ~5 (ML knowledge, Python, Experience, Degree, Data skills)",
            "- Requirements Met: 8+ (Python✓, TensorFlow✓, Keras✓, PyTorch✓, OpenCV✓, Pandas✓, NumPy✓, CV✓, Exp✓, Degree✓)",
            "- Key Matches: ['Python', 'TensorFlow', 'Keras', 'PyTorch', 'OpenCV', 'Computer Vision', 'Pandas', 'NumPy']",
            "- Match Score: 0.70-0.75 (STRONG MATCH)",
            "- Reasoning: 'Excellent ML/CV background for data annotation. Python, TensorFlow, and computer vision experience directly support ML data curation.'",
        ],
        show_tool_calls=True,
        markdown=False,
        response_format={"type": "json_object"} if "gpt-4" in model_name.lower() or ("o1" in model_name.lower() and "gpt-5" not in model_name.lower()) else None,
    )


def build_summarizer(model_name: str) -> Agent:
    """Agent that creates detailed job match summaries without hallucination."""
    model = get_model_config(model_name, default_temperature=0.3)
    return Agent(
        name="Summarizer",
        role="Generate accurate, unique summaries based only on provided data",
        model=model,
        instructions=[
            "You receive: candidate_profile, job_details, and match_score.",
            "Write a unique 150-200 word summary that:",
            "- States whether candidate is a good/poor fit (based on match_score)",
            "- Explains specific skills and experience that match",
            "- Highlights any experience gaps or concerns",
            "- Discusses growth opportunities if applicable",
            "- Mentions practical considerations (location, compensation, etc.)",
            "- Includes information about visa sponsorship or scholarship opportunities if mentioned in the job posting",
            "",
            "CRITICAL ANTI-HALLUCINATION RULES:",
            "- Use ONLY information from the provided data - do NOT invent details",
            "- Do NOT mention skills the candidate doesn't have",
            "- Do NOT assume company details not in the job description",
            "- Do NOT invent visa sponsorship info - only mention if explicitly stated",
            "- Reference ACTUAL skills from candidate_profile.skills[] list",
            "- Quote ACTUAL job requirements from job_details",
            "",
            "STRUCTURE:",
            "1. Fit assessment (1 sentence based on match_score)",
            "2. Why this is a good/weak match (specific skills that align)",
            "3. Gaps and concerns (missing skills or experience)",
            "4. Growth opportunities (if match_score >= 0.5)",
            "5. Practical considerations (location, training, etc. if mentioned)",
            "6. Visa sponsorship (ONLY if explicitly mentioned in job description)",
            "",
            "TONE GUIDELINES:",
            "- Each summary must be unique - never reuse text from other jobs",
            "- Be honest about fit - don't oversell poor matches",
            "- Reference the actual job title and company name from the data",
            "- If match_score < 0.5, explain why it's not a good fit",
            "- If match_score >= 0.5, explain why it's a strong match",
            "",
            "EXAMPLE (for Data Annotator with ML candidate):",
            "\"With a match score of 65%, the candidate is a good fit. Their Python, TensorFlow, and computer vision experience directly supports data annotation for ML models. However, they lack specific annotation tool experience (CVAT, Labelbox). Growth opportunity in ML data engineering. Location: Coimbatore. Training required in Aurangabad.\"",
        ],
        show_tool_calls=True,
        markdown=True,
    )


# Removed unused example function process_job_matching - not used in production code