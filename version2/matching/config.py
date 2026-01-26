"""
Configuration for the deterministic job-resume matching system.
Adjust weights and parameters here.
"""

# Component weights (must sum to 1.0)
WEIGHTS = {
    "skills": 0.35,
    "experience": 0.25,
    "responsibilities": 0.20,
    "education": 0.10,
    "seniority": 0.10,
}

# Skills scoring weights
SKILLS_WEIGHTS = {
    "required": 0.70,
    "preferred": 0.30,
}

# Experience scoring parameters
EXPERIENCE_PENALTY = 0.7  # Penalty multiplier when below required years

# Education degree hierarchy
DEGREE_HIERARCHY = {
    "None": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "PhD": 4,
}

# Education level scoring
EDUCATION_LEVEL_SCORES = {
    "exact_match": 100,
    "one_below": 70,
    "two_below": 40,
    "other": 0,
}

# Education field multipliers
EDUCATION_FIELD_MULTIPLIERS = {
    "exact_match": 1.0,
    "related_field": 0.85,
    "different_field": 0.6,
}

# Related field groups for education
RELATED_FIELD_GROUPS = [
    ["computer science", "software engineering", "information technology", "computer engineering", "data science"],
    ["business administration", "management", "finance", "economics", "accounting"],
    ["mechanical engineering", "civil engineering", "electrical engineering", "industrial engineering"],
    ["biology", "chemistry", "physics", "biochemistry", "biotechnology"],
    ["mathematics", "statistics", "applied mathematics"],
]

# Seniority hierarchy
SENIORITY_HIERARCHY = {
    "Entry": 0,
    "Mid": 1,
    "Senior": 2,
    "Lead": 3,
    "Executive": 4,
}

# Seniority scoring based on difference
SENIORITY_SCORES = {
    0: 100,  # Exact match
    1: {"above": 90, "below": 60},  # One level difference
    2: 30,  # Two levels difference
    "other": 30,  # More than two levels
}

# LLM configuration
LLM_CONFIG = {
    "temperature": 0,  # For maximum consistency
    "model": "gpt-4o",  # Default model
    "max_retries": 3,
}

# Skill normalization mappings
SKILL_NORMALIZATIONS = {
    "react.js": "React",
    "reactjs": "React",
    "react js": "React",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "sql server": "SQL",
    "mysql": "SQL",
    "postgresql": "SQL",
    "postgres": "SQL",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "python3": "Python",
    "c++": "C++",
    "c#": "C#",
    "aws": "AWS",
    "amazon web services": "AWS",
    "azure": "Azure",
    "microsoft azure": "Azure",
    "google cloud": "GCP",
    "gcp": "GCP",
}

# Validation thresholds
VALIDATION = {
    "min_skills": 1,  # Minimum skills required for valid extraction
    "min_responsibilities": 1,  # Minimum responsibilities for valid extraction
}

