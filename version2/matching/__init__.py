"""
Deterministic Job-Resume Matching System

This package provides a two-step matching system:
1. LLM extraction of structured data (LangGraph + GPT-4)
2. Deterministic mathematical scoring

Usage:
    from matching import match_job_resume
    
    result = match_job_resume(job_description, resume_text)
    print(f"Match: {result['match_percentage']}%")
"""

from .matcher import match_job_resume
from .config import WEIGHTS

__all__ = ["match_job_resume", "WEIGHTS"]
__version__ = "1.0.0"

