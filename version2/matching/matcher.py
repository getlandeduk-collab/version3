"""
Main Matcher Module

Orchestrates the complete matching process:
1. Extract structured data using LLM
2. Calculate deterministic match score
3. Return complete results with breakdown
"""

import logging
from typing import Dict, Any
from .llm_extractor import extract_data_with_llm, validate_extracted_data
from .scoring_engine import calculate_match_score

logger = logging.getLogger(__name__)


def match_job_resume(
    job_description: str,
    resume: str,
    model_name: str = None,
    return_extracted_data: bool = True
) -> Dict[str, Any]:
    """
    Calculate match percentage between a job and resume.
    
    This is the main entry point for the matching system. It:
    1. Uses LLM (LangGraph + GPT-4) to extract structured data
    2. Calculates match score using deterministic formulas
    3. Returns match percentage with detailed breakdown
    
    Args:
        job_description: Full job description text
        resume: Full resume text
        model_name: Optional model name (defaults to config)
        return_extracted_data: Whether to include extracted data in response
    
    Returns:
        {
            "match_percentage": float (0-100),
            "breakdown": {
                "skills": float,
                "experience": float,
                "responsibilities": float,
                "education": float,
                "seniority": float
            },
            "extracted_data": {  # Optional, if return_extracted_data=True
                "job": {...},
                "resume": {...}
            }
        }
    
    Raises:
        ValueError: If extraction or scoring fails
    
    Example:
        >>> result = match_job_resume(job_desc, resume_text)
        >>> print(f"Match: {result['match_percentage']}%")
        >>> print(f"Skills: {result['breakdown']['skills']}%")
    """
    logger.info("=" * 80)
    logger.info("STARTING JOB-RESUME MATCHING")
    logger.info("=" * 80)
    
    try:
        # Step 1: Extract structured data using LLM
        logger.info("Step 1: Extracting structured data with LLM...")
        extracted_data = extract_data_with_llm(job_description, resume, model_name)
        
        # Validate extracted data
        validate_extracted_data(extracted_data)
        logger.info("✓ Data extraction successful")
        
        # Log extracted data summary
        job_data = extracted_data["job"]
        resume_data = extracted_data["resume"]
        logger.info(f"Job: {len(job_data.get('skills_required', []))} required skills, "
                   f"{job_data.get('years_experience', 0)} years exp, "
                   f"{job_data.get('seniority_level', 'N/A')} level")
        logger.info(f"Resume: {len(resume_data.get('candidate_skills', []))} skills, "
                   f"{resume_data.get('candidate_years', 0)} years exp, "
                   f"{resume_data.get('candidate_seniority', 'N/A')} level")
        
        # Step 2: Calculate deterministic score
        logger.info("Step 2: Calculating deterministic match score...")
        score_result = calculate_match_score(job_data, resume_data)
        logger.info("✓ Score calculation complete")
        
        # Step 3: Return complete result
        result = {
            **score_result
        }
        
        if return_extracted_data:
            result["extracted_data"] = extracted_data
        
        logger.info("=" * 80)
        logger.info(f"MATCHING COMPLETE - Score: {result['match_percentage']}%")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Matching failed: {e}", exc_info=True)
        raise ValueError(f"Job-resume matching failed: {e}")


def match_multiple_jobs(
    job_descriptions: list[str],
    resume: str,
    model_name: str = None,
    return_extracted_data: bool = False
) -> list[Dict[str, Any]]:
    """
    Match a resume against multiple job descriptions.
    
    Args:
        job_descriptions: List of job description texts
        resume: Resume text
        model_name: Optional model name
        return_extracted_data: Whether to include extracted data
    
    Returns:
        List of match results, sorted by match_percentage (highest first)
    
    Example:
        >>> jobs = [job1_desc, job2_desc, job3_desc]
        >>> results = match_multiple_jobs(jobs, resume_text)
        >>> for i, result in enumerate(results, 1):
        >>>     print(f"#{i}: {result['match_percentage']}%")
    """
    logger.info(f"Matching resume against {len(job_descriptions)} jobs")
    
    results = []
    for i, job_desc in enumerate(job_descriptions, 1):
        try:
            logger.info(f"\nProcessing job {i}/{len(job_descriptions)}")
            result = match_job_resume(
                job_desc,
                resume,
                model_name,
                return_extracted_data
            )
            result["job_index"] = i - 1  # 0-based index
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to match job {i}: {e}")
            results.append({
                "job_index": i - 1,
                "match_percentage": 0.0,
                "error": str(e),
                "breakdown": {
                    "skills": 0.0,
                    "experience": 0.0,
                    "responsibilities": 0.0,
                    "education": 0.0,
                    "seniority": 0.0
                }
            })
    
    # Sort by match percentage (highest first)
    results.sort(key=lambda x: x.get("match_percentage", 0), reverse=True)
    
    logger.info(f"\nCompleted matching {len(results)} jobs")
    logger.info(f"Top match: {results[0]['match_percentage']}%")
    
    return results

