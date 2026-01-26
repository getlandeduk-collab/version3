"""
Deterministic Scoring Engine

All scoring functions are deterministic - same inputs produce same outputs.
No AI/LLM is used in this module.
"""

import logging
from typing import List, Dict, Any, Set
from .config import (
    WEIGHTS, SKILLS_WEIGHTS, EXPERIENCE_PENALTY,
    DEGREE_HIERARCHY, EDUCATION_LEVEL_SCORES, EDUCATION_FIELD_MULTIPLIERS,
    SENIORITY_HIERARCHY, SENIORITY_SCORES, RELATED_FIELD_GROUPS
)

logger = logging.getLogger(__name__)


def calculate_skills_score(
    job_skills_required: List[str],
    job_skills_preferred: List[str],
    candidate_skills: List[str]
) -> float:
    """
    Calculate skills match score (0-100).
    
    Formula:
    - Required match: (matched_required / total_required) * 100
    - Preferred match: (matched_preferred / total_preferred) * 100
    - Final: (0.70 * required_score) + (0.30 * preferred_score)
    
    Args:
        job_skills_required: List of required skills
        job_skills_preferred: List of preferred skills
        candidate_skills: List of candidate's skills
    
    Returns:
        Score from 0-100
    """
    # Normalize to sets for intersection
    required_set = set(s.lower() for s in job_skills_required)
    preferred_set = set(s.lower() for s in job_skills_preferred)
    candidate_set = set(s.lower() for s in candidate_skills)
    
    # Calculate required skills match
    if required_set:
        required_match = required_set & candidate_set
        required_score = (len(required_match) / len(required_set)) * 100
        logger.debug(f"Required skills: {len(required_match)}/{len(required_set)} = {required_score:.2f}%")
    else:
        required_score = 100  # No requirements = perfect score
        logger.debug("No required skills specified, score = 100")
    
    # Calculate preferred skills match
    if preferred_set:
        preferred_match = preferred_set & candidate_set
        preferred_score = (len(preferred_match) / len(preferred_set)) * 100
        logger.debug(f"Preferred skills: {len(preferred_match)}/{len(preferred_set)} = {preferred_score:.2f}%")
    else:
        preferred_score = 100  # No preferences = perfect score
        logger.debug("No preferred skills specified, score = 100")
    
    # Weighted combination
    skills_score = (
        SKILLS_WEIGHTS["required"] * required_score +
        SKILLS_WEIGHTS["preferred"] * preferred_score
    )
    
    logger.info(f"Skills score: {skills_score:.2f}%")
    return skills_score


def calculate_experience_score(
    required_years: float,
    candidate_years: float
) -> float:
    """
    Calculate experience match score (0-100).
    
    Formula:
    - If candidate >= required: min(100, (candidate / required) * 100)
    - If candidate < required: (candidate / required) * 100 * 0.7 (penalty)
    
    Args:
        required_years: Minimum years required
        candidate_years: Candidate's years of experience
    
    Returns:
        Score from 0-100
    """
    if required_years == 0:
        # No experience required
        return 100
    
    if candidate_years >= required_years:
        # Meets or exceeds requirement (cap at 100)
        score = min(100, (candidate_years / required_years) * 100)
        logger.debug(f"Experience: {candidate_years} >= {required_years} years, score = {score:.2f}%")
    else:
        # Below requirement (apply penalty)
        score = (candidate_years / required_years) * 100 * EXPERIENCE_PENALTY
        logger.debug(f"Experience: {candidate_years} < {required_years} years, score = {score:.2f}% (with penalty)")
    
    logger.info(f"Experience score: {score:.2f}%")
    return score


def extract_keywords(text_list: List[str]) -> Set[str]:
    """Extract keywords from list of text strings."""
    keywords = set()
    for text in text_list:
        # Simple keyword extraction: split by spaces and punctuation
        words = text.lower().replace(',', ' ').replace('.', ' ').split()
        # Filter out common words (basic stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords.update(w for w in words if len(w) > 2 and w not in stop_words)
    return keywords


def calculate_responsibility_score(
    job_responsibilities: List[str],
    candidate_responsibilities: List[str]
) -> float:
    """
    Calculate responsibility match score (0-100) using TF-IDF cosine similarity.
    Falls back to Jaccard similarity if sklearn is not available.
    
    Args:
        job_responsibilities: List of job responsibilities
        candidate_responsibilities: List of candidate's past responsibilities
    
    Returns:
        Score from 0-100
    """
    if not job_responsibilities or not candidate_responsibilities:
        logger.warning("Empty responsibilities list, returning default score 50")
        return 50.0
    
    try:
        # Option A: TF-IDF + Cosine similarity (preferred)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        job_text = " ".join(job_responsibilities)
        resume_text = " ".join(candidate_responsibilities)
        
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([job_text, resume_text])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        score = similarity * 100
        
        logger.debug(f"Responsibility score (TF-IDF): {score:.2f}%")
        logger.info(f"Responsibility score: {score:.2f}%")
        return score
        
    except ImportError:
        # Option B: Jaccard similarity (fallback)
        logger.warning("sklearn not available, using Jaccard similarity")
        
        job_keywords = extract_keywords(job_responsibilities)
        resume_keywords = extract_keywords(candidate_responsibilities)
        
        intersection = job_keywords & resume_keywords
        union = job_keywords | resume_keywords
        
        if union:
            jaccard = len(intersection) / len(union)
            score = jaccard * 100
        else:
            score = 0
        
        logger.debug(f"Responsibility score (Jaccard): {score:.2f}%")
        logger.info(f"Responsibility score: {score:.2f}%")
        return score


def is_related_field(field1: str, field2: str) -> bool:
    """
    Check if two education fields are related.
    
    Args:
        field1: First field name
        field2: Second field name
    
    Returns:
        True if fields are in the same group
    """
    field1_lower = field1.lower()
    field2_lower = field2.lower()
    
    for group in RELATED_FIELD_GROUPS:
        if field1_lower in group and field2_lower in group:
            return True
    
    return False


def calculate_education_score(
    required_level: str,
    required_field: str,
    candidate_level: str,
    candidate_field: str
) -> float:
    """
    Calculate education match score (0-100).
    
    Formula:
    - Level score based on degree hierarchy
    - Field multiplier based on field match/relation
    - Final: level_score * field_multiplier
    
    Args:
        required_level: Required degree level
        required_field: Required field of study
        candidate_level: Candidate's degree level
        candidate_field: Candidate's field of study
    
    Returns:
        Score from 0-100
    """
    required_rank = DEGREE_HIERARCHY.get(required_level, 0)
    candidate_rank = DEGREE_HIERARCHY.get(candidate_level, 0)
    
    # Calculate level score
    if candidate_rank >= required_rank:
        level_score = EDUCATION_LEVEL_SCORES["exact_match"]
        logger.debug(f"Education level: {candidate_level} >= {required_level}, score = {level_score}")
    elif candidate_rank == required_rank - 1:
        level_score = EDUCATION_LEVEL_SCORES["one_below"]
        logger.debug(f"Education level: {candidate_level} one below {required_level}, score = {level_score}")
    elif candidate_rank == required_rank - 2:
        level_score = EDUCATION_LEVEL_SCORES["two_below"]
        logger.debug(f"Education level: {candidate_level} two below {required_level}, score = {level_score}")
    else:
        level_score = EDUCATION_LEVEL_SCORES["other"]
        logger.debug(f"Education level: {candidate_level} significantly below {required_level}, score = {level_score}")
    
    # Calculate field multiplier
    if required_field.lower() == "any" or candidate_field.lower() == required_field.lower():
        field_multiplier = EDUCATION_FIELD_MULTIPLIERS["exact_match"]
        logger.debug(f"Education field: exact match or 'any', multiplier = {field_multiplier}")
    elif is_related_field(required_field, candidate_field):
        field_multiplier = EDUCATION_FIELD_MULTIPLIERS["related_field"]
        logger.debug(f"Education field: related ({candidate_field} ~ {required_field}), multiplier = {field_multiplier}")
    else:
        field_multiplier = EDUCATION_FIELD_MULTIPLIERS["different_field"]
        logger.debug(f"Education field: different, multiplier = {field_multiplier}")
    
    final_score = level_score * field_multiplier
    logger.info(f"Education score: {final_score:.2f}%")
    return final_score


def calculate_seniority_score(
    required_level: str,
    candidate_level: str
) -> float:
    """
    Calculate seniority match score (0-100).
    
    Formula based on hierarchy difference:
    - 0 levels: 100
    - 1 level above: 90
    - 1 level below: 60
    - 2+ levels: 30
    
    Args:
        required_level: Required seniority level
        candidate_level: Candidate's seniority level
    
    Returns:
        Score from 0-100
    """
    required_rank = SENIORITY_HIERARCHY.get(required_level, 1)
    candidate_rank = SENIORITY_HIERARCHY.get(candidate_level, 1)
    
    difference = abs(required_rank - candidate_rank)
    
    if difference == 0:
        score = SENIORITY_SCORES[0]
        logger.debug(f"Seniority: exact match ({candidate_level}), score = {score}")
    elif difference == 1:
        if candidate_rank > required_rank:
            score = SENIORITY_SCORES[1]["above"]
            logger.debug(f"Seniority: one level above ({candidate_level} > {required_level}), score = {score}")
        else:
            score = SENIORITY_SCORES[1]["below"]
            logger.debug(f"Seniority: one level below ({candidate_level} < {required_level}), score = {score}")
    elif difference == 2:
        score = SENIORITY_SCORES[2]
        logger.debug(f"Seniority: two levels difference, score = {score}")
    else:
        score = SENIORITY_SCORES["other"]
        logger.debug(f"Seniority: {difference} levels difference, score = {score}")
    
    logger.info(f"Seniority score: {score:.2f}%")
    return score


def calculate_match_score(
    job_data: Dict[str, Any],
    resume_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate final match score using all components.
    
    Args:
        job_data: Extracted job data
        resume_data: Extracted resume data
    
    Returns:
        Dict with match_percentage and breakdown
    """
    logger.info("=" * 60)
    logger.info("Starting deterministic score calculation")
    logger.info("=" * 60)
    
    # Calculate component scores
    skills_score = calculate_skills_score(
        job_data.get("skills_required", []),
        job_data.get("skills_preferred", []),
        resume_data.get("candidate_skills", [])
    )
    
    experience_score = calculate_experience_score(
        job_data.get("years_experience", 0),
        resume_data.get("candidate_years", 0)
    )
    
    responsibility_score = calculate_responsibility_score(
        job_data.get("responsibilities", []),
        resume_data.get("candidate_responsibilities", [])
    )
    
    education_score = calculate_education_score(
        job_data.get("education_required", "None"),
        job_data.get("education_field", "Any"),
        resume_data.get("candidate_education_level", "None"),
        resume_data.get("candidate_education_field", "Any")
    )
    
    seniority_score = calculate_seniority_score(
        job_data.get("seniority_level", "Mid"),
        resume_data.get("candidate_seniority", "Mid")
    )
    
    # Apply weights (must sum to 100)
    final_score = (
        (WEIGHTS["skills"] * skills_score) +
        (WEIGHTS["experience"] * experience_score) +
        (WEIGHTS["responsibilities"] * responsibility_score) +
        (WEIGHTS["education"] * education_score) +
        (WEIGHTS["seniority"] * seniority_score)
    )
    
    logger.info("=" * 60)
    logger.info(f"FINAL MATCH SCORE: {final_score:.2f}%")
    logger.info("=" * 60)
    
    # Return final score and breakdown for explainability
    return {
        "match_percentage": round(final_score, 2),
        "breakdown": {
            "skills": round(skills_score, 2),
            "experience": round(experience_score, 2),
            "responsibilities": round(responsibility_score, 2),
            "education": round(education_score, 2),
            "seniority": round(seniority_score, 2)
        }
    }

