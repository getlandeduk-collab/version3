"""
Stage 2: Candidate Requirement Evaluator

Evaluates candidate profile against FROZEN requirements from Stage 1.
NEVER modifies, adds, or removes requirements - only evaluates status.
"""
import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from model_config import get_model_for_task

logger = logging.getLogger(__name__)


def evaluate_candidate_against_requirements(
    candidate_profile: Dict[str, Any],
    frozen_requirements: List[Dict[str, Any]],
    job_id: Optional[str] = None,
    job_title: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Stage 2: Evaluate candidate against FROZEN requirements.
    
    This function:
    - Takes frozen requirements from Stage 1
    - Evaluates candidate profile against EACH requirement
    - Assigns status: MET, PARTIALLY_MET, or NOT_MET
    - NEVER modifies the requirement list
    - NEVER adds new requirements
    
    Args:
        candidate_profile: Candidate profile dict with skills, experience, education, etc.
        frozen_requirements: Frozen requirements list from Stage 1
        job_id: Optional job identifier
        job_title: Optional job title for context
        openai_api_key: OpenAI API key (uses env var if not provided)
    
    Returns:
        Dict with structure:
        {
            "job_id": "<job_id>",
            "evaluation": [
                {
                    "req_id": "REQ_01",
                    "name": "<Requirement Name>",
                    "status": "MET | PARTIALLY_MET | NOT_MET",
                    "justification": "<Evidence-based explanation>"
                }
            ]
        }
    """
    if not frozen_requirements:
        logger.warning("No frozen requirements provided for evaluation")
        return {
            "job_id": job_id or "unknown",
            "evaluation": []
        }
    
    # FIX 2: Freeze requirements - deepcopy to prevent mutation
    import copy
    frozen_requirements = copy.deepcopy(frozen_requirements)
    initial_count = len(frozen_requirements)
    initial_req_ids = {req.get("req_id") for req in frozen_requirements}
    
    logger.info(f"Evaluating against {initial_count} frozen requirements (req_ids: {sorted(initial_req_ids)})")
    
    # Get OpenAI API key
    import os
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be provided or set as environment variable")
    
    client = OpenAI(api_key=api_key)
    
    # CRITICAL: Convert datetime objects to strings before JSON serialization
    def convert_datetime_to_str(obj):
        """Recursively convert datetime objects to ISO format strings for JSON serialization"""
        from datetime import datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_datetime_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_datetime_to_str(item) for item in obj]
        else:
            return obj
    
    # Convert any datetime objects in candidate_profile before serialization
    candidate_profile_serializable = convert_datetime_to_str(candidate_profile)
    
    # Build evaluation prompt with context engineering
    frozen_requirements_json = json.dumps(frozen_requirements, indent=2)
    candidate_profile_json = json.dumps(candidate_profile_serializable, indent=2)
    
    prompt = f"""Evaluate candidate against frozen requirements. Each requirement must be exactly one status: MET, PARTIALLY_MET, or NOT_MET.

CANDIDATE PROFILE:
{candidate_profile_json}

FROZEN REQUIREMENTS:
{frozen_requirements_json}

JOB TITLE: {job_title or "Not specified"}

EVALUATION CONTEXT:
- Check ALL sources: skills array, experience_summary, experience_entries (including descriptions), projects, education, certifications
- CRITICAL: Check experience_entries descriptions - these contain project work, technologies used, systems built
- CRITICAL: Check projects field - these contain standalone projects with technologies and descriptions
- Infer skills from role titles (e.g., "AI Engineer" → AI/ML skills), company domains, project descriptions
- Look for technologies mentioned in project descriptions (e.g., "RAG", "LLM", "vector DB", "Weaviate", "Neo4j" → RAG/LLM experience)
- For education: Check education field AND infer from professional roles (Engineer/Developer roles → likely has degree)
- "Pursuing" or "Expected" degree → PARTIALLY_MET (not completed)
- Related experience/technologies → PARTIALLY_MET (e.g., JavaScript when TypeScript required, cloud infrastructure when AWS required)
- Example: If requirement is "RAG pipelines" and candidate has "developed RAG system using vector DB" in experience description → MET or PARTIALLY_MET
- Example: If requirement is "LLMs" and candidate mentions "GPT-4o-mini" or "Large Language Models" in project description → MET or PARTIALLY_MET
- Only NOT_MET when truly no related evidence in ANY source

STATUS RULES:
- MET: Direct match in profile (exact skill, completed degree, sufficient experience)
- PARTIALLY_MET: Related evidence (pursuing degree, related skills, partial experience, inferred from roles)
- NOT_MET: No evidence found

Return JSON:
{{
  "job_id": "{job_id or 'unknown'}",
  "evaluation": [
    {{"req_id": "REQ_01", "name": "...", "status": "MET|PARTIALLY_MET|NOT_MET", "justification": "..."}}
  ]
}}

Evaluate ALL {len(frozen_requirements)} requirements. Return ONLY JSON."""
    
    # Use task-specific model for matching
    # Use gpt-4o for matching/evaluation (as per user requirement)
    model_name = "gpt-4o"
    
    logger.info(f"Evaluating candidate against {len(frozen_requirements)} frozen requirements for job_id={job_id} using {model_name}")
    
    # Call LLM with temperature=0 and seed=42 for deterministic output
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # MUST be 0 for deterministic output
        seed=42,  # Fixed seed for consistency
        response_format={"type": "json_object"}
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Parse and validate JSON
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from requirement evaluation: {e}")
        logger.error(f"Response text: {response_text[:500]}")
        raise ValueError(f"Invalid JSON response from requirement evaluation: {e}")
    
    # Validate structure
    if "evaluation" not in result:
        raise ValueError("Response missing 'evaluation' field")
    
    if not isinstance(result["evaluation"], list):
        raise ValueError("'evaluation' must be a list")
    
    # FIX 3: Hard assertion - requirements must not be mutated during evaluation
    current_count = len(frozen_requirements)
    current_req_ids = {req.get("req_id") for req in frozen_requirements}
    
    if current_count != initial_count:
        raise ValueError(
            f"CRITICAL: Requirement count changed from {initial_count} to {current_count} during evaluation. "
            f"Freezing violated - requirements list was mutated!"
        )
    
    if current_req_ids != initial_req_ids:
        missing = initial_req_ids - current_req_ids
        extra = current_req_ids - initial_req_ids
        raise ValueError(
            f"CRITICAL: Requirement IDs changed during evaluation. "
            f"Missing: {missing}, Extra: {extra}. Freezing violated!"
        )
    
    # CRITICAL: Sort evaluation results by req_id for deterministic ordering
    # This ensures same evaluation always appears in same order
    evaluation = result["evaluation"]
    evaluation = sorted(evaluation, key=lambda e: e.get("req_id", ""))
    result["evaluation"] = evaluation
    
    # CRITICAL: Normalize justification and name text for consistency
    for eval_item in evaluation:
        if "justification" in eval_item and isinstance(eval_item["justification"], str):
            eval_item["justification"] = ' '.join(eval_item["justification"].strip().split())
        if "name" in eval_item and isinstance(eval_item["name"], str):
            eval_item["name"] = ' '.join(eval_item["name"].strip().split())
    
    # Validate evaluation matches frozen requirements
    if len(result["evaluation"]) != len(frozen_requirements):
        raise ValueError(
            f"Evaluation count mismatch: {len(result['evaluation'])} evaluations "
            f"vs {len(frozen_requirements)} frozen requirements"
        )
    
    # Validate each evaluation
    frozen_req_ids = {req["req_id"] for req in frozen_requirements}
    eval_req_ids = {eval_item["req_id"] for eval_item in result["evaluation"]}
    
    if frozen_req_ids != eval_req_ids:
        missing = frozen_req_ids - eval_req_ids
        extra = eval_req_ids - frozen_req_ids
        raise ValueError(
            f"Requirement ID mismatch: missing {missing}, extra {extra}"
        )
    
    for eval_item in result["evaluation"]:
        if not isinstance(eval_item, dict):
            raise ValueError("Each evaluation item must be a dict")
        required_fields = ["req_id", "name", "status", "justification"]
        for field in required_fields:
            if field not in eval_item:
                raise ValueError(f"Evaluation item missing required field: {field}")
        if eval_item["status"] not in ["MET", "PARTIALLY_MET", "NOT_MET"]:
            raise ValueError(f"Invalid status: {eval_item['status']} (must be MET, PARTIALLY_MET, or NOT_MET)")
    
    logger.info(f"Successfully evaluated {len(result['evaluation'])} requirements for job_id={job_id}")
    
    return result


def calculate_deterministic_score(
    evaluation: List[Dict[str, Any]],
    frozen_requirements: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate deterministic match score based on frozen requirements evaluation.
    
    Scoring Rules:
    - MET = 1.0
    - PARTIALLY_MET = 0.6
    - NOT_MET = 0.0
    
    Only CORE requirements count toward score (preferred requirements are excluded).
    
    Args:
        evaluation: Evaluation results from Stage 2
        frozen_requirements: Frozen requirements from Stage 1
    
    Returns:
        Dict with match_score, requirements_met, total_requirements, etc.
    """
    # Create mapping of req_id to category
    req_category_map = {req["req_id"]: req["category"] for req in frozen_requirements}
    
    # Filter to only core requirements for scoring
    core_evaluations = [
        eval_item for eval_item in evaluation
        if req_category_map.get(eval_item["req_id"]) == "core"
    ]
    
    if not core_evaluations:
        # If no core requirements, use all requirements
        core_evaluations = evaluation
    
    # Calculate weighted scores
    total_requirements = len(core_evaluations)
    if total_requirements == 0:
        return {
            "match_score": 0.0,
            "requirements_met": 0,
            "requirements_partially_met": 0,
            "requirements_not_met": 0,
            "total_requirements": 0,
            "total_core_requirements": 0,
            "total_preferred_requirements": len([r for r in frozen_requirements if r["category"] == "preferred"])
        }
    
    status_weights = {
        "MET": 1.0,
        "PARTIALLY_MET": 0.6,  # User specified 0.6 in requirements
        "NOT_MET": 0.0
    }
    
    weighted_sum = sum(
        status_weights.get(eval_item["status"], 0.0)
        for eval_item in core_evaluations
    )
    
    match_score = weighted_sum / total_requirements
    
    # Count by status (for core requirements only - used for scoring)
    requirements_met = len([e for e in core_evaluations if e["status"] == "MET"])
    requirements_partially_met = len([e for e in core_evaluations if e["status"] == "PARTIALLY_MET"])
    requirements_not_met = len([e for e in core_evaluations if e["status"] == "NOT_MET"])
    
    # CRITICAL: Validate that ALL requirements were evaluated
    if len(evaluation) != len(frozen_requirements):
        raise ValueError(
            f"Evaluation count mismatch: {len(evaluation)} evaluations vs {len(frozen_requirements)} frozen requirements. "
            f"Every requirement must be evaluated exactly once."
        )
    
    # Validate that every requirement has exactly one status
    eval_req_ids = {e["req_id"] for e in evaluation}
    frozen_req_ids = {r["req_id"] for r in frozen_requirements}
    if eval_req_ids != frozen_req_ids:
        missing = frozen_req_ids - eval_req_ids
        extra = eval_req_ids - frozen_req_ids
        raise ValueError(
            f"Requirement ID mismatch: missing evaluations for {missing}, extra evaluations {extra}. "
            f"Every requirement must be evaluated exactly once."
        )
    
    # Build lists from ALL evaluations (core + preferred) for reporting
    all_met = [e for e in evaluation if e["status"] == "MET"]
    all_partially_met = [e for e in evaluation if e["status"] == "PARTIALLY_MET"]
    all_not_met = [e for e in evaluation if e["status"] == "NOT_MET"]
    
    # Validate that every requirement appears in exactly one list
    total_evaluated = len(all_met) + len(all_partially_met) + len(all_not_met)
    if total_evaluated != len(evaluation):
        raise ValueError(
            f"Status classification error: {total_evaluated} requirements classified but {len(evaluation)} evaluated. "
            f"Every requirement must have exactly one status (MET, PARTIALLY_MET, or NOT_MET)."
        )
    
    return {
        "match_score": round(match_score, 3),
        "requirements_met": requirements_met,  # Core only (for scoring)
        "requirements_partially_met": requirements_partially_met,  # Core only (for scoring)
        "requirements_not_met": requirements_not_met,  # Core only (for scoring)
        "total_requirements": total_requirements,  # Core only (for scoring)
        "total_core_requirements": len([r for r in frozen_requirements if r["category"] == "core"]),
        "total_preferred_requirements": len([r for r in frozen_requirements if r["category"] == "preferred"]),
        "total_all_requirements": len(frozen_requirements),  # All requirements (core + preferred)
        "requirements_satisfied": sorted([
            f"{e['name']} (MET - {e['justification']})" for e in all_met
        ]),
        "requirements_partially_met": sorted([
            f"{e['name']} (PARTIALLY MET - {e['justification']})" for e in all_partially_met
        ]),
        "requirements_missing": sorted([
            f"{e['name']} (NOT MET - {e['justification']})" for e in all_not_met
        ]),
        # Include full evaluation for validation
        "evaluation": evaluation
    }
