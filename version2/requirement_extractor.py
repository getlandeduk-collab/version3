"""
Stage 1: Job Requirement Extractor

Extracts canonical, normalized requirements STRICTLY from Job Description (JD) text.
Requirements are frozen and cached - same JD always produces same requirements.
"""
import json
import hashlib
import logging
import time
import os
from typing import Dict, List, Any, Optional
from collections import OrderedDict
from openai import OpenAI
from model_config import get_model_for_task

logger = logging.getLogger(__name__)

# Cache configuration
MAX_CACHE_SIZE = int(os.getenv("REQUIREMENTS_CACHE_MAX_SIZE", "1000"))  # Max number of cached requirements
CACHE_TTL_SECONDS = int(os.getenv("REQUIREMENTS_CACHE_TTL_SECONDS", "86400"))  # 24 hours default

# LRU cache for frozen requirements
# Structure: {job_hash: {"data": frozen_requirements, "timestamp": unix_time}}
_FROZEN_REQUIREMENTS_CACHE: OrderedDict[str, Dict[str, Any]] = OrderedDict()


def extract_stable_job_id_from_url(job_url: str) -> Optional[str]:
    """
    Extract stable job ID from job URL.
    
    CRITICAL: LinkedIn URLs contain dynamic query parameters that change.
    We must extract ONLY the stable job ID (currentJobId).
    
    Examples:
    - LinkedIn: "https://linkedin.com/jobs/search/?currentJobId=4364682864&..." -> "linkedin:4364682864"
    - Other portals: Extract domain-specific stable ID if available
    
    Args:
        job_url: Full job URL
    
    Returns:
        Stable job ID string (e.g., "linkedin:4364682864") or None if cannot extract
    """
    if not job_url:
        return None
    
    job_url_lower = job_url.lower()
    
    # LinkedIn: Extract currentJobId parameter
    if 'linkedin.com' in job_url_lower:
        import re
        match = re.search(r'currentJobId=(\d+)', job_url)
        if match:
            job_id = match.group(1)
            logger.debug(f"Extracted LinkedIn job ID: {job_id} from URL")
            return f"linkedin:{job_id}"
        else:
            logger.warning(f"Could not extract LinkedIn job ID from URL: {job_url[:100]}...")
            return None
    
    # Other portals: Could add extraction logic here
    # For now, return None to fall back to other methods
    return None


def get_job_hash(job_description: str, job_id: Optional[str] = None, job_url: Optional[str] = None) -> str:
    """
    Generate deterministic hash for job description.
    
    CRITICAL: Hash must be stable - same job must always produce same hash.
    - Normalize job description (strip, normalize whitespace)
    - Use stable identifier (extracted job ID from URL > job_id > description only)
    - Do NOT include unstable fields (idx, timestamps, dynamic URL params, etc.)
    - NEVER hash full LinkedIn URL (contains dynamic query parameters)
    
    Args:
        job_description: Raw job description text
        job_id: Optional job ID (use only if stable, not position-based)
        job_url: Optional job URL (will extract stable ID from it)
    
    Returns:
        Hash string for caching
    """
    # Normalize job description: strip and normalize whitespace
    normalized_description = ' '.join(job_description.strip().split())
    
    # FIX 1: Extract stable job ID from URL (CRITICAL for LinkedIn)
    stable_id = None
    if job_url:
        extracted_id = extract_stable_job_id_from_url(job_url)
        if extracted_id:
            stable_id = extracted_id
            logger.debug(f"Using extracted stable job ID from URL: {extracted_id}")
        else:
            logger.warning(f"Could not extract stable ID from URL, will use fallback: {job_url[:100]}...")
    
    # Fallback to job_id if no stable ID extracted from URL
    if not stable_id:
        if job_id and not job_id.startswith('job_') and '_' not in job_id:
            # Use job_id only if it's stable (not position-based like "job_1_...")
            stable_id = job_id
        else:
            # Last resort: use description only (less ideal but stable)
            stable_id = None
    
    if stable_id:
        content = f"{stable_id}|{normalized_description}"
    else:
        # Hash description only (for backward compatibility)
        content = normalized_description
    
    hash_result = hashlib.sha256(content.encode('utf-8')).hexdigest()
    logger.debug(f"Job hash generated: {hash_result[:16]}... (using {'extracted ID from URL' if job_url and stable_id else ('job_id' if job_id and not job_id.startswith('job_') else 'description only')})")
    return hash_result


def extract_frozen_requirements(
    job_description: str,
    job_id: Optional[str] = None,
    job_title: Optional[str] = None,
    job_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Stage 1: Extract frozen requirements from Job Description ONLY.
    
    This function:
    - Extracts requirements STRICTLY from JD text
    - Groups related items into conceptual requirements
    - Distinguishes between Core and Preferred requirements
    - Caches results for consistency
    - NEVER evaluates the candidate
    
    Args:
        job_description: Raw job description text
        job_id: Optional job identifier
        job_title: Optional job title for context
        openai_api_key: OpenAI API key (uses env var if not provided)
        use_cache: Whether to use cached requirements if available
    
    Returns:
        Dict with structure:
        {
            "job_id": "<job_id>",
            "frozen_requirements": [
                {
                    "req_id": "REQ_01",
                    "name": "<Requirement Name>",
                    "category": "core | preferred",
                    "jd_evidence": "<Exact JD sentence(s)>"
                }
            ]
        }
    """
    # Normalize job description before hashing (critical for stability)
    normalized_description = ' '.join(job_description.strip().split())
    
    # Check cache first - use stable hash
    job_hash = get_job_hash(normalized_description, job_id, job_url)
    logger.info(f"Checking cache for job requirements: hash={job_hash[:16]}..., use_cache={use_cache}")
    if use_cache and job_hash in _FROZEN_REQUIREMENTS_CACHE:
        cache_entry = _FROZEN_REQUIREMENTS_CACHE[job_hash]
        current_time = time.time()
        
        # Check if cache entry has expired
        if "timestamp" in cache_entry:
            age = current_time - cache_entry["timestamp"]
            if age > CACHE_TTL_SECONDS:
                # Expired - remove from cache
                logger.info(f"Cache entry expired for job_id={job_id}, hash={job_hash[:16]}... (age: {age:.0f}s)")
                del _FROZEN_REQUIREMENTS_CACHE[job_hash]
            else:
                # Valid cache entry - move to end (LRU)
                _FROZEN_REQUIREMENTS_CACHE.move_to_end(job_hash)
                # CRITICAL: Return deepcopy to prevent cache corruption
                import copy
                cached_data = copy.deepcopy(cache_entry["data"])
                req_count = len(cached_data.get('frozen_requirements', []))
                logger.info(f"✅ Using cached frozen requirements for job_id={job_id}, hash={job_hash[:16]}... (age: {age:.0f}s, {req_count} requirements)")
                logger.info(f"CRITICAL: Same job (hash={job_hash[:16]}...) will always return these {req_count} cached requirements")
                # Log requirement IDs for validation
                if req_count > 0:
                    req_ids = [req.get('req_id', '') for req in cached_data.get('frozen_requirements', [])]
                    logger.info(f"Cached requirement IDs: {req_ids}")
                return cached_data
        else:
            # Legacy format without timestamp - return as-is but update format
            # CRITICAL: Return deepcopy to prevent cache corruption
            import copy
            legacy_data = cache_entry if isinstance(cache_entry, dict) and "frozen_requirements" in cache_entry else cache_entry
            _FROZEN_REQUIREMENTS_CACHE.move_to_end(job_hash)
            logger.info(f"✅ Using cached frozen requirements (legacy format) for job_id={job_id}, hash={job_hash[:16]}...")
            return copy.deepcopy(legacy_data) if isinstance(legacy_data, dict) else legacy_data
    
    # Get OpenAI API key
    import os
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be provided or set as environment variable")
    
    client = OpenAI(api_key=api_key)
    
    # Build extraction prompt
    prompt = f"""You are a Job Requirement Extractor. Your ONLY task is to extract ALL requirements STRICTLY from the Job Description text below.

CRITICAL RULES:
1. Extract requirements ONLY if they are explicitly stated or clearly implied in the JD
2. Extract ALL requirements - do not skip any mentioned in the JD
   - If the JD mentions 10 technologies, extract all 10 (grouped appropriately)
   - If the JD mentions education, experience, and skills, extract all of them
   - Be comprehensive - missing requirements is worse than extracting too many
3. Group related items into conceptual requirements:
   - Example: "Python, Java, C#" → "Programming Languages (Python, Java, C#)"
   - Example: "AWS, Azure, GCP" → "Cloud Platforms (AWS, Azure, GCP)"
   - Example: "TensorFlow, PyTorch, Keras" → "Deep Learning Frameworks (TensorFlow, PyTorch, Keras)"
4. Do NOT split cloud providers, languages, or frameworks into separate requirements
5. Distinguish between:
   - Core Requirements (explicitly required, "must have", "required", "basic qualifications", "essential")
   - Preferred Requirements ("nice to have", "preferred", "bonus", "additional qualifications", "desirable")
6. Do NOT invent requirements - only extract what is in the JD
7. Do NOT evaluate any candidate - this is extraction ONLY
8. Extract from BOTH job title AND job description
9. Extract ALL types of requirements including:
   - Education (degree level, field of study, certifications)
   - Experience (years, type, domain-specific experience)
   - Technical Skills (programming languages, frameworks, tools, platforms)
   - Soft Skills (if explicitly mentioned)
   - Domain Knowledge (industry-specific knowledge)
   - Other qualifications (licenses, clearances, etc.)

JOB TITLE: {job_title or "Not specified"}

JOB DESCRIPTION:
{job_description}

EXTRACTION INSTRUCTIONS:
1. Read the ENTIRE job description systematically from start to finish
2. Extract requirements in this EXACT order (extract ALL you find - do not skip any):
   a) **Education Requirements** (degree level, field of study, certifications)
      - Look for: "Bachelor's", "Master's", "PhD", "degree in", "education", "qualifications"
      - **CRITICAL**: Extract education from ANY section - "What You Bring", "Preferred", "Requirements", "Qualifications", etc.
      - **CRITICAL**: Even if under "Preferred:", still extract it as a requirement (mark as "preferred" category)
      - Example: "Bachelor's degree in Computer Science" → Extract as education requirement
      - Example: "Preferred: Bachelor's or Master's degree" → Extract as "Education: Bachelor's or Master's degree" (category: preferred)
   
   b) **Experience Requirements** (CRITICAL - MUST EXTRACT IF MENTIONED)
      - Look for: "years of experience", "X+ years", "experience in", "proven experience", "background in"
      - **CRITICAL**: Extract experience from ANY section - "What You Bring", "Preferred", "Requirements", etc.
      - **CRITICAL**: Even if under "Preferred:", still extract it as a requirement (mark as "preferred" category)
      - Extract: Number of years, type of experience (e.g., "AI/ML experience", "software development", "industry experience")
      - Example: "3+ years of AI/ML experience" → Extract as "Experience: 3+ years in AI/ML"
      - Example: "Preferred: 1+ years of experience in AI engineering" → Extract as "Experience: 1+ years in AI engineering" (category: preferred)
      - Example: "Proven experience with machine learning" → Extract as "Experience: Machine learning"
      - If no specific years mentioned but experience type is mentioned, still extract it
      - **DO NOT SKIP experience requirements - they are critical**
   
   c) **Technical Skills REQUIRED** (group related technologies together)
      - Look for: "required", "must have", "essential", "proficiency in", "experience with", "expertise in", "deep understanding of", "knowledge of"
      - **CRITICAL**: Items listed under "What You Bring" (main bullet list) are typically CORE requirements - extract EVERY bullet point
      - **CRITICAL**: Items listed under "Requirements" or "Qualifications" are typically CORE
      - **CRITICAL**: If "What You Bring" has 7 bullet points, extract ALL 7 as separate requirements
      - Group related items: languages, frameworks, tools, platforms
      - Example: "Expertise in cloud platforms (AWS, Azure, GCP)" → Extract as "Cloud Platforms (AWS, Azure, GCP)" - core
      - Example: "Proficiency in programming languages (Python, Java, C#)" → Extract as "Programming Languages (Python, Java, C#)" - core
      - Example: "Deep understanding of large language models, machine learning frameworks, and AI architectures" → Extract as "AI/ML Knowledge (LLMs, ML frameworks, AI architectures)" - core
      - Example: "Experience with DevOps practices and deploying scalable AI services" → Extract as "DevOps and AI Deployment" - core
      - Example: "Knowledge of enterprise security, compliance, and data governance" → Extract as "Enterprise Security and Compliance" - core
   
   d) **Technical Skills PREFERRED** (group related technologies together)
      - Look for: "preferred", "nice to have", "bonus", "desirable"
      - **CRITICAL**: Items explicitly marked "Preferred:" or in a "Preferred:" subsection are preferred
      - **CRITICAL**: Items listed after "Preferred:" label are preferred
      - **CRITICAL**: If "Preferred:" section has 3 bullet points, extract ALL 3 as separate requirements
      - Group related items: languages, frameworks, tools, platforms
   
   e) **Other Requirements** (soft skills, domain knowledge, etc. if explicitly mentioned)
      - Look for: "communication", "leadership", "teamwork", "problem-solving", etc.

3. For each requirement:
   - Assign a unique req_id (REQ_01, REQ_02, REQ_03, etc.)
   - Provide a clear, normalized name that groups related items
   - Mark as "core" (required) or "preferred" (nice to have)
   - Include exact JD sentence(s) as evidence

4. **CRITICAL**: Be thorough - extract every requirement mentioned in the JD
   - If JD mentions experience, you MUST extract it
   - If JD mentions education, you MUST extract it
   - If JD mentions skills, you MUST extract them
   - Missing a requirement type is a critical error

5. If the job title suggests technical skills (e.g., "AI Engineer" → extract AI/ML requirements, "Data Scientist" → extract data science requirements), extract those from the title as well

6. Pay special attention to:
   - Job title implications (e.g., "AI Engineer" implies AI/ML skills are core)
   - **Section headers** - these are CRITICAL sources of requirements:
     * "What You Bring" → Usually contains CORE requirements (extract everything listed, including education/experience if mentioned)
     * "Requirements" → Core requirements
     * "Qualifications" → Core requirements
     * "Skills Needed" → Technical skills requirements
     * "Must Have" → Core requirements
     * "Nice to Have" → Preferred requirements
     * "Preferred" → Preferred requirements (but STILL extract education/experience from here - mark as "preferred" category)
     * "Experience" → Experience requirements
     * "Education" → Education requirements
   - **IMPORTANT**: Education and Experience can appear in ANY section - always extract them regardless of section header
   - **IMPORTANT**: If education/experience appears under "Preferred:" label, extract it but mark category as "preferred"
   - **IMPORTANT**: Do NOT skip education/experience just because they're not in a "Requirements" section
   - **Bullet points and lists** - these almost always contain requirements
   - **Items listed under "What You Bring"** are typically CORE requirements unless marked as "Preferred"
   - Any mention of technologies, tools, frameworks, or platforms
   - **Experience mentions** (years, type, domain) - these are requirements too!
   - **Format example**: If you see "What You Bring" followed by bullet points, extract ALL items as requirements

OUTPUT FORMAT (MUST BE VALID JSON):
{{
  "job_id": "{job_id or 'unknown'}",
  "frozen_requirements": [
    {{
      "req_id": "REQ_01",
      "name": "Education: Bachelor's degree in Computer Science",
      "category": "core",
      "jd_evidence": "Bachelor's degree in Computer Science or related field required"
    }},
    {{
      "req_id": "REQ_02",
      "name": "Experience: 3+ years of software development",
      "category": "core",
      "jd_evidence": "Minimum 3 years of professional software development experience"
    }},
    {{
      "req_id": "REQ_03",
      "name": "Programming Languages (Python, Java, C++)",
      "category": "core",
      "jd_evidence": "Proficiency in Python, Java, or C++ required"
    }},
    {{
      "req_id": "REQ_04",
      "name": "Cloud Platforms (AWS, Azure)",
      "category": "preferred",
      "jd_evidence": "Experience with AWS or Azure preferred"
    }}
  ]
}}

IMPORTANT REMINDERS:
- Extract ALL requirements mentioned in the JD - be comprehensive
- Do not skip any requirements - missing a requirement is a critical error
- **EXPERIENCE REQUIREMENTS ARE CRITICAL** - if JD mentions experience (years, type, domain), you MUST extract it
- **EDUCATION REQUIREMENTS ARE CRITICAL** - if JD mentions education (degree, field), you MUST extract it
- Group related technologies together (do not split them into separate requirements)
- If the job is "AI Engineer", extract AI/ML requirements from both title and description
- If multiple programming languages are mentioned, group them as "Programming Languages (Python, Java, C++)"
- If multiple ML frameworks are mentioned, group them as "Machine Learning Frameworks (TensorFlow, PyTorch, Keras)"
- If multiple cloud platforms are mentioned, group them as "Cloud Platforms (AWS, Azure, GCP)"
- Be thorough - a typical job description should yield 5-15 requirements, not just 1-2
- **CRITICAL**: If "What You Bring" section has multiple bullet points, extract EACH bullet point as a separate requirement
- **CRITICAL**: Count the bullet points in "What You Bring" - if there are 7 bullets, you MUST extract at least 7 requirements from that section alone
- **CRITICAL**: Do NOT combine multiple distinct requirements into one - each distinct skill/knowledge area should be a separate requirement
- **CRITICAL**: Do NOT extract a single generic requirement like "Job requirements from description" - you MUST break it down into specific requirements
- **CRITICAL**: If the job title is "AI Engineer", extract specific AI/ML requirements (e.g., "Technical Skills: Machine Learning", "Technical Skills: Deep Learning", "Experience: AI/ML projects")
- **CRITICAL**: If the job mentions "internship", extract education requirements (e.g., "Education: Currently pursuing degree" or "Education: Bachelor's/Master's degree")
- **CRITICAL**: Example: If "What You Bring" lists:
  * "Expertise in cloud platforms (AWS, Azure, GCP)" → Extract as REQ_01
  * "Deep understanding of large language models, machine learning frameworks, and AI architectures" → Extract as REQ_02
  * "Proficiency in programming languages (Python, Java, C#)" → Extract as REQ_03
  * "Experience with DevOps practices" → Extract as REQ_04
  * etc. - EACH bullet point is a separate requirement
- **Common requirement types to look for:**
  * Education (degree, field, certifications)
  * Experience (years, type, domain)
  * Technical skills (languages, frameworks, tools, platforms)
  * Soft skills (if explicitly mentioned)
  * Domain knowledge (industry-specific)
- Return ONLY valid JSON, no markdown, no explanations."""
    
    # Use task-specific model for job extraction
    extraction_model = get_model_for_task("job_extraction", json_mode=True)
    
    logger.info(f"Extracting frozen requirements for job_id={job_id} using model={extraction_model.model_name}")
    
    # Call LLM with temperature=0 and seed=42 for deterministic output
    response = client.chat.completions.create(
        model=extraction_model.model_name,
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
        logger.error(f"Failed to parse JSON from requirement extraction: {e}")
        logger.error(f"Response text: {response_text[:500]}")
        raise ValueError(f"Invalid JSON response from requirement extraction: {e}")
    
    # Validate structure
    if "frozen_requirements" not in result:
        raise ValueError("Response missing 'frozen_requirements' field")
    
    if not isinstance(result["frozen_requirements"], list):
        raise ValueError("'frozen_requirements' must be a list")
    
    # Validate each requirement
    for req in result["frozen_requirements"]:
        if not isinstance(req, dict):
            raise ValueError("Each requirement must be a dict")
        required_fields = ["req_id", "name", "category", "jd_evidence"]
        for field in required_fields:
            if field not in req:
                raise ValueError(f"Requirement missing required field: {field}")
        if req["category"] not in ["core", "preferred"]:
            raise ValueError(f"Invalid category: {req['category']} (must be 'core' or 'preferred')")
    
    # CRITICAL: Validate requirements are not overly generic
    frozen_requirements = result["frozen_requirements"]
    
    # Reject overly generic requirements that don't provide specific information
    generic_patterns = [
        "job requirements from description",
        "requirements from job description",
        "job requirements",
        "requirements",
        "qualifications",
        "skills required",
        "must have",
        "should have"
    ]
    
    valid_requirements = []
    for req in frozen_requirements:
        req_name_lower = req.get("name", "").lower().strip()
        # Check if requirement name is too generic
        is_generic = any(pattern in req_name_lower for pattern in generic_patterns)
        if is_generic and len(req_name_lower.split()) <= 5:
            # Too generic - log warning but keep it for now (better than nothing)
            logger.warning(f"Generic requirement detected: '{req.get('name')}' - but keeping it")
        
        # Ensure requirement name is specific enough (at least 10 characters, not just "Job requirements")
        if len(req_name_lower) < 10 and req_name_lower in generic_patterns:
            logger.error(f"CRITICAL: Overly generic requirement rejected: '{req.get('name')}'")
            continue
        
        valid_requirements.append(req)
    
    if len(valid_requirements) == 0:
        raise ValueError(
            "CRITICAL: No valid requirements extracted. All requirements were too generic. "
            "The extraction prompt needs to be more specific about breaking down requirements."
        )
    
    if len(valid_requirements) == 1 and valid_requirements[0].get("name", "").lower() in generic_patterns:
        logger.error(f"CRITICAL: Only one generic requirement extracted: '{valid_requirements[0].get('name')}'")
        logger.error("This indicates the extraction prompt is not working correctly.")
        # Don't fail completely, but log the issue
    
    # CRITICAL: Sort requirements by req_id for deterministic ordering
    # This ensures same requirements always appear in same order
    frozen_requirements = sorted(valid_requirements, key=lambda r: r.get("req_id", ""))
    result["frozen_requirements"] = frozen_requirements
    
    # CRITICAL: Normalize requirement names and evidence for consistency
    for req in frozen_requirements:
        # Normalize whitespace in name and evidence
        if "name" in req and isinstance(req["name"], str):
            req["name"] = ' '.join(req["name"].strip().split())
        if "jd_evidence" in req and isinstance(req["jd_evidence"], str):
            req["jd_evidence"] = ' '.join(req["jd_evidence"].strip().split())
    
    # Cache the result with timestamp
    current_time = time.time()
    
    # Remove oldest entries if cache is full (LRU eviction)
    while len(_FROZEN_REQUIREMENTS_CACHE) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(_FROZEN_REQUIREMENTS_CACHE))
        del _FROZEN_REQUIREMENTS_CACHE[oldest_key]
        logger.debug(f"Evicted oldest cache entry (LRU): {oldest_key[:16]}...")
    
    # Add new entry to cache (at end - most recently used)
    # CRITICAL: Use deepcopy to prevent cache corruption
    import copy
    _FROZEN_REQUIREMENTS_CACHE[job_hash] = {
        "data": copy.deepcopy(result),
        "timestamp": current_time
    }
    
    req_count = len(result['frozen_requirements'])
    logger.info(f"✅ Cached frozen requirements: {req_count} requirements for job_id={job_id}, hash={job_hash[:16]}... (cache size: {len(_FROZEN_REQUIREMENTS_CACHE)}/{MAX_CACHE_SIZE})")
    logger.info(f"CRITICAL: Same job (hash={job_hash[:16]}...) will now always return these {req_count} cached requirements")
    # Log requirement IDs for validation
    if req_count > 0:
        req_ids = [req.get('req_id', '') for req in result['frozen_requirements']]
        logger.info(f"Extracted requirement IDs: {req_ids}")
    
    return result


def get_cached_requirements(job_description: str, job_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get cached frozen requirements if available.
    
    Args:
        job_description: Raw job description text
        job_id: Optional job identifier
    
    Returns:
        Cached requirements dict or None if not cached
    """
    job_hash = get_job_hash(job_description, job_id)
    if job_hash in _FROZEN_REQUIREMENTS_CACHE:
        cache_entry = _FROZEN_REQUIREMENTS_CACHE[job_hash]
        # Handle both new format (with timestamp) and legacy format
        if isinstance(cache_entry, dict) and "data" in cache_entry:
            return cache_entry["data"]
        return cache_entry
    return None


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics for monitoring.
    
    Returns:
        Dict with cache size, max size, TTL, and memory usage estimate
    """
    current_time = time.time()
    expired_count = 0
    total_size_bytes = 0
    
    for entry in _FROZEN_REQUIREMENTS_CACHE.values():
        if isinstance(entry, dict) and "timestamp" in entry:
            age = current_time - entry["timestamp"]
            if age > CACHE_TTL_SECONDS:
                expired_count += 1
        # Estimate size (rough calculation)
        if isinstance(entry, dict) and "data" in entry:
            total_size_bytes += len(json.dumps(entry["data"]).encode('utf-8'))
        elif isinstance(entry, dict):
            total_size_bytes += len(json.dumps(entry).encode('utf-8'))
    
    return {
        "cache_size": len(_FROZEN_REQUIREMENTS_CACHE),
        "max_cache_size": MAX_CACHE_SIZE,
        "cache_usage_percent": (len(_FROZEN_REQUIREMENTS_CACHE) / MAX_CACHE_SIZE * 100) if MAX_CACHE_SIZE > 0 else 0,
        "ttl_seconds": CACHE_TTL_SECONDS,
        "expired_entries": expired_count,
        "estimated_size_mb": round(total_size_bytes / (1024 * 1024), 2)
    }


def clear_cache():
    """Clear the frozen requirements cache (for testing)."""
    global _FROZEN_REQUIREMENTS_CACHE
    size_before = len(_FROZEN_REQUIREMENTS_CACHE)
    _FROZEN_REQUIREMENTS_CACHE.clear()
    logger.info(f"Cleared frozen requirements cache ({size_before} entries removed)")


def cleanup_expired_cache():
    """
    Remove expired entries from cache.
    Call this periodically to free memory.
    """
    global _FROZEN_REQUIREMENTS_CACHE
    current_time = time.time()
    expired_keys = []
    
    for key, entry in _FROZEN_REQUIREMENTS_CACHE.items():
        if isinstance(entry, dict) and "timestamp" in entry:
            age = current_time - entry["timestamp"]
            if age > CACHE_TTL_SECONDS:
                expired_keys.append(key)
    
    for key in expired_keys:
        del _FROZEN_REQUIREMENTS_CACHE[key]
    
    if expired_keys:
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    return len(expired_keys)
