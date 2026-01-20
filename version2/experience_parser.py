"""
Deterministic Experience Parser

Parses experience entries and calculates durations from raw date strings.
Handles two-column resumes by ignoring visual order and using pattern matching.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date

logger = logging.getLogger(__name__)

# Experience type keywords
EXPERIENCE_TYPE_KEYWORDS = {
    "internship": ["intern", "internship", "trainee", "training"],
    "academic": ["hackathon", "project", "coursework", "competition", "contest", "research project", "academic project"],
    "freelance": ["freelance", "freelancer", "contract", "consulting", "consultant", "self-employed"],
    "part_time": ["part-time", "part time", "parttime"],
    "full_time": []  # Default fallback
}

MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12
}


def classify_experience_type(entry_text: str) -> str:
    """
    Classify experience entry type based on keywords.
    
    Args:
        entry_text: Text containing role/company/description
        
    Returns:
        One of: "full-time", "internship", "part-time", "freelance", "academic"
    """
    text_lower = entry_text.lower()
    
    # Check in order of specificity
    for exp_type, keywords in EXPERIENCE_TYPE_KEYWORDS.items():
        if exp_type == "full_time":
            continue  # Check last
        if any(keyword in text_lower for keyword in keywords):
            logger.debug(f"Classified as {exp_type}: {entry_text[:50]}...")
            return exp_type
    
    return "full-time"  # Default


def parse_date_string(date_str: str) -> Optional[datetime]:
    """
    Parse date string to datetime object.
    Handles formats like: "May 2025", "2025-05", "Jan 2024 - Present"
    
    Args:
        date_str: Date string to parse
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Handle "Present", "Current", "Now"
    if date_str.lower() in ["present", "current", "now"]:
        return datetime.now()
    
    # Try manual parsing for "Month Year" format (most common)
    month_year_pattern = r'(\w+)\s+(\d{4})'
    match = re.search(month_year_pattern, date_str, re.IGNORECASE)
    if match:
        month_str = match.group(1).lower()
        year_str = match.group(2)
        if month_str in MONTH_NAMES:
            try:
                return datetime(int(year_str), MONTH_NAMES[month_str], 1)
            except Exception as e:
                logger.debug(f"Error parsing month/year: {e}")
    
    # Try "YYYY-MM" format
    yyyy_mm_pattern = r'(\d{4})-(\d{1,2})'
    match = re.search(yyyy_mm_pattern, date_str)
    if match:
        try:
            year = int(match.group(1))
            month = int(match.group(2))
            if 1 <= month <= 12:
                return datetime(year, month, 1)
        except Exception as e:
            logger.debug(f"Error parsing YYYY-MM: {e}")
    
    # Try "YYYY" format
    year_pattern = r'^(\d{4})$'
    match = re.search(year_pattern, date_str)
    if match:
        try:
            year = int(match.group(1))
            return datetime(year, 1, 1)
        except Exception as e:
            logger.debug(f"Error parsing year: {e}")
    
    logger.warning(f"Failed to parse date: {date_str}")
    return None


def parse_date_range(date_range_str: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parse date range string to start and end dates.
    
    Args:
        date_range_str: Date range like "May 2025 - Present" or "Jan 2024 - Apr 2024"
        
    Returns:
        Tuple of (start_date, end_date), either may be None
    """
    if not date_range_str:
        return None, None
    
    # Split on common separators
    separators = [" - ", " – ", " -", "- ", " to ", "–"]
    start_str = None
    end_str = None
    
    for sep in separators:
        if sep in date_range_str:
            parts = date_range_str.split(sep, 1)
            if len(parts) == 2:
                start_str = parts[0].strip()
                end_str = parts[1].strip()
                break
    
    if not start_str:
        # Single date or unparseable
        parsed = parse_date_string(date_range_str)
        return parsed, None
    
    start_date = parse_date_string(start_str)
    end_date = parse_date_string(end_str) if end_str else None
    
    return start_date, end_date


def calculate_duration_months(start_date: Optional[datetime], end_date: Optional[datetime]) -> int:
    """
    Calculate duration in months between two dates.
    
    Args:
        start_date: Start date (required)
        end_date: End date (None means Present/Current)
        
    Returns:
        Number of months (0 if dates invalid)
    """
    if not start_date:
        return 0
    
    if not end_date:
        end_date = datetime.now()
    
    # Calculate months manually
    try:
        # Calculate year and month difference
        year_diff = end_date.year - start_date.year
        month_diff = end_date.month - start_date.month
        
        total_months = year_diff * 12 + month_diff
        
        # If end day is before start day in same month, subtract one month
        # But since we use day=1 for all dates, this shouldn't be an issue
        # However, we add 1 month if both dates are in same month to count it as 1 month minimum
        if total_months == 0 and start_date != end_date:
            if start_date.year == end_date.year and start_date.month == end_date.month:
                total_months = 1
        
        return max(0, total_months)
    except Exception as e:
        logger.warning(f"Error calculating duration: {e}")
        return 0


def format_duration(months: int) -> str:
    """
    Format duration as "X years Y months" or "X months".
    
    Args:
        months: Number of months
        
    Returns:
        Formatted string like "2 years 3 months" or "5 months"
    """
    if months == 0:
        return "0 months"
    
    years = months // 12
    remaining_months = months % 12
    
    if years == 0:
        return f"{months} months"
    elif remaining_months == 0:
        return f"{years} year{'s' if years > 1 else ''}"
    else:
        return f"{years} year{'s' if years > 1 else ''} {remaining_months} month{'s' if remaining_months > 1 else ''}"


def parse_duration_string(duration_str: str) -> float:
    """
    Parse duration string back to years (for internal scoring use).
    
    Args:
        duration_str: Formatted string like "2 years 3 months" or "5 months"
        
    Returns:
        Years as float (for backward compatibility with scoring functions)
    """
    if not duration_str or duration_str == "0 months":
        return 0.0
    
    months = 0
    years_match = re.search(r'(\d+)\s*year', duration_str, re.IGNORECASE)
    months_match = re.search(r'(\d+)\s*month', duration_str, re.IGNORECASE)
    
    if years_match:
        months += int(years_match.group(1)) * 12
    if months_match:
        months += int(months_match.group(1))
    
    return round(months / 12.0, 1) if months > 0 else 0.0


def calculate_experience_breakdown(experience_entries: List[Dict[str, any]]) -> Dict[str, str]:
    """
    Calculate experience breakdown from parsed entries.
    
    Args:
        experience_entries: List of dicts with keys: "type", "start_date", "end_date", "date_range"
        
    Returns:
        Dict with keys: full_time, internship, freelance, part_time, contract, academic, total
        All values are formatted duration strings
    """
    breakdown_months = {
        "full_time": 0,
        "internship": 0,
        "freelance": 0,
        "part_time": 0,
        "contract": 0,
        "academic": 0
    }
    
    for entry in experience_entries:
        exp_type = entry.get("type", "full-time")
        
        # Get dates
        start_date = entry.get("start_date")
        end_date = entry.get("end_date")
        
        # If dates not parsed, try parsing date_range
        if not start_date and entry.get("date_range"):
            start_date, end_date = parse_date_range(entry["date_range"])
        
        # Calculate duration
        months = calculate_duration_months(start_date, end_date)
        
        # Add to appropriate category
        if exp_type in breakdown_months:
            breakdown_months[exp_type] += months
        else:
            breakdown_months["full_time"] += months  # Default
        
        logger.debug(f"Entry: {entry.get('role', 'Unknown')[:30]}... Type: {exp_type}, Duration: {months} months")
    
    # Format all durations
    breakdown = {
        "full_time": format_duration(breakdown_months["full_time"]),
        "internship": format_duration(breakdown_months["internship"]),
        "freelance": format_duration(breakdown_months["freelance"]),
        "part_time": format_duration(breakdown_months["part_time"]),
        "contract": format_duration(breakdown_months["contract"]),
        "academic": format_duration(breakdown_months["academic"]),
    }
    
    # Calculate total
    total_months = sum(breakdown_months.values())
    breakdown["total"] = format_duration(total_months)
    
    logger.info(f"Experience breakdown: {breakdown}")
    logger.info(f"Total entries processed: {len(experience_entries)}")
    
    return breakdown
