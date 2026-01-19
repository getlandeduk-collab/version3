"""
Example usage of the deterministic job-resume matching system.

Run this file to see the system in action:
    python matching/example_usage.py
"""

import os
import logging
from matching import match_job_resume
from matching.matcher import match_multiple_jobs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample job description
JOB_DESCRIPTION = """
Data Annotator - AI/ML Team

Expert Global Solutions Pvt. Ltd is seeking a Data Annotator for our AI/ML car platform projects.

Requirements:
- Experience: 0 to 1 Year
- Qualification: B.E. / M.Tech (CSE/ETC/EE/Mechanical/Instrumental) / MCA / MSc
- Skills: Python, Machine Learning, Data Analysis
- Understanding of AI/ML concepts and data curation
- Ability to clean and organize data for machine learning models

Responsibilities:
- Annotate data for machine learning models
- Clean and curate datasets
- Quality check labeled data
- Work with annotation tools
- Collaborate with ML engineers

Location: Coimbatore, Tamil Nadu
Training: 2 weeks project training in Aurangabad
"""

# Sample resume
RESUME = """
Hari Varadhan N R
Email: hshri511@gmail.com | Phone: +919597103099

EDUCATION
Sri Krishna College of Engineering and Technology
B.E. in Computer Science and Engineering (Expected: April 2027)

SKILLS
Programming: Python, C++, Java
ML/AI: TensorFlow, Keras, PyTorch, OpenCV, Computer Vision
Data: Pandas, NumPy, Tableau, MySQL
Tools: Git, Nvidia Jetson, N8n

EXPERIENCE (1.42 years total)

AI Research Intern | Arivara AI | Nov-Dec 2024
- Built AI-powered classroom tutor using GPT-4o-mini
- Implemented hybrid retrieval with Weaviate, Neo4j, Redis
- Developed RAG+LLM model for document eligibility

AI/ML Practitioner | Various Projects | 2023-2024
- Crop disease detection using TensorFlow/Keras
- Deepfake detection with DenseNet and ViT
- Voice sales assistant with N8n workflows
- Computer vision and OCR projects

PROJECTS
- Hybrid retrieval system with vector DB and knowledge graphs
- CNN models for image classification
- Data augmentation pipelines
- Automation workflows for CRM

CERTIFICATIONS
- Supervised Machine Learning (DeepLearning.AI, Coursera)

INTERESTS
AI research, Machine Learning, Computer Vision, Hackathons
"""


def example_basic_matching():
    """Example 1: Basic job-resume matching."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Matching")
    print("="*80)
    
    result = match_job_resume(JOB_DESCRIPTION, RESUME)
    
    print(f"\n📊 MATCH RESULT")
    print(f"{'='*80}")
    print(f"Overall Match: {result['match_percentage']}%")
    print(f"\nComponent Breakdown:")
    for component, score in result['breakdown'].items():
        bar = "█" * int(score / 5)  # Visual bar
        print(f"  {component.capitalize():15} {score:5.1f}% {bar}")
    
    print(f"\n📋 Extracted Data Summary:")
    job_data = result['extracted_data']['job']
    resume_data = result['extracted_data']['resume']
    
    print(f"\nJob Requirements:")
    print(f"  Required Skills: {', '.join(job_data.get('skills_required', []))}")
    print(f"  Preferred Skills: {', '.join(job_data.get('skills_preferred', []))}")
    print(f"  Experience: {job_data.get('years_experience', 0)} years")
    print(f"  Education: {job_data.get('education_required', 'N/A')} in {job_data.get('education_field', 'Any')}")
    print(f"  Seniority: {job_data.get('seniority_level', 'N/A')}")
    
    print(f"\nCandidate Profile:")
    print(f"  Skills: {', '.join(resume_data.get('candidate_skills', [])[:10])}...")  # First 10
    print(f"  Experience: {resume_data.get('candidate_years', 0)} years")
    print(f"  Education: {resume_data.get('candidate_education_level', 'N/A')} in {resume_data.get('candidate_education_field', 'Any')}")
    print(f"  Seniority: {resume_data.get('candidate_seniority', 'N/A')}")
    
    print(f"{'='*80}\n")


def example_multiple_jobs():
    """Example 2: Match against multiple jobs."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multiple Job Matching")
    print("="*80)
    
    # Create variations of the job
    jobs = [
        JOB_DESCRIPTION,  # Original
        JOB_DESCRIPTION.replace("0 to 1 Year", "3 to 5 Years"),  # More experience required
        JOB_DESCRIPTION.replace("Data Annotator", "Senior ML Engineer").replace("0 to 1 Year", "5+ Years"),  # Senior role
    ]
    
    print(f"\nMatching resume against {len(jobs)} job postings...")
    results = match_multiple_jobs(jobs, RESUME, return_extracted_data=False)
    
    print(f"\n📊 RESULTS (Sorted by Match %)")
    print(f"{'='*80}")
    for i, result in enumerate(results, 1):
        print(f"\n#{i} - Match: {result['match_percentage']}%")
        print(f"  Breakdown: Skills={result['breakdown']['skills']:.1f}%, "
              f"Exp={result['breakdown']['experience']:.1f}%, "
              f"Resp={result['breakdown']['responsibilities']:.1f}%, "
              f"Edu={result['breakdown']['education']:.1f}%, "
              f"Sen={result['breakdown']['seniority']:.1f}%")
    
    print(f"{'='*80}\n")


def example_determinism():
    """Example 3: Demonstrate determinism."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Determinism Test")
    print("="*80)
    
    print("\nRunning same match 3 times to verify determinism...")
    
    scores = []
    for i in range(3):
        result = match_job_resume(JOB_DESCRIPTION, RESUME, return_extracted_data=False)
        scores.append(result['match_percentage'])
        print(f"  Run {i+1}: {result['match_percentage']}%")
    
    if len(set(scores)) == 1:
        print(f"\n✅ DETERMINISTIC: All runs produced the same score ({scores[0]}%)")
    else:
        print(f"\n⚠️  WARNING: Scores varied: {scores}")
        print("   Note: LLM extraction may vary slightly even with temperature=0")
    
    print(f"{'='*80}\n")


def main():
    """Run all examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("   Please set it: export OPENAI_API_KEY='sk-...'")
        return
    
    print("\n" + "="*80)
    print("DETERMINISTIC JOB-RESUME MATCHING SYSTEM - EXAMPLES")
    print("="*80)
    
    try:
        # Run examples
        example_basic_matching()
        example_multiple_jobs()
        example_determinism()
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

