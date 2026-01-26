from playwright.sync_api import sync_playwright
import json


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    url = "https://www.linkedin.com/jobs/search/?currentJobId=4337895748&origin=JOBS_HOME_JYMBII"
    page.goto(url)
    
    # Wait for page to load
    page.wait_for_load_state("networkidle")
    
    # Web scrape the page
    print("\n" + "="*80)
    print("SCRAPING PAGE CONTENT")
    print("="*80)
    
    # Get page title
    page_title = page.title()
    print(f"\nPage Title: {page_title}")
    
    # Get page URL
    current_url = page.url
    print(f"Current URL: {current_url}")
    
    # Get full HTML content
    html_content = page.content()
    print(f"\nHTML Content Length: {len(html_content)} characters")
    
    # Get page text content
    text_content = page.inner_text("body")
    print(f"Text Content Length: {len(text_content)} characters")
    
    # Extract specific elements (example: job title, company name)
    try:
        # Try to find job title (LinkedIn specific selectors)
        job_title = None
        title_selectors = [
            '.jobs-details-top-card__job-title',
            'h1[data-test-id*="job-title"]',
            '.topcard__title',
            'h1',
            '.job-title'
        ]
        for selector in title_selectors:
            try:
                element = page.query_selector(selector)
                if element:
                    job_title = element.inner_text().strip()
                    print(f"\nJob Title Found: {job_title}")
                    break
            except:
                continue
        
        # Try to find company name
        company_name = None
        company_selectors = [
            '.jobs-details-top-card__company-name',
            '[data-test-id*="company"]',
            '.topcard__org-name',
            '.company-name',
            'a[href*="/company/"]',
            '.jobs-details-top-card__company-info'
        ]
        for selector in company_selectors:
            try:
                element = page.query_selector(selector)
                if element:
                    company_name = element.inner_text().strip()
                    print(f"Company Name Found: {company_name}")
                    break
            except:
                continue
        
        # If company name not found, extract from text_content
        if not company_name:
            import re
            # Look for pattern: "Job Title\nCompany Name  Location"
            company_match = re.search(rf'{re.escape(job_title) if job_title else ""}\n([A-Za-z0-9\s&]+)\s+([A-Za-z,\s]+,\s*[A-Za-z,\s]+)', text_content)
            if company_match:
                company_name = company_match.group(1).strip()
                print(f"Company Name Extracted from text: {company_name}")
        
        # Try to find job description using multiple methods
        description = None
        desc_selectors = [
            '.jobs-description__text',
            '.jobs-box__html-content',
            '[data-test-id*="description"]',
            '.job-description',
            '#job-description',
            '.jobs-description-content__text',
            'div[data-test-id*="job-details"]'
        ]
        for selector in desc_selectors:
            try:
                element = page.query_selector(selector)
                if element:
                    description = element.inner_text().strip()
                    print(f"Description Found via selector: {len(description)} characters")
                    break
            except:
                continue
        
        # If description not found, extract from text_content
        if not description:
            import re
            # Extract everything from "Job Description" to "Additional Information" or end
            desc_match = re.search(r'Job Description\s*\n\n(.*?)(?:\n\nAdditional Information|\n\nShow more|\Z)', text_content, re.DOTALL)
            if desc_match:
                description = desc_match.group(1).strip()
                print(f"Description Extracted from text: {len(description)} characters")
            else:
                # Fallback: get everything after "Job Description"
                desc_match = re.search(r'Job Description\s*\n\n(.*)', text_content, re.DOTALL)
                if desc_match:
                    description = desc_match.group(1).strip()
                    # Limit to reasonable length (remove trailing similar jobs section)
                    description = re.split(r'\n\nSimilar jobs|\n\nReferrals', description)[0]
                    print(f"Description Extracted (fallback): {len(description)} characters")
        
        # Extract location
        location = None
        location_selectors = [
            '.jobs-details-top-card__primary-description-without-tagline',
            '.jobs-details-top-card__bullet',
            '[data-test-id*="location"]'
        ]
        for selector in location_selectors:
            try:
                element = page.query_selector(selector)
                if element:
                    location = element.inner_text().strip()
                    print(f"Location Found: {location}")
                    break
            except:
                continue
        
        # If location not found, extract from text_content
        if not location:
            import re
            # Look for pattern: "Company Name  Location"
            if company_name:
                location_match = re.search(rf'{re.escape(company_name)}\s+([A-Za-z,\s]+,\s*[A-Za-z,\s]+)', text_content)
                if location_match:
                    location = location_match.group(1).strip()
                    print(f"Location Extracted from text: {location}")
        
        # Extract additional structured information from text_content
        qualifications = None
        skills = None
        
        if text_content:
            import re
            # Extract Qualifications section
            qual_match = re.search(r'Qualifications\s*\n\n(.*?)(?:\n\nSuggested skills|\n\nAdditional Information|\Z)', text_content, re.DOTALL)
            if qual_match:
                qualifications = qual_match.group(1).strip()
            
            # Extract Suggested skills
            skills_match = re.search(r'Suggested skills\s*\n\n(.*?)(?:\n\nAdditional Information|\Z)', text_content, re.DOTALL)
            if skills_match:
                skills = skills_match.group(1).strip()
        
        # Create scraped data dictionary with all information
        scraped_data = {
            "url": current_url,
            "title": page_title,
            "job_title": job_title,
            "company_name": company_name,
            "location": location,
            "description": description,
            "qualifications": qualifications,
            "suggested_skills": skills,
            "text_content": text_content,  # Full text content (not truncated)
            "html_length": len(html_content)
        }
        
        # Print scraped data summary
        print("\n" + "="*80)
        print("SCRAPED DATA SUMMARY")
        print("="*80)
        print(json.dumps(scraped_data, indent=2, ensure_ascii=False))
        
        # Save to file (optional)
        with open("scraped_data.json", "w", encoding="utf-8") as f:
            json.dump(scraped_data, f, indent=2, ensure_ascii=False)
        print(f"\nScraped data saved to: scraped_data.json")
        
    except Exception as e:
        print(f"\nError during scraping: {e}")
        import traceback
        traceback.print_exc()
    
    # Keep browser open - wait for user to press Enter
    print("\n" + "="*80)
    print("Browser is open. Press Enter to close...")
    print("="*80)
    input()  # Wait for Enter key
    
    browser.close()

