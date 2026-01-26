Internshala Scraper & Matcher

Scrapes internships from `https://internshala.com/internships/`, filters, and matches to a resume using semantic similarity. Provides both an API (FastAPI) and a CLI.

Setup

1. Create/activate venv (Windows PowerShell):
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

CLI Usage

- Minimal (uses built-in example payload and no resume):
```bash
python main.py
```

- With custom JSON input and resume:
```bash
python main.py --input payload.json --resume resume.txt
```
Where `payload.json` follows:
```json
{
  "url": "https://internshala.com/internships/",
  "categories": ["engineering"],
  "Role": "Machine learning engineer",
  "types": ["full-time"],
  "instruction": "Summarize and group by the {User Resume} and suggest the best internships suitable for them."
}
```

- Print JSON instead of table:
```bash
python main.py --as-json
```

API Usage

Run the API:
```bash
uvicorn main:app --reload
```

POST request:
```http
POST /internships
Content-Type: application/json

{
  "url": "https://internshala.com/internships/",
  "categories": ["engineering"],
  "Role": "Machine learning engineer",
  "types": ["full-time"],
  "instruction": "Summarize and group by the {User Resume} and suggest the best internships suitable for them.",
  "resume": "...paste resume text..."
}
```

Response fields:
- `count`: number of filtered listings
- `internships`: raw filtered internships
- `summary` (optional): LLM summary if `instruction` provided and OpenAI configured
- `internships_matched` (optional): matched internships with `match_score`

Matching Function

The function exported in `main.py`:
```python
from main import match_jobs_to_resume
```
It returns a list of dicts:
```json
[
  {
    "title": "Machine Learning Intern",
    "company": "XYZ AI Labs",
    "location": "Remote",
    "duration": "3 months",
    "stipend": "₹10,000 /month",
    "apply_link": "https://internshala.com/internship/detail/...",
    "match_score": 92
  }
]
```

Notes
- The scraper uses Requests + BeautifulSoup first; if no cards are found, it falls back to headless Selenium. Ensure Chrome is installed for Selenium fallback.
- Sentence embeddings use `sentence-transformers/all-MiniLM-L6-v2`. If that fails, TF-IDF cosine similarity is used.


## version2: Intelligent Job Matching API (Phidata Agents)

This version provides a new FastAPI service at `version2` that accepts a resume PDF and a list of job URLs, scrapes job details using Firecrawl, matches via GPT-4o, and returns ranked matches with summaries.

Setup:

1) Install deps (adds phidata/openai/pdfplumber):
```bash
pip install -r requirements.txt
```

2) Environment variables (create `.env` in project root):
```
OPENAI_API_KEY=...
FIRECRAWL_API_KEY=...
OPENAI_MODEL=gpt-4o
```

3) Run the app:
```bash
python -m version2.main
```

### Endpoint

- `POST /api/match-jobs`

JSON body example:
```json
{
  "pdf": "base64_of_resume_pdf",
  "urls": [
    "https://internshala.com/internships/job1",
    "https://internshala.com/internships/job2"
  ]
}
```

Multipart example (form fields):
```
json_body: {"urls": ["https://..."], "pdf": null}
pdf_file: <attach resume.pdf>
```

Progress:
```
GET /api/progress/{request_id}
```

Response shape:
```json
{
  "candidate_profile": {"skills": [], "experience_summary": "..."},
  "matched_jobs": [
    {
      "rank": 1,
      "job_url": "...",
      "job_title": "...",
      "company": "...",
      "match_score": 0.95,
      "summary": "...",
      "key_matches": ["skill1"],
      "requirements_met": 8,
      "total_requirements": 10
    }
  ],
  "processing_time": "",
  "jobs_analyzed": 2,
  "request_id": "..."
}
```

Notes:
- Uses Phidata Agents (`OpenAIChat` gpt-4o, `FirecrawlTools`) with async concurrency and in-memory caching.
- Set `OPENAI_API_KEY` and `FIRECRAWL_API_KEY` in environment before starting.


