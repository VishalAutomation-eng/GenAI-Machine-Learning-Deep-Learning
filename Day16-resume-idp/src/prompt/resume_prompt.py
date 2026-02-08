def build_resume_prompt(resume_text: str) -> str:
    return f"""
You are an Intelligent Document Processing system specialized in Resume Information Extraction.

TASK:
Extract structured information from the resume text.

FIELDS TO EXTRACT:
- name
- qualification
- skills
- certifications

RULES:
- Extract name ONLY if explicitly mentioned (e.g., "Name:", header line)
- Do NOT infer or guess the name
- If name is missing, return "NOT_PRESENT"


OUTPUT FORMAT:
{{
    "name": "",
    "qualification": "",
    "skills": "",
    "certifications": ""
}}

RESUME TEXT:
\"\"\"
{resume_text}
\"\"\"
"""
