from google import genai
from llm_utils import generate_with_retry
import json

client = genai.Client()

def extract_tools_and_datasets(all_sources_text):
    prompt = f"""You are a research extraction agent.

From the following research source material, extract:
1. Tools and software mentioned (libraries, frameworks, platforms, APIs)
2. Datasets mentioned (benchmark datasets, public datasets, data sources)
3. Methods or algorithms mentioned that researchers can directly apply

Source material:
{all_sources_text[:4000]}

Return ONLY valid JSON:
{{
  "tools": [
    {{"name": "...", "description": "...", "url": "..."}}
  ],
  "datasets": [
    {{"name": "...", "description": "...", "url": "..."}}
  ],
  "methods": [
    {{"name": "...", "description": "..."}}
  ]
}}
"""
    response = generate_with_retry(client, prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(text)
    except Exception:
        return {"tools": [], "datasets": [], "methods": []}