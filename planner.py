from google import genai
from llm_utils import generate_with_retry
import json

client = genai.Client()

def plan_research(topic):
    prompt = f"""You are a research planning agent.
Given a research topic, generate a JSON object with:
- "subtopics": list of 8-10 deep, specific subtopics covering foundational, current, and emerging aspects
- "queries": list of 8-10 specific web search queries (one per subtopic, search-engine friendly)
- "description": one sentence describing each subtopic's importance

Topic: {topic}

Return ONLY valid JSON, no markdown, no explanation. Format:
{{
  "subtopics": ["...", "..."],
  "queries": ["...", "..."],
  "descriptions": ["...", "..."]
}}
"""
    response = generate_with_retry(client, prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()
    return json.loads(text)


def refine_queries(subtopics, user_edits):
    refined = []
    for i, subtopic in enumerate(subtopics):
        if i < len(user_edits) and user_edits[i].strip():
            refined.append(user_edits[i].strip())
        else:
            refined.append(subtopics[i])
    return refined