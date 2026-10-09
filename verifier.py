from google import genai
from llm_utils import generate_with_retry
from trust import trust_url, trust_label
import json

client = genai.Client()

def verify_sources(subtopic, sources):
    combined = "\n\n".join(
        f"[{i+1}] (Trust: {trust_url(s.get('href',''))}/10) {s['title']} ({s.get('href','')}): {s['body']}"
        for i, s in enumerate(sources)
    )
    prompt = f"""You are a research verification agent.
Subtopic: {subtopic}

Sources (with trust scores out of 10):
{combined}

Perform the following analysis:
1. List the 3-5 main factual claims relevant to this subtopic.
2. For each claim state:
   - Whether sources agree, partially agree, or contradict
   - The trust weight of supporting sources (high-trust sources carry more weight)
   - Whether the claim is high-confidence, moderate-confidence, or low-confidence
3. Identify any direct contradictions between sources and state both positions.
4. Note any claims supported by only one source as unverified.
5. Assess overall source diversity and recency.

Return ONLY valid JSON:
{{
  "claims": [
    {{
      "claim": "...",
      "consensus": "agree|partial|contradict",
      "confidence": "high|moderate|low",
      "supporting_sources": [1, 2],
      "contradiction_note": "..." 
    }}
  ],
  "contradictions": ["..."],
  "unverified_claims": ["..."],
  "overall_confidence": "high|moderate|low",
  "diversity_note": "..."
}}
"""
    response = generate_with_retry(client, prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(text)
    except Exception:
        return {
            "claims": [],
            "contradictions": [],
            "unverified_claims": [],
            "overall_confidence": "low",
            "diversity_note": "Verification parsing failed.",
            "raw": text
        }


def compute_section_confidence(verification_result):
    conf = verification_result.get("overall_confidence", "low")
    contradictions = verification_result.get("contradictions", [])
    unverified = verification_result.get("unverified_claims", [])
    if conf == "high" and not contradictions:
        return 90
    elif conf == "high":
        return 75
    elif conf == "moderate" and not contradictions:
        return 65
    elif conf == "moderate":
        return 50
    elif unverified and not contradictions:
        return 40
    return 30