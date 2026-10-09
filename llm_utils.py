import time
import re
import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER_CONFIG = {
    "groq": {
        "type":     "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "models":   ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    },
    "openai": {
        "type":     "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "models":   ["gpt-4o-mini", "gpt-3.5-turbo"]
    },
    "gemini": {
        "type":    "gemini",
        "models":  ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    }
}

# Your exact agent → provider → key mapping
AGENT_CONFIG = {
    "planner":   {"provider": "groq",   "key_env": "api_key_planner"},
    "verifier":  {"provider": "groq",   "key_env": "api_key_verifier"},
    "extractor": {"provider": "openai", "key_env": "api_key_extractor"},
    "writer":    {"provider": "gemini", "key_env": "api_key_writer"},
}

class UnifiedResponse:
    """Single response wrapper — every agent uses .text regardless of provider."""
    def __init__(self, text):
        self.text = text

def get_client(agent_name):
    config        = AGENT_CONFIG.get(agent_name)
    if not config:
        raise ValueError(f"Unknown agent: {agent_name}. Add it to AGENT_CONFIG.")

    provider_name = config["provider"]
    provider      = PROVIDER_CONFIG[provider_name]
    api_key       = os.environ.get(config["key_env"]) or os.environ.get("api_key")

    if not api_key:
        raise ValueError(
            f"No API key found for agent '{agent_name}'. "
            f"Set {config['key_env']} in your .env file."
        )

    if provider["type"] == "gemini":
        from google import genai
        return {
            "type":     "gemini",
            "client":   genai.Client(api_key=api_key),
            "models":   provider["models"]
        }

    elif provider["type"] == "openai_compatible":
        from openai import OpenAI
        return {
            "type":     "openai_compatible",
            "client":   OpenAI(base_url=provider["base_url"], api_key=api_key),
            "models":   provider["models"]
        }


def generate_with_retry(client_info, contents, max_retries=3, base_delay=5):
    # Legacy support: raw gemini client passed directly
    if not isinstance(client_info, dict):
        from google import genai
        models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        return _gemini_generate(client_info, contents, models, max_retries, base_delay)

    if client_info["type"] == "gemini":
        return _gemini_generate(
            client_info["client"], contents,
            client_info["models"], max_retries, base_delay
        )
    elif client_info["type"] == "openai_compatible":
        return _openai_generate(
            client_info["client"], contents,
            client_info["models"], max_retries, base_delay
        )


def _gemini_generate(client, contents, models, max_retries, base_delay):
    from google.genai.errors import ServerError, ClientError
    last_err = None

    for model in models:
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(model=model, contents=contents)
                return UnifiedResponse(resp.text)

            except ClientError as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    match = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
                    wait  = float(match.group(1)) + 2 if match else 35
                    print(f"[Gemini] {model} quota exhausted. Waiting {wait:.0f}s...")
                    time.sleep(wait)
                    break
                else:
                    raise

            except ServerError as e:
                last_err = e
                if attempt == max_retries - 1:
                    print(f"[Gemini] {model} failed after {max_retries} attempts. Trying next model...")
                    break
                wait = base_delay * (2 ** attempt)
                print(f"[Gemini] {model} overloaded. Retrying in {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)

    raise RuntimeError(f"[Gemini] All models exhausted. Last error: {last_err}")


def _openai_generate(client, contents, models, max_retries, base_delay):
    last_err = None

    for model in models:
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": contents}],
                    max_tokens=4096,
                    temperature=0.3
                )
                return UnifiedResponse(response.choices[0].message.content)

            except Exception as e:
                last_err = e
                msg = str(e).lower()

                if "429" in str(e) or "rate_limit" in msg or "quota" in msg:
                    print(f"[OpenAI/Groq] {model} rate limited. Waiting 30s...")
                    time.sleep(30)
                    break

                if "insufficient_quota" in msg or "billing" in msg:
                    print(f"[OpenAI] {model} billing quota exceeded. Trying next model...")
                    break

                if attempt == max_retries - 1:
                    print(f"[OpenAI/Groq] {model} failed after {max_retries} attempts. Trying next model...")
                    break

                wait = base_delay * (2 ** attempt)
                print(f"[OpenAI/Groq] {model} error. Retrying in {wait}s... ({attempt+1}/{max_retries})")
                time.sleep(wait)

    raise RuntimeError(f"[OpenAI/Groq] All models exhausted. Last error: {last_err}")