from ddgs import DDGS
from trust import trust_url, filter_trusted

def web_search(query, max_results=10):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        print(f"[Search] Error: {e}")
        return []

def dedupe_sources(sources):
    seen = set()
    unique = []
    for s in sources:
        url = s.get("href")
        if url and url not in seen:
            seen.add(url)
            unique.append(s)
    return unique

def gather_sources(query, max_results=10, threshold=7):
    raw = web_search(query, max_results=max_results)
    raw = dedupe_sources(raw)
    trusted = filter_trusted(raw, threshold=threshold)
    return trusted

def enrich_with_trust(sources):
    for s in sources:
        s["trust_score"] = trust_url(s.get("href", ""))
        s["trust_label"] = "High" if s["trust_score"] >= 9 else "Medium" if s["trust_score"] >= 7 else "Low"
    return sources