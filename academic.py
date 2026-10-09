import requests
import urllib.parse
import xml.etree.ElementTree as ET
import time
from collections import Counter

ARXIV_BASE = "https://export.arxiv.org/api/query"
S2_BASE    = "https://api.semanticscholar.org/graph/v1/paper/search"

def search_arxiv(query, max_results=8):
    params = {
        "search_query": f"all:{urllib.parse.quote(query)}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    try:
        resp = requests.get(ARXIV_BASE, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Academic] arXiv error: {e}")
        return []

    ns   = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []

    for entry in root.findall("atom:entry", ns):
        try:
            title   = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
            url     = entry.find("atom:id", ns).text.strip()
            year    = int(entry.find("atom:published", ns).text[:4])
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)][:3]
            papers.append({
                "title":     title,
                "authors":   authors,
                "year":      year,
                "url":       url,
                "abstract":  summary[:500] + "..." if len(summary) > 500 else summary,
                "source":    "arXiv",
                "citations": None
            })
        except Exception:
            continue
    return papers


def search_semantic_scholar(query, max_results=8):
    params = {
        "query":  query,
        "limit":  max_results,
        "fields": "title,authors,year,citationCount,abstract,externalIds,openAccessPdf"
    }
    try:
        resp = requests.get(S2_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[Academic] Semantic Scholar error: {e}")
        return []

    papers = []
    for p in data.get("data", []):
        try:
            ext  = p.get("externalIds", {})
            pdf  = p.get("openAccessPdf") or {}
            url  = (
                pdf.get("url") or
                (f"https://arxiv.org/abs/{ext['ArXiv']}" if "ArXiv" in ext else None) or
                f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"
            )
            abstract = (p.get("abstract") or "")[:500]
            if len(p.get("abstract") or "") > 500:
                abstract += "..."
            year = p.get("year")
            papers.append({
                "title":     p.get("title", "Untitled"),
                "authors":   [a["name"] for a in p.get("authors", [])[:3]],
                "year":      int(year) if year else None,
                "url":       url,
                "abstract":  abstract,
                "source":    "Semantic Scholar",
                "citations": p.get("citationCount")
            })
        except Exception:
            continue
    return papers


def dedupe_papers(arxiv_papers, s2_papers):
    seen   = set()
    merged = []
    for p in arxiv_papers + s2_papers:
        key = p["title"].lower().strip()[:60]
        if key not in seen:
            seen.add(key)
            merged.append(p)
    merged.sort(
        key=lambda x: (x["citations"] is not None, x["citations"] or 0, x["year"] or 0),
        reverse=True
    )
    return merged


def get_academic_papers(query, max_per_source=8):
    print("[Academic Agent] Searching arXiv...")
    arxiv = search_arxiv(query, max_results=max_per_source)
    time.sleep(0.5)
    print("[Academic Agent] Searching Semantic Scholar...")
    s2    = search_semantic_scholar(query, max_results=max_per_source)
    papers = dedupe_papers(arxiv, s2)
    print(f"[Academic Agent] {len(papers)} papers found.")
    return papers


def get_papers_per_year(papers):
    years = [p["year"] for p in papers if p.get("year")]
    return dict(sorted(Counter(years).items()))


def get_top_papers(papers, n=3):
    cited = [p for p in papers if p.get("citations") is not None]
    cited.sort(key=lambda x: x["citations"], reverse=True)
    return cited[:n]


def get_research_timeline(papers):
    by_year = {}
    for p in papers:
        y = p.get("year")
        if y:
            by_year.setdefault(y, []).append(p["title"])
    return dict(sorted(by_year.items()))