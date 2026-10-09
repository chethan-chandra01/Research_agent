from datetime import datetime

def trust_url(url):
    score = 5
    if ".gov" in url:        score = 10
    elif ".edu" in url:      score = 9
    elif "wikipedia" in url: score = 8
    elif "ibm" in url:       score = 8
    elif "oracle" in url:    score = 8
    elif "microsoft" in url: score = 8
    elif "reuters" in url:   score = 7
    elif "bbc" in url:       score = 7
    elif "nature" in url:    score = 9
    elif "springer" in url:  score = 9
    elif "ieee" in url:      score = 10
    elif "acm" in url:       score = 10
    elif "arxiv" in url:     score = 9
    elif "ncbi" in url:      score = 9
    elif "nih" in url:       score = 10
    elif "coursera" in url:  score = 7
    elif "mit" in url:       score = 9
    elif "stanford" in url:  score = 9
    return score

def trust_label(score):
    if score >= 9:   return "High"
    elif score >= 7: return "Medium"
    else:            return "Low"

def apply_recency_penalty(score, year):
    if not year:
        return score
    age = datetime.now().year - int(year)
    if age > 10:  return max(score - 2, 1)
    elif age > 5: return max(score - 1, 1)
    return score

def filter_trusted(sources, threshold=7):
    return [s for s in sources if trust_url(s.get("href", "")) >= threshold]

def get_consensus_trust(sources, claim):
    count = len(sources)
    if count >= 4: return "high-consensus"
    elif count >= 2: return "moderate-consensus"
    return "low-consensus"

def source_diversity_check(sources):
    domains = [s.get("href", "").split("/")[2] for s in sources if s.get("href")]
    unique_domains = set(domains)
    warnings = []
    if len(unique_domains) < 3:
        warnings.append("Low source diversity — most sources from same domain.")
    scores = [trust_url(s.get("href", "")) for s in sources]
    low_trust = [s for s in scores if s < 7]
    if len(low_trust) > len(sources) // 2:
        warnings.append("More than half of sources have low trust scores.")
    return warnings