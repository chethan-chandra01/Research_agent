from google import genai
from llm_utils import generate_with_retry
from trust import trust_url

client = genai.Client()

def build_source_index(research_data, papers, books):
    ref_list          = []
    ref_counter       = 1
    global_source_map = {}

    for item in research_data:
        for s in item["sources"]:
            url = s.get("href", "")
            if url and url not in global_source_map:
                global_source_map[url] = f"W{ref_counter}"
                score = trust_url(url)
                ref_list.append({
                    "id":    f"W{ref_counter}",
                    "type":  "web",
                    "title": s.get("title", ""),
                    "url":   url,
                    "trust": score
                })
                ref_counter += 1

    paper_refs = []
    for i, p in enumerate(papers):
        authors_str = ", ".join(p.get("authors", []))
        paper_refs.append({
            "id":       f"P{i+1}",
            "type":     "paper",
            "title":    p.get("title", ""),
            "authors":  authors_str,
            "year":     p.get("year", "n.d."),
            "url":      p.get("url", ""),
            "source":   p.get("source", ""),
            "citations": p.get("citations")
        })

    book_refs = []
    for i, b in enumerate(books):
        authors_str = ", ".join(b.get("authors", []))
        book_refs.append({
            "id":      f"B{i+1}",
            "type":    "book",
            "title":   b.get("title", ""),
            "authors": authors_str,
            "year":    b.get("year", "n.d."),
            "url":     b.get("ol_url", "")
        })

    return global_source_map, ref_list, paper_refs, book_refs


def build_sources_section(research_data, global_source_map):
    sources_section = ""
    for item in research_data:
        sources_section += f"\n## Subtopic: {item['subtopic']}\n"
        ver = item.get("verification", {})
        conf_score = item.get("confidence_score", 50)
        sources_section += f"Section confidence: {conf_score}%\n"

        contradictions = ver.get("contradictions", [])
        if contradictions:
            sources_section += f"CONTRADICTIONS DETECTED: {'; '.join(contradictions)}\n"

        unverified = ver.get("unverified_claims", [])
        if unverified:
            sources_section += f"UNVERIFIED CLAIMS: {'; '.join(unverified)}\n"

        sources_section += "\nSources:\n"
        for s in item["sources"]:
            url = s.get("href", "")
            ref_id = global_source_map.get(url, "W?")
            trust  = trust_url(url)
            sources_section += f"[{ref_id}] (Trust {trust}/10) {s.get('title','')}: {s.get('body','')}\n"
    return sources_section


def build_papers_section(paper_refs):
    section = ""
    for p in paper_refs:
        section += f"[{p['id']}] \"{p['title']}\" — {p['authors']} ({p['year']})"
        if p["citations"] is not None:
            section += f" | {p['citations']} citations"
        section += f"\n  Source: {p['source']} | URL: {p['url']}\n\n"
    return section


def build_books_section(book_refs):
    section = ""
    for b in book_refs:
        section += f"[{b['id']}] \"{b['title']}\" — {b['authors']} ({b['year']}) | {b['url']}\n"
    return section


def generate_report(topic, research_data, papers, books, selected_subtopics, tools_data):
    global_source_map, ref_list, paper_refs, book_refs = build_source_index(research_data, papers, books)
    sources_section = build_sources_section(research_data, global_source_map)
    papers_section  = build_papers_section(paper_refs)
    books_section   = build_books_section(book_refs)

    tools_text = ""
    if tools_data.get("tools"):
        tools_text += "Tools: " + ", ".join(t["name"] for t in tools_data["tools"]) + "\n"
    if tools_data.get("datasets"):
        tools_text += "Datasets: " + ", ".join(d["name"] for d in tools_data["datasets"]) + "\n"
    if tools_data.get("methods"):
        tools_text += "Methods: " + ", ".join(m["name"] for m in tools_data["methods"]) + "\n"

    focused = "\n".join(f"- {s}" for s in selected_subtopics)

    prompt = f"""You are a professional academic research report writing agent.

Research Topic: {topic}

Selected focus areas:
{focused}

Web source material with trust scores and verification notes:
{sources_section}

Academic papers:
{papers_section}

Recommended books:
{books_section}

Extracted tools, datasets, and methods:
{tools_text}

Write a complete, publication-quality research report strictly following this IEEE-style structure:

# {topic}

## Abstract
(200 word summary covering motivation, methodology, key findings, and conclusions)

## 1. Introduction
(Background, motivation, problem statement, scope of this report)

## 2. Background and Related Work
(Prior work — cite academic papers [P1][P2] and books [B1][B2] heavily here)

## 3. Key Concepts and Definitions
(Core terminology, foundational ideas — cite trusted web sources [W1][W2])

## 4. Current State of Research
(What research has been done, where the field stands today — cite [P] heavily)

## 5. Applications and Use Cases
(Real-world applications with specific examples — cite [W] and [P])

## 6. Advantages and Limitations
(Benefits and known limitations — if verification found contradictions, present both sides explicitly with a CONTRADICTION NOTE callout)

## 7. Emerging Technologies and New Research Frontiers
(Technologies currently emerging, adjacent fields intersecting with this topic, potential new disciplines forming — be specific and forward-looking, cite recent papers)

## 8. Research Gaps and Unanswered Questions
(Questions existing literature has not answered, problems acknowledged but unsolved, areas of active debate — this is critical for researchers)

## 9. Future Scope and Open Problems
(Where researchers are heading next, what breakthroughs are needed, timeline if discernible)

## 10. Conclusion
(200 word synthesis of key takeaways)

## 11. Resources for Further Study
(Organize as four labeled subsections:
  Must-Read Papers: top 3 papers by citation count with one sentence on why each matters
  Recommended Books: all books from [B] list with one sentence on coverage
  Trusted Web Resources: top 3 [W] sources with trust score 8+ with one sentence on what each offers
  Tools and Datasets: list all extracted tools and datasets with brief description and URL if available)

## References
(List ALL references in order: first [W] web sources, then [P] papers, then [B] books)

Citation rules:
- Web sources: [W1], [W2] etc inline for every major claim
- Academic papers: [P1], [P2] etc inline
- Books: [B1], [B2] etc inline
- Every major claim must have at least one citation
- For low-confidence claims write: (low confidence — limited sources)
- For contradictions write a callout: > CONTRADICTION NOTE: Source A states X while Source B states Y
- For unverified claims write: (unverified — single source only)
- Section 7 and 8 are the most important — be thorough and specific

Return ONLY the report text. No preamble. No explanation.
"""
    response = generate_with_retry(client, prompt)

    all_web_refs   = "\n".join(f"[{r['id']}] {r['title']} (Trust: {r['trust']}/10) — {r['url']}" for r in ref_list)
    all_paper_refs = "\n".join(f"[{p['id']}] {p['title']} — {p['authors']} ({p['year']}) | {p['source']} | {p['url']}" for p in paper_refs)
    all_book_refs  = "\n".join(f"[{b['id']}] {b['title']} — {b['authors']} ({b['year']}) | {b['url']}" for b in book_refs)

    full_refs = f"\n\n---\n## Full Reference List\n\n### Web Sources\n{all_web_refs}\n\n### Academic Papers\n{all_paper_refs}\n\n### Books\n{all_book_refs}"
    return response.text + full_refs