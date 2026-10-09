import os, json
from google import genai
from search import gather_sources, dedupe_sources, enrich_with_trust
from trust import source_diversity_check
from planner import plan_research
from academic import get_academic_papers, get_top_papers
from books import get_books
from verifier import verify_sources, compute_section_confidence
from extractor import extract_tools_and_datasets
from writer import generate_report
from llm_utils import generate_with_retry
from dotenv import load_dotenv

load_dotenv()

MIN_TRUSTED = 3
MAX_SOURCES = 5

client = genai.Client(api_key=os.environ.get("api_key"))

def research_subtopic(subtopic, query):
    print(f"\n[Search Agent] {query}")
    trusted = gather_sources(query, max_results=10)

    if len(trusted) < MIN_TRUSTED:
        print(f"[Search Agent] Only {len(trusted)} sources. Refining...")
        more    = gather_sources(f"{query} site:.edu OR site:.gov OR wikipedia")
        trusted = dedupe_sources(trusted + more)

    trusted = enrich_with_trust(trusted[:MAX_SOURCES])

    diversity_warnings = source_diversity_check(trusted)
    if diversity_warnings:
        for w in diversity_warnings:
            print(f"[Diversity Warning] {w}")

    print(f"[Verification Agent] Verifying {len(trusted)} sources...")
    verification = verify_sources(subtopic, trusted) if trusted else {}
    confidence   = compute_section_confidence(verification)

    return {
        "subtopic":          subtopic,
        "query":             query,
        "sources":           trusted,
        "verification":      verification,
        "confidence_score":  confidence,
        "diversity_warnings": diversity_warnings
    }

def main():
    query = input("Enter your research query: ")

    print("\n[Planner Agent] Generating research plan...")
    plan = plan_research(query)
    print(json.dumps(plan, indent=2))

    print("\n[Academic Agent] Fetching papers...")
    papers = get_academic_papers(query)

    print("\n[Books Agent] Fetching books...")
    books = get_books(query)

    selected = plan["subtopics"]
    queries  = plan["queries"]

    research_data = []
    for subtopic, sub_query in zip(selected, queries):
        result = research_subtopic(subtopic, sub_query)
        research_data.append(result)

    all_text = " ".join(
        s.get("body","") for item in research_data for s in item["sources"]
    )
    print("\n[Extractor Agent] Extracting tools and datasets...")
    tools_data = extract_tools_and_datasets(all_text)

    print("\n[Writer Agent] Generating report...")
    report = generate_report(query, research_data, papers, books, selected, tools_data)

    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n=== REPORT SAVED: research_report.md ===")

if __name__ == "__main__":
    main()