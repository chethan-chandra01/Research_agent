import streamlit as st
import os, json
import plotly.graph_objects as go
from google import genai
from search import gather_sources, dedupe_sources, enrich_with_trust
from trust import source_diversity_check, trust_url
from planner import plan_research
from academic import get_academic_papers, get_papers_per_year, get_top_papers
from books import get_books
from verifier import verify_sources, compute_section_confidence
from extractor import extract_tools_and_datasets
from writer import generate_report
from pdf_export import markdown_to_pdf_bytes
from llm_utils import generate_with_retry
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ResearchAI Agent", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.stApp{background:#0a0e1a;color:#e2e8f0;}
#MainMenu,footer,header{visibility:hidden;}
.hero{text-align:center;padding:3rem 1rem 2rem;}
.hero-title{font-size:3rem;font-weight:700;background:linear-gradient(135deg,#60a5fa 0%,#a78bfa 50%,#34d399 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.03em;line-height:1.1;margin-bottom:0.5rem;}
.hero-sub{color:#64748b;font-size:1rem;margin-bottom:2rem;}
.pipeline{display:flex;justify-content:center;gap:0.4rem;flex-wrap:wrap;margin-bottom:2rem;}
.badge{background:#111827;border:1px solid #1e293b;border-radius:999px;padding:0.3rem 0.9rem;font-size:0.75rem;font-family:'JetBrains Mono',monospace;color:#94a3b8;}
.badge span{margin-right:0.3rem;}
.stTextInput>div>div>input{background:#111827!important;border:1.5px solid #1e293b!important;border-radius:12px!important;color:#e2e8f0!important;font-size:1rem!important;padding:0.85rem 1.1rem!important;}
.stTextInput>div>div>input:focus{border-color:#60a5fa!important;box-shadow:0 0 0 3px rgba(96,165,250,0.15)!important;}
.stTextInput>div>div>input::placeholder{color:#334155!important;}
.stButton>button{background:linear-gradient(135deg,#2563eb,#7c3aed)!important;color:white!important;border:none!important;border-radius:10px!important;padding:0.75rem 2.5rem!important;font-size:1rem!important;font-weight:600!important;width:100%!important;}
.stButton>button:hover{opacity:0.88!important;}
.log-card{background:#0d1117;border:1px solid #1e293b;border-left:3px solid #3b82f6;border-radius:8px;padding:0.65rem 1rem;margin:0.35rem 0;font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#7dd3fc;}
.log-card.success{border-left-color:#34d399;color:#6ee7b7;}
.log-card.warn{border-left-color:#fbbf24;color:#fde68a;}
.log-card.error{border-left-color:#f87171;color:#fca5a5;}
.card{background:#0d1117;border:1px solid #1e293b;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:0.8rem;}
.source-chip{display:inline-block;background:#1e293b;border-radius:6px;padding:0.2rem 0.6rem;font-size:0.7rem;font-family:'JetBrains Mono',monospace;color:#60a5fa;margin:0.12rem;text-decoration:none;}
.trust-high{color:#34d399;}.trust-mid{color:#fbbf24;}
.report-container{background:#0d1117;border:1px solid #1e293b;border-radius:14px;padding:2rem 2.5rem;margin-top:1rem;line-height:1.8;color:#cbd5e1;}
.report-container h1,.report-container h2,.report-container h3{color:#f1f5f9;font-weight:600;}
.stDownloadButton>button{background:#111827!important;border:1.5px solid #1e293b!important;color:#60a5fa!important;border-radius:10px!important;font-weight:500!important;}
.metric-row{display:flex;gap:1rem;margin:1.5rem 0;}
.metric-card{flex:1;background:#0d1117;border:1px solid #1e293b;border-radius:10px;padding:1rem 1.2rem;text-align:center;}
.metric-value{font-size:2rem;font-weight:700;color:#60a5fa;font-family:'JetBrains Mono',monospace;}
.metric-label{font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;margin-top:0.2rem;}
.conf-badge{display:inline-block;border-radius:6px;padding:0.15rem 0.6rem;font-size:0.7rem;font-family:'JetBrains Mono',monospace;font-weight:600;}
.conf-high{background:#052e16;color:#34d399;}.conf-mid{background:#451a03;color:#fbbf24;}.conf-low{background:#450a0a;color:#f87171;}
.contradiction-box{background:#1c1007;border:1px solid #b45309;border-left:4px solid #f59e0b;border-radius:8px;padding:0.8rem 1rem;margin:0.5rem 0;font-size:0.85rem;color:#fde68a;}
.resource-bucket{background:#0d1117;border:1px solid #1e293b;border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;}
.resource-title{font-size:0.72rem;font-family:'JetBrains Mono',monospace;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;}
.divider{border:none;border-top:1px solid #1e293b;margin:2rem 0;}
</style>
""", unsafe_allow_html=True)

MIN_TRUSTED = 3
MAX_SOURCES = 5
MAX_PAPERS  = 10

client = genai.Client(api_key=os.environ.get("api_key"))

def research_subtopic(subtopic, query, log_ph):
    logs = []
    def log(msg, kind="info"):
        logs.append((msg, kind))
        log_ph.markdown("".join(f'<div class="log-card {k}">{m}</div>' for m,k in logs), unsafe_allow_html=True)

    log(f"🔍 Searching: {query}")
    trusted = gather_sources(query, max_results=10)

    if len(trusted) < MIN_TRUSTED:
        log(f"⚠️ Only {len(trusted)} sources. Refining query...", "warn")
        more    = gather_sources(f"{query} site:.edu OR site:.gov OR wikipedia")
        trusted = dedupe_sources(trusted + more)

    trusted = enrich_with_trust(trusted[:MAX_SOURCES])
    div_warnings = source_diversity_check(trusted)
    for w in div_warnings:
        log(f"⚠️ {w}", "warn")

    log(f"✅ {len(trusted)} trusted sources gathered.", "success")
    log("🔎 Cross-verifying claims...")

    verification = verify_sources(subtopic, trusted) if trusted else {}
    confidence   = compute_section_confidence(verification)
    log(f"✅ Verification complete. Section confidence: {confidence}%", "success")

    return {
        "subtopic":          subtopic,
        "query":             query,
        "sources":           trusted,
        "verification":      verification,
        "confidence_score":  confidence,
        "diversity_warnings": div_warnings
    }

def make_year_chart(papers):
    year_data = get_papers_per_year(papers)
    if not year_data:
        return None
    years  = list(year_data.keys())
    counts = list(year_data.values())
    fig = go.Figure(go.Bar(
        x=years, y=counts,
        marker=dict(color=counts, colorscale=[[0,"#1e3a5f"],[0.5,"#2563eb"],[1,"#60a5fa"]], showscale=False),
        text=counts, textposition="outside", textfont=dict(color="#94a3b8", size=11)
    ))
    fig.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#94a3b8", family="Space Grotesk"),
        xaxis=dict(title="Year", tickmode="linear", dtick=1, gridcolor="#1e293b", linecolor="#1e293b"),
        yaxis=dict(title="Papers", gridcolor="#1e293b", linecolor="#1e293b"),
        margin=dict(l=40,r=40,t=20,b=40), height=300, bargap=0.3
    )
    return fig

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">ResearchAI Agent</div>
  <div class="hero-sub">Autonomous · Multi-Agent · Academic-Aware · Citation-Grounded · IEEE Format</div>
  <div class="pipeline">
    <div class="badge"><span>01</span>Planner</div><div class="badge">→</div>
    <div class="badge"><span>02</span>Academic</div><div class="badge">→</div>
    <div class="badge"><span>03</span>Books</div><div class="badge">→</div>
    <div class="badge"><span>04</span>Select Focus</div><div class="badge">→</div>
    <div class="badge"><span>05</span>Search</div><div class="badge">→</div>
    <div class="badge"><span>06</span>Verify</div><div class="badge">→</div>
    <div class="badge"><span>07</span>Extract</div><div class="badge">→</div>
    <div class="badge"><span>08</span>Write</div>
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    query = st.text_input("", placeholder="e.g. Transformer models in medical imaging")
    run   = st.button("Run Research Agent")

for k,v in {"plan":None,"papers":[],"books":[],"query":"","phase":"idle","report":None,"selections":{},"research_data":[],"tools_data":{}}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Phase 1: Plan + Academic + Books ─────────────────────────────────────────
if run and query.strip():
    st.session_state.update({"query":query,"phase":"select","plan":None,"papers":[],"books":[],"report":None,"selections":{},"research_data":[],"tools_data":{}})

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### 🗂 Planner Agent")
    pl = st.empty()
    pl.markdown('<div class="log-card">Generating deep subtopics...</div>', unsafe_allow_html=True)
    plan = plan_research(query)
    st.session_state.plan = plan
    pl.markdown(f'<div class="log-card success">✅ {len(plan["subtopics"])} subtopics generated.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    ac1, ac2 = st.columns(2)

    with ac1:
        st.markdown("#### 📚 Academic Agent")
        al = st.empty()
        al.markdown('<div class="log-card">Searching arXiv & Semantic Scholar...</div>', unsafe_allow_html=True)
        papers = get_academic_papers(query, max_per_source=8)
        st.session_state.papers = papers
        al.markdown(f'<div class="log-card success">✅ {len(papers)} papers found.</div>', unsafe_allow_html=True)

    with ac2:
        st.markdown("#### 📖 Books Agent")
        bl = st.empty()
        bl.markdown('<div class="log-card">Searching Open Library...</div>', unsafe_allow_html=True)
        books = get_books(query, max_results=5)
        st.session_state.books = books
        bl.markdown(f'<div class="log-card success">✅ {len(books)} books found.</div>', unsafe_allow_html=True)

# ── Phase 2: Select + display papers/books ────────────────────────────────────
if st.session_state.phase in ("select","researching","done") and st.session_state.plan:
    plan   = st.session_state.plan
    papers = st.session_state.papers
    books  = st.session_state.books
    query  = st.session_state.query

    # Papers + year chart
    if papers:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("#### 📊 Research Landscape")
        fig = make_year_chart(papers)
        if fig:
            st.markdown('<div style="color:#64748b;font-size:0.85rem;margin-bottom:0.4rem;">📈 Papers published per year in this field</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            with st.expander(f"📄 {len(papers)} Academic Papers", expanded=False):
                for i, p in enumerate(papers[:MAX_PAPERS]):
                    authors_str   = ", ".join(p["authors"]) if p["authors"] else "Unknown"
                    citations_str = f" · {p['citations']:,} citations" if p.get("citations") else ""
                    src_color     = "#60a5fa" if p["source"]=="arXiv" else "#a78bfa"
                    st.markdown(f"""
<div class="card" style="margin-bottom:0.6rem;">
  <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
    <span style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:{src_color};text-transform:uppercase;">{p['source']}</span>
    <span style="font-size:0.7rem;color:#475569;">{p.get('year','n.d.')}{citations_str}</span>
  </div>
  <div style="font-size:0.9rem;font-weight:600;color:#e2e8f0;margin-bottom:0.25rem;">
    <a href="{p['url']}" target="_blank" style="color:#e2e8f0;text-decoration:none;">[P{i+1}] {p['title']}</a>
  </div>
  <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.4rem;">{authors_str}</div>
  <div style="font-size:0.8rem;color:#94a3b8;line-height:1.6;">{p['abstract']}</div>
</div>""", unsafe_allow_html=True)

        with c2:
            with st.expander(f"📖 {len(books)} Recommended Books", expanded=False):
                for i, b in enumerate(books):
                    authors_str = ", ".join(b["authors"]) if b["authors"] else "Unknown"
                    subjects    = ", ".join(b["subjects"][:3]) if b["subjects"] else ""
                    st.markdown(f"""
<div class="card" style="margin-bottom:0.6rem;">
  <div style="font-size:0.68rem;font-family:'JetBrains Mono',monospace;color:#34d399;text-transform:uppercase;margin-bottom:0.25rem;">Open Library</div>
  <div style="font-size:0.9rem;font-weight:600;color:#e2e8f0;margin-bottom:0.25rem;">
    <a href="{b['ol_url']}" target="_blank" style="color:#e2e8f0;text-decoration:none;">[B{i+1}] {b['title']}</a>
  </div>
  <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;">{authors_str} · {b.get('year','n.d.')}</div>
  <div style="font-size:0.75rem;color:#475569;">{subjects}</div>
  <div style="margin-top:0.4rem;">
    <a href="{b['ol_url']}" target="_blank" style="font-size:0.7rem;color:#60a5fa;margin-right:0.8rem;">Open Library</a>
    <a href="{b['gb_url']}" target="_blank" style="font-size:0.7rem;color:#a78bfa;">Google Books</a>
  </div>
</div>""", unsafe_allow_html=True)

    # Subtopic checkboxes
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### ☑️ Select Focus Areas")
    st.markdown('<div style="color:#64748b;font-size:0.85rem;margin-bottom:1rem;">Select which subtopics the agents should research. You can also refine the search query for each.</div>', unsafe_allow_html=True)

    sc1, sc2, _ = st.columns([1,1,4])
    with sc1:
        if st.button("✅ Select All"):
            for s in plan["subtopics"]: st.session_state.selections[s] = True
    with sc2:
        if st.button("❌ Clear All"):
            for s in plan["subtopics"]: st.session_state.selections[s] = False

    cols = st.columns(2)
    refined_queries = {}
    for i, (subtopic, query_str, desc) in enumerate(zip(plan["subtopics"], plan["queries"], plan.get("descriptions", [""] * len(plan["subtopics"])))):
        default = st.session_state.selections.get(subtopic, True)
        with cols[i % 2]:
            checked = st.checkbox(f"**{subtopic}**", value=default, key=f"cb_{i}", help=desc)
            st.session_state.selections[subtopic] = checked
            if checked:
                rq = st.text_input("Search query", value=query_str, key=f"rq_{i}", label_visibility="collapsed")
                refined_queries[subtopic] = rq

    selected_pairs = [
        (s, refined_queries.get(s, q))
        for s, q in zip(plan["subtopics"], plan["queries"])
        if st.session_state.selections.get(s, True)
    ]

    st.markdown(f'<div style="color:#64748b;font-size:0.82rem;margin-top:0.5rem;">{len(selected_pairs)} of {len(plan["subtopics"])} subtopics selected</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    if not selected_pairs:
        st.warning("Please select at least one subtopic.")
    else:
        go_btn = st.button(f"🚀 Start Research on {len(selected_pairs)} Subtopics")

        if go_btn:
            # Search + Verify
            st.markdown("#### 🔍 Search & Verification Agents")
            research_data  = []
            total_sources  = 0
            total_verified = 0

            for i, (subtopic, sub_query) in enumerate(selected_pairs):
                with st.expander(f"Subtopic {i+1} — {subtopic}", expanded=(i==0)):
                    log_ph = st.empty()
                    result = research_subtopic(subtopic, sub_query, log_ph)
                    research_data.append(result)
                    total_sources  += len(result["sources"])
                    total_verified += 1

                    conf = result["confidence_score"]
                    conf_class = "conf-high" if conf>=70 else "conf-mid" if conf>=45 else "conf-low"
                    st.markdown(f'<span class="conf-badge {conf_class}">Confidence: {conf}%</span>', unsafe_allow_html=True)

                    ver = result.get("verification", {})
                    if ver.get("contradictions"):
                        for c in ver["contradictions"]:
                            st.markdown(f'<div class="contradiction-box">⚠️ CONTRADICTION: {c}</div>', unsafe_allow_html=True)

                    if result["sources"]:
                        chips = ""
                        for s in result["sources"]:
                            score = trust_url(s["href"])
                            cls   = "trust-high" if score>=9 else "trust-mid" if score>=7 else ""
                            chips += f'<a class="source-chip {cls}" href="{s["href"]}" target="_blank">[{score}] {s["title"][:40]}...</a>'
                        st.markdown(f'<div style="margin-top:0.5rem;">{chips}</div>', unsafe_allow_html=True)

            st.session_state.research_data = research_data

            # Extractor
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("#### 🔧 Extractor Agent")
            ext_log = st.empty()
            ext_log.markdown('<div class="log-card">Extracting tools, datasets, and methods...</div>', unsafe_allow_html=True)
            all_text = " ".join(s.get("body","") for item in research_data for s in item["sources"])
            tools_data = extract_tools_and_datasets(all_text)
            st.session_state.tools_data = tools_data
            total_extracted = len(tools_data.get("tools",[])) + len(tools_data.get("datasets",[])) + len(tools_data.get("methods",[]))
            ext_log.markdown(f'<div class="log-card success">✅ {total_extracted} items extracted.</div>', unsafe_allow_html=True)

            if total_extracted > 0:
                with st.expander("View Extracted Tools & Datasets", expanded=False):
                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        st.markdown("**Tools**")
                        for t in tools_data.get("tools",[]):
                            url_part = f" — [{t.get('url','')}]({t.get('url','')})" if t.get("url") else ""
                            st.markdown(f"• **{t['name']}**: {t.get('description','')}{url_part}")
                    with tc2:
                        st.markdown("**Datasets**")
                        for d in tools_data.get("datasets",[]):
                            st.markdown(f"• **{d['name']}**: {d.get('description','')}")
                    with tc3:
                        st.markdown("**Methods**")
                        for m in tools_data.get("methods",[]):
                            st.markdown(f"• **{m['name']}**: {m.get('description','')}")

            # Metrics
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            avg_conf = int(sum(r["confidence_score"] for r in research_data) / len(research_data)) if research_data else 0
            st.markdown(f"""
<div class="metric-row">
  <div class="metric-card"><div class="metric-value">{len(selected_pairs)}</div><div class="metric-label">Subtopics</div></div>
  <div class="metric-card"><div class="metric-value">{total_sources}</div><div class="metric-label">Web Sources</div></div>
  <div class="metric-card"><div class="metric-value">{len(papers)}</div><div class="metric-label">Papers</div></div>
  <div class="metric-card"><div class="metric-value">{len(books)}</div><div class="metric-label">Books</div></div>
  <div class="metric-card"><div class="metric-value">{avg_conf}%</div><div class="metric-label">Avg Confidence</div></div>
</div>""", unsafe_allow_html=True)

            # Writer
            st.markdown("#### 📝 Writer Agent")
            with st.spinner("Generating IEEE-format research report..."):
                selected_subtopics = [s for s, _ in selected_pairs]
                report = generate_report(query, research_data, papers, books, selected_subtopics, tools_data)
                st.session_state.report = report
                st.session_state.phase  = "done"

# ── Phase 3: Report + downloads ───────────────────────────────────────────────
if st.session_state.phase == "done" and st.session_state.report:
    report = st.session_state.report
    query  = st.session_state.query
    papers = st.session_state.papers
    books  = st.session_state.books
    tools_data = st.session_state.tools_data

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### 📄 Research Report")
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown(report)
    st.markdown('</div>', unsafe_allow_html=True)

    # Resources panel
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### 🗂 Resources for Further Study")
    r1, r2, r3, r4 = st.columns(4)

    with r1:
        top_papers = sorted([p for p in papers if p.get("citations")], key=lambda x: x["citations"], reverse=True)[:3]
        st.markdown('<div class="resource-bucket"><div class="resource-title">📑 Must-Read Papers</div>', unsafe_allow_html=True)
        for i, p in enumerate(top_papers):
            st.markdown(f'<div style="font-size:0.82rem;margin-bottom:0.5rem;"><a href="{p["url"]}" target="_blank" style="color:#60a5fa;">[P{i+1}] {p["title"][:60]}...</a><br><span style="color:#475569;font-size:0.72rem;">{p.get("citations",0):,} citations</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="resource-bucket"><div class="resource-title">📖 Recommended Books</div>', unsafe_allow_html=True)
        for i, b in enumerate(books[:3]):
            st.markdown(f'<div style="font-size:0.82rem;margin-bottom:0.5rem;"><a href="{b["ol_url"]}" target="_blank" style="color:#34d399;">[B{i+1}] {b["title"][:55]}...</a><br><span style="color:#475569;font-size:0.72rem;">{", ".join(b["authors"][:1])} · {b.get("year","n.d.")}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r3:
        rd = st.session_state.research_data
        top_web = sorted([s for item in rd for s in item["sources"]], key=lambda x: trust_url(x.get("href","")), reverse=True)
        seen_urls = set()
        shown = []
        for s in top_web:
            url = s.get("href","")
            if url not in seen_urls and trust_url(url) >= 8:
                seen_urls.add(url)
                shown.append(s)
            if len(shown) >= 3:
                break
        st.markdown('<div class="resource-bucket"><div class="resource-title">🌐 Trusted Web Resources</div>', unsafe_allow_html=True)
        for s in shown:
            score = trust_url(s.get("href",""))
            st.markdown(f'<div style="font-size:0.82rem;margin-bottom:0.5rem;"><a href="{s["href"]}" target="_blank" style="color:#fbbf24;">{s["title"][:55]}...</a><br><span style="color:#475569;font-size:0.72rem;">Trust: {score}/10</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r4:
        tools = tools_data.get("tools", [])[:3]
        datasets = tools_data.get("datasets", [])[:2]
        st.markdown('<div class="resource-bucket"><div class="resource-title">🔧 Tools & Datasets</div>', unsafe_allow_html=True)
        for t in tools:
            url_tag = f'<a href="{t["url"]}" target="_blank" style="color:#a78bfa;">{t["name"]}</a>' if t.get("url") else f'<span style="color:#a78bfa;">{t["name"]}</span>'
            st.markdown(f'<div style="font-size:0.82rem;margin-bottom:0.4rem;">{url_tag}<br><span style="color:#475569;font-size:0.72rem;">{t.get("description","")[:60]}</span></div>', unsafe_allow_html=True)
        for d in datasets:
            st.markdown(f'<div style="font-size:0.82rem;margin-bottom:0.4rem;"><span style="color:#a78bfa;">{d["name"]}</span><br><span style="color:#475569;font-size:0.72rem;">{d.get("description","")[:60]}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Downloads
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    dl1, dl2, _ = st.columns([1,1,2])
    with dl1:
        st.download_button(
            label="⬇ Download Markdown",
            data=report,
            file_name=f"research_{query[:30].replace(' ','_')}.md",
            mime="text/markdown"
        )
    with dl2:
        with st.spinner("Preparing PDF..."):
            pdf_bytes = markdown_to_pdf_bytes(report, query)
        st.download_button(
            label="⬇ Download PDF",
            data=pdf_bytes,
            file_name=f"research_{query[:30].replace(' ','_')}.pdf",
            mime="application/pdf"
        )

elif run and not query.strip():
    st.warning("Please enter a research query.")