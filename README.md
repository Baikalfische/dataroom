<div align="center">

# 🏠 Real Estate Dataroom

A minimal but working intelligent Q&A prototype for real estate / contract documents: upload PDF / CSV, invoke a single RAG retrieval tool, generate answers with citations via Google Gemini. Focus is on architecture & design trade‑offs, not piling on features.

</div>

## ✨ Current Functionality

This version is a “minimum auditable” RAG skeleton for real estate / contract docs: multiple PDF / CSV uploads, automatic parse → chunk → embed → persist; a single retrieval tool is called by the Agent; answers include traceable citations. Goal: keep the full loop (Ingest → Retrieve → Assemble → Answer with Citations) stable & demo‑able, deferring heavy optimization.

Implemented highlights (only what matters for future improvements):
- Ingestion: multiple uploads; PDF by page, CSV by row (citations map directly to physical page/row).
- Separate stores: PDF / CSV in distinct Chroma collections to avoid cross‑modality noise; enables later per‑type k / filters.
- Embeddings: CLIP 512d (fast + cached) baseline before swapping to domain models or adding rerank.
- Retrieval pipeline: independent TopK per store then merge; debug hooks allow recall inspection.
- Agent loop: LangGraph single‑tool cycle; system prompt enforces “retrieve when unsure / no fabrication.”
- Citations: `[filename p.X]` / `[filename row Y]`; duplicates allowed (later can merge).
- File listing: button shows current documents & page/row counts.
- Local run: single process, no external vector infra.

> These foundations act as the “slab”; later precision tactics (Hybrid, rerank, finer splitting, structured index) can layer on without API breaks.

## 🚧 Pain Points to Improve
- Repetitive clause templates (e.g. rent escalation / liability) blur embeddings.
- Long multi‑topic pages introduce noise when recalled wholesale.
- Precise numeric / date alignment fragile (minor phrasing variance).
- Cross‑document comparisons (e.g. highest escalation) need structured extraction & aggregation.
- Clause traceability limited (page/row only; lacks clause numbering / headings).
- Vague queries (“early termination?”) need rewrite / sub‑query expansion.
- No labeled eval set yet → tuning lacks objective feedback.

## 🛠️ Retrieval Accuracy Strategies

Startup‑pragmatic two tiers: first “structure + coverage”, then “granularity + rerank”.

1. Lightweight (fast, no core refactor)
   - Hybrid: vector + simple inverted (BM25 / keyword scoring) to patch pure semantic misses.
   - Clause header signal: parse “Section 3.2 / Article X” into metadata to boost direct hits.
   - Query expansion: LLM synonyms / field sub‑queries (amount, date) → parallel retrieve + dedupe merge.
   - Candidate dedup: similarity cluster initial TopK to reduce near‑duplicate clauses.
   - Adaptive TopK: numeric / clause‑pattern queries increase structural weighting.

2. Mid Layer (adds small components)
   - Secondary splitting: page → sentence / clause; row → field tokens; retrieve fine units then aggregate upward.
   - Semantic rerank: cross‑encoder reorders ~30 first‑stage candidates.
   - Structured field extraction: regex + light model for amounts / dates / percentages (aux key/value index).
   - Multi‑stage retrieval: keyword / clause pre‑filter → vector expansion → rerank.
   - Hard example logging: store “no answer / irrelevant” feedback for retraining / reweighting.

## 🧱 Architecture Overview

```
╔════════════════════════════════════════════════════════════════╗
║                             User / Browser                      ║
╚════════════════════════════════════════════════════════════════╝
      │ question / upload (PDF, CSV)
      ▼
┌───────────────────────────┐      Single turn flow (message→tool→message)
│         Gradio UI         │──────────────────────────────────────┐
│  - File upload (FS)       │                                      │
│  - Chat history           │◀─ answer + citations ────────────────┘
└───────────────┬──────────┘
      │ invoke
      ▼
   ┌───────────────────┐   single tool bound + system prompt
   │  Agent (LangGraph)│
   │ - process node    │
   │ - execute node    │
   └─────────┬─────────┘
        │ tool call: real_estate_document_search(query)
        ▼
   ┌──────────────────────────────┐
   │           RAG Chain          │
   │ 1. text→vector (CLIP)        │
   │ 2. PDF vector search (k_pdf) │
   │ 3. CSV vector search (k_csv) │
   │ 4. merge / debug             │
   │ 5. LLM answer + citations    │
   └──────────┬─────────┬────────┘
         │          │
   ┌─────────▼───┐   ┌──▼─────────┐
   │ PDF Collection│ │ CSV Collection│ (Chroma persisted)
   └──────────────┘  └─────────────┘
```

## 📂 Project Structure

```
.
├── main.py                     # entrypoint: init agent + launch UI
├── docs/
│   └── system_prompt.txt       # system prompt
├── src/dataroom/agent/agent.py # agent (LangGraph state machine + tool exec)
├── src/dataroom/rag/
│   ├── build_database.py       # vector store mgmt
│   ├── rag_chain.py            # retrieval + context assembly
│   ├── embedder.py             # CLIP embeddings
│   ├── chunks.py               # CSV row-level chunking
│   ├── document_manager.py     # (upload wrapper, can be trimmed)
│   └── rag_config.yaml         # model / retrieval config
├── src/dataroom/tools/rag_tool.py  # RAG tool
├── src/dataroom/tools/parser.py    # PDF/CSV parsing
├── src/dataroom/ui/interface.py    # minimal Gradio UI
├── src/dataroom/utils/utils.py     # helpers
├── chroma_db/ pdf_db/ csv_db/      # persistent vector DBs
├── data/                           # sample docs
└── tests/ test_parser.py           # basic test
```

## ⚙️ Quick Start

```bash
git clone <your-repo-url>
cd dataroom
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
export GOOGLE_API_KEY=YOUR_KEY   # Windows: set GOOGLE_API_KEY=...
python main.py
# visit http://127.0.0.1:7860
```

## 💬 Usage
1. Upload PDF or CSV (repeat to build corpus).
2. Click “List Current Files” to view ingested documents and statistics.
3. Ask focused questions (include fields / assets / amounts).
4. Citations appear at the end; if unsure the system states uncertainty.

### Example Queries
```
“Does the contract contain a rent escalation clause?”
“Which assets in the CSV have the highest NOI?”
“What lease term (start/end) is mentioned on page 10 of the first PDF?”
```

## 🧠 Key Design Decisions & Rationale

| Decision | Chosen | Main (Real) Motivation | Alternatives | Why Not Chosen | Future Direction |
|----------|--------|------------------------|--------------|----------------|------------------|
| Tool count | Single tool `real_estate_document_search` | Validate retrieval+citation loop only | Multiple tools (search/calc/format) | More orchestration complexity | Add structured calc / batch analysis tools later |
| Retrieval paradigm | Pure vector TopK (CLIP) | Lowest effort to prove viability | Hybrid (Vector+BM25) / keyword filter | Extra index & tuning | Introduce BM25 or custom inverted → score fusion |
| Store split | Separate PDF / CSV collections | Avoid noise; clearer stats | Single collection + type field | Filtering still splits; stats less direct | Add unified aggregation view |
| Chunk granularity | PDF=page / CSV=row | Stable citations; quick | Sentence / sliding / semantic | Higher cleanup & complexity | Add secondary sentence split for long pages |
| Embedding model | CLIP 512d | Existing dep; good for MVP | bge / jina / text-embedding-004 | Time to evaluate & swap | Replace with domain text model + rerank |
| Context assembly | Simple concat + separators | Fast prototype; readable | Structured tables / clustering | Higher time cost | Field grouped summaries (rent/area) |
| Citation format | `[file p.X]` / `[file row Y]` | Short & clear | Inline JSON / hyperlinks | Verbose or UI work | Merge duplicate refs + anchors |
| Agent framework | LangGraph loop | Explicit states (respond vs tool) | Direct single-shot LLM | Harder to extend | Add rerank / filter nodes |
| UI | Single page upload + chat | Reduce UI surface; focus validation | Multi-tab / session mgmt | Out of scope | Add session save & citation replay |

> Reason for deferring Hybrid / rerank / secondary splitting now: 
> (1) Need more real doc diversity (contracts, statutes, rent schedules, ops reports) to profile template variance; 
> (2) Need a minimal evaluation set (20~50 labeled Q/A) to avoid blind tuning; 
> (3) Multiple strategies together require abstraction of a retrieval strategy layer & fusion weighting, adding short‑term complexity. **Hence: lock a working end‑to‑end baseline first, then iterate.**

## 🧗 Key Challenges & Reflections

| Challenge | Current Handling | Pain / Impact | Next Idea | Priority |
|-----------|-----------------|---------------|-----------|----------|
| Low accuracy (synonyms / long pages) | Basic vector TopK; page granularity | Missed recall / noise pages | Hybrid + sentence split + local rerank | High |
| Repetition & templated clauses | None yet | Subtle differences lost | Clause number regex + weighted fields | High |
| Single tool extensibility | All needs in one RAG | Prompt inflation later | Split tools (retrieval / calc / summary) | Medium |
| Coarse granularity | Quick build | Hard fine location; citation coarse | Secondary split (page→sentence; row→fields) | Medium |
| No rerank | None | Unstable TopK order | Add cross-encoder / miniLM reranker | Medium |
| Embedding generality | CLIP used | Domain nuance weak | Evaluate domain embeddings + cache migration | Medium |
| File listing scalability | Full scan get() | O(N) latency at scale | Maintain document_manifest index | Low |

> Current main bottleneck: lack of hybrid retrieval + finer-grain context leads to weaker performance on “semantic + precise field” questions. **Value delivered: traceable citations & basic cross-document Q&A loop.**


## 📄 License
MIT License

## 📬 Submission Alignment
1. Included: code + setup guide (see Quick Start).
2. Architecture, design decisions & challenges: see respective sections.

