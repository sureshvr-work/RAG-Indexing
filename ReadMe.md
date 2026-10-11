# RAG-Indexing

A local Vector Search Architecture Lab comparing FAISS, PostgreSQL with pgvector, and Qdrant.

Below is a project specification you can copy directly into Codex in VS Code. I've structured it as an implementation prompt with architecture, features, milestones, acceptance criteria, and cost constraints.
The goal is to build a working learning environment, not just three independent demos.
Project: Vector Search Architecture Lab
FAISS vs PostgreSQL + pgvector vs Qdrant
Copy the following prompt into Codex
1. Project Objective
Build a local, hands-on Vector Search Architecture Lab to compare FAISS, PostgreSQL with pgvector, and Qdrant.
The application should help me understand how each technology handles vector storage, indexing, retrieval, metadata filtering, hybrid search, and scaling.
The project must be educational, allowing me to modify indexing parameters, execute searches, visualize results, and measure performance differences.
Primary goals:
- Understand internal indexing algorithms: Flat, HNSW, IVF, and PQ.
- Compare the same dataset and embeddings across three technologies.
- Experiment with cosine similarity, L2 distance, and normalization.
- Understand dense and sparse vector representations.
- Compare metadata filtering implementations.
- Explore Qdrant collections, shards, segments, and ACORN.
- Measure Recall@K, P50/P95 latency, indexing time, and memory consumption.
- Understand production trade-offs through controlled benchmarks.
The application must be fully functional locally on an Apple Silicon Mac without requiring cloud services or paid APIs.
2. Technology Stack
Layer	Technology
Backend	Python 3.11+
API	FastAPI
Frontend	Streamlit
Vector search library	FAISS CPU
Relational database	PostgreSQL + pgvector
Vector database	Qdrant
Embedding model	Sentence Transformers
Default model	all-MiniLM-L6-v2
Data processing	NumPy, Pandas
Benchmarking	Python, pytest
Visualization	Streamlit + Plotly
Infrastructure	Docker Compose
Configuration	YAML + environment variables


Use Python type hints, modular services, structured logging, and clear error handling.
Avoid unnecessary frameworks such as LangChain initially. I want to understand the actual APIs and indexing behavior without framework abstractions.
3. System Architecture
                         STREAMLIT UI
                              |
                              v
                          FASTAPI
                              |
               +--------------+--------------+
               |              |              |
               v              v              v
           Ingestion      Search API     Benchmark API
               |              |              |
               +--------------+--------------+
                              |
                       Vector Service
                              |
                 +------------+------------+
                 |            |            |
                 v            v            v
               FAISS       PostgreSQL    Qdrant
                 |          pgvector       |
                 |            |            |
                 +------------+------------+
                              |
                       Results Service
                              |
                   +----------+----------+
                   |                     |
                   v                     v
               Metrics              Visualization

Implement a common interface so all three search engines can execute equivalent operations.
Suggested interface:
class VectorStore:    def create_index(self, config): ...    def insert(self, documents, embeddings): ...    def search(self, query_vector, top_k, filters=None): ...    def delete(self, ids): ...    def get_stats(self): ...


Use appropriate abstract base classes or protocols and define typed request/response models. Allow index creation and collection initialization to differ internally where the backends require it.
Each implementation should be isolated:
- FaissVectorStore
- PgVectorStore
- QdrantVectorStore
The UI should not need technology-specific logic to execute a basic vector search.
4. Project Structure
Create this structure, adjusting where necessary for a maintainable implementation.
vector-search-lab/
│
├── README.md
├── docker-compose.yml
├── .env.example
├── pyproject.toml
├── Makefile
│
├── config/
│   ├── settings.yaml
│   └── benchmark.yaml
│
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── ingestion.py
│   │   ├── search.py
│   │   └── benchmark.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── models.py
│   │   └── interfaces.py
│   │
│   ├── embeddings/
│   │   └── embedding_service.py
│   │
│   ├── stores/
│   │   ├── faiss_store.py
│   │   ├── pgvector_store.py
│   │   └── qdrant_store.py
│   │
│   ├── services/
│   │   ├── ingestion_service.py
│   │   ├── retrieval_service.py
│   │   └── benchmark_service.py
│   │
│   └── evaluation/
│       ├── recall.py
│       ├── latency.py
│       └── metrics.py
│
├── ui/
│   └── streamlit_app.py
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── scripts/
│   ├── generate_dataset.py
│   ├── generate_embeddings.py
│   ├── initialize_stores.py
│   ├── run_benchmarks.py
│   └── cleanup.py
│
├── tests/
│   ├── test_embeddings.py
│   ├── test_faiss.py
│   ├── test_pgvector.py
│   ├── test_qdrant.py
│   └── test_recall.py
│
└── results/
    ├── benchmarks/
    └── charts/

5. Dataset and Embedding Requirements
Support two dataset types.
A. Real text dataset
Start with 1,000–10,000 short documents or research-style text chunks.
Prefer a freely available dataset that can be downloaded without authentication. Include the source and licensing information.
Each record should contain:
{
  "id": 101,
  "document_id": "DOC-001",
  "chunk_id": 1,
  "content": "Research about cancer immunotherapy...",
  "category": "Oncology",
  "publication_year": 2026,
  "region": "USA"
}

Use the local Sentence Transformers model to create embeddings.
Generate embeddings once and reuse the exact same vectors for FAISS, pgvector, and Qdrant.
Persist the dataset, IDs, and embeddings locally so benchmarks are reproducible.
B. Synthetic vector dataset
Generate normalized synthetic vectors for performance testing.
Supported sizes:
- 1,000
- 10,000
- 100,000
- 500,000
- 1,000,000
The default should be 10,000.
Use deterministic random seeds and provide memory estimates before generating large datasets. Warn or refuse when an experiment exceeds a configurable resource budget.
Synthetic vectors are for performance experiments, not semantic retrieval-quality evaluation.
6. FAISS Implementation
Implement the following index types:
- IndexFlatL2
- IndexFlatIP
- IndexHNSWFlat
- IndexIVFFlat
- IndexIVFPQ
Expose configuration options:
faiss:
  index_type: HNSW
  metric: cosine
  hnsw:
    m: 16
    ef_construction: 100
    ef_search: 64
  ivf:
    nlist: 100
    nprobe: 10
  pq:
    subquantizers: 24
    bits_per_code: 8

Validate index compatibility, dimensions, and training requirements before building an index.
For cosine search, normalize vectors and use inner-product search, preserving the distinction between cosine similarity and L2 distance.
Support saving and loading indexes.
Maintain a separate mapping between FAISS IDs, source documents, and metadata. Use stable IDs so retrieval results are comparable across backends.
Clearly display FAISS limitations around metadata filtering, persistence management, transactions, and distribution.
7. PostgreSQL + pgvector Implementation
Use PostgreSQL in Docker.
Implement:
- Exact vector search.
- HNSW indexing.
- IVFFlat indexing.
- Dense vector storage.
- Metadata filtering through SQL.
- PostgreSQL full-text search using tsvector.
- Optional true sparse embeddings using sparsevec, subject to dimensional limits.
Example table:
CREATE TABLE research_chunks (
    id BIGINT PRIMARY KEY,
    document_id TEXT,
    content TEXT,
    embedding vector(384),
    search_vector tsvector,
    category TEXT,
    publication_year INT,
    region TEXT
);

Support configuring:
- HNSW m.
- HNSW ef_construction.
- HNSW ef_search.
- IVFFlat lists.
- IVFFlat probes.
- Iterative HNSW scans.
Provide an interface to execute:
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)

Display:
- Whether the query uses an index scan or sequential scan.
- Execution time.
- Shared buffer hits.
- Shared buffer reads.
- Index size.
- Table size.
Explain that buffer reads don't necessarily correspond one-to-one with physical disk reads.
Allow users to compare filtered and unfiltered search.
8. Qdrant Implementation
Deploy Qdrant locally through Docker Compose.
Implement:
- Collection creation.
- Point insertion and deletion.
- Dense vectors.
- Sparse vectors.
- Named vectors.
- Payload storage.
- Payload indexes.
- Exact search.
- HNSW search.
- Metadata filtering.
- Hybrid retrieval using RRF.
- Quantization experiments where supported.
Expose relevant settings:
qdrant:
  vector_size: 384
  distance: Cosine
  hnsw:
    m: 16
    ef_construct: 100
  payload_indexes:
    - category
    - publication_year
    - region

Provide an educational view explaining:
Collection
   |
   +-- Shards
         |
         +-- Segments
               |
               +-- Points
               +-- Vector Index
               +-- Payload Index

Use public Qdrant APIs to inspect available collection, shard, and optimizer information. Do not depend on undocumented internal segment interfaces or claim to display information unavailable through supported APIs.
If the installed Qdrant version supports ACORN, provide an experiment comparing filtered HNSW retrieval with ACORN-enabled traversal.
Clearly distinguish filterable HNSW graph construction from ACORN query-time traversal.
Include memory and performance warnings for quantization and indexing changes.
9. Search and Evaluation
Create a common search endpoint:
POST /api/search

Request:
{
  "backend": "qdrant",
  "query": "Latest research on cancer immunotherapy",
  "top_k": 10,
  "filters": {
    "category": "Oncology",
    "publication_year": 2026
  }
}

Return:
- Matching document IDs.
- Content.
- Metadata.
- Similarity score or distance.
- Search latency.
- Index type.
- Backend name.
Do not assume raw similarity scores are directly comparable across backends or metrics.
Implement exact nearest-neighbor search as the evaluation baseline.
Calculate:
- Recall@1
- Recall@5
- Recall@10
- P50 latency
- P95 latency
- P99 latency
- Queries per second
- Index construction time
- Storage size
- Process/container memory usage
Use the same query set, embedding configuration, distance metric, filter conditions, and hardware environment for comparisons.
Run warm-up queries before measurements.
Report indexing and query performance separately.
10. Hybrid Search
Implement a hybrid retrieval experiment across all three systems.
For Qdrant, use named dense and sparse vectors and native fusion.
For PostgreSQL, start with pgvector dense search plus full-text search using tsvector, and combine rankings using RRF.
For FAISS, combine FAISS dense retrieval with a separate lexical retrieval implementation such as BM25.
Optionally add true model-generated sparse embeddings to PostgreSQL and Qdrant in a later milestone.
Do not treat BM25, PostgreSQL FTS, and learned sparse embeddings as identical algorithms. Label which approach each backend is using, since a hybrid benchmark is only fair when its retrieval methods are comparable.
11. Streamlit Dashboard
Create a practical dashboard with six sections.
Section	Features
Overview	Dataset size, embedding dimensions, backend status
Data Explorer	Inspect chunks, vectors, and metadata
Index Explorer	Create indexes and modify supported settings
Search Playground	Run the same query against selected backends
Benchmark Lab	Compare latency, recall, index sizes, and memory
Architecture Explorer	Explain and inspect available backend internals


The Search Playground should display results side by side.
The Benchmark Lab should use Plotly charts to show:
- Recall vs latency.
- P95 latency by backend.
- Index build time.
- Storage/memory consumption.
- Impact of metadata-filter selectivity.
Display measured values only. Never populate charts with fabricated benchmark results.
For expensive or destructive operations, such as rebuilding an index or loading one million vectors, require explicit UI confirmation.
12. Index Lifecycle and Ingestion Isolation
This is an important architectural requirement.
I want to explore how each technology handles new data and index maintenance without degrading the active search workload.
Implement two modes.
Mode A — Incremental ingestion
Insert new vectors into the existing searchable dataset and measure the effect on:
- Query latency.
- Recall.
- CPU.
- Memory.
- Indexing time.
Mode B — Build and publish
Create a complete new index or dataset version separately from the active searchable version.
The workflow should be:
                  Active Search
                       |
                       v
                 Index Version A
                       |
                       | Serves queries
                       |
New Data --> Staging --> Index Version B
                             |
                         Build Index
                             |
                          Validate
                             |
                       Atomic Publish
                             |
                             v
                       Version B Active

The staged version must not become searchable until the full ingestion, indexing, and validation process is complete.
Implement an application-level active-version pointer or appropriate backend-specific alias/swap mechanism.
Ensure concurrent readers see a consistent active version during cutover.
Support rollback to the previous version.
Important: Logical isolation does not guarantee CPU, memory, or I/O isolation when both versions run on the same machine. Record resource contention separately and explain what additional deployment isolation would be required in production.
This feature will help demonstrate blue-green indexing and zero-partial-publication patterns.
13. Deliverables
Build the project in progressive milestones.
Milestone 1 — Working foundation
- Docker Compose environment.
- FAISS, PostgreSQL, and Qdrant connections.
- Shared 1,000-document dataset.
- Local embedding generation.
- Ingest and search across all three backends.
- Basic Streamlit UI.
- README with setup commands.
Milestone 2 — Indexing experiments
- FAISS Flat, HNSW, IVF, and IVF-PQ.
- PostgreSQL HNSW and IVFFlat.
- Qdrant HNSW.
- Index configuration UI.
- Exact-search ground truth.
- Recall and latency comparisons.
Milestone 3 — Filtering and hybrid retrieval
- Metadata filtering across backends.
- Filter-selectivity experiments.
- Dense + lexical hybrid retrieval.
- RRF comparison.
- Qdrant payload indexing and supported ACORN configuration.
Milestone 4 — Performance and scalability
- 100K to 1M vector experiments.
- Resource monitoring.
- Benchmark reports.
- Performance tuning guidance.
Milestone 5 — Production architecture
- Index versioning.
- Staging and publication.
- Rollback.
- Ingestion/search interference testing.
- Architecture comparison report.
14. Acceptance Criteria
The POC is complete when:
1. All three backends can search the same embeddings and retrieve consistent IDs for exact search.
2. The UI can switch between FAISS, pgvector, and Qdrant.
3. Index parameters can be changed through supported operations.
4. Search results expose content, metadata, scores, and timing.
5. Benchmarks produce real, repeatable metrics.
6. Recall is calculated against an exact baseline using the same distance metric and eligible candidate set.
7. Metadata filtering works across all three backends.
8. Hybrid retrieval can be demonstrated, with implementation differences documented.
9. Index storage and memory use can be inspected or measured.
10. Staged index publication prevents partial datasets from becoming visible.
11. Automated tests cover ingestion, retrieval, filtering, recall, and index publication.
12. The entire project runs locally without cloud credentials or paid services.
15. Cost, Security, and Development Constraints
- Cloud cost target: $0.
- Use CPU-only FAISS by default.
- Do not call OpenAI, Azure OpenAI, or other paid embedding APIs.
- Download the embedding model once and cache it locally.
- Use Docker volumes for database persistence.
- Bind development services to localhost.
- Do not commit credentials, embedding caches, downloaded datasets, or large index files.
- Include a .gitignore and .env.example.
- Provide commands to start, stop, reset, and remove the environment.
- Make large benchmarks opt-in.
- Detect or document resource limitations on Apple Silicon.
- Use maintained, compatible library versions; avoid unverified or deprecated APIs.
16. Instructions to Codex
Before coding, review this specification and create a short implementation plan.
Then:
1. Implement Milestone 1 first.
2. Create the project files directly in the current VS Code workspace.
3. Verify installation and Docker service health.
4. Run the ingestion workflow.
5. Execute the same search against all three backends.
6. Run the automated tests.
7. Fix failures before progressing.
8. Document the commands and architectural decisions.
9. Summarize what is working and what remains for Milestone 2.
Do not attempt all five milestones in a single pass.
Prioritize a working, testable architecture over unnecessary abstractions.
For Milestone 1, I expect a fully runnable application, not pseudocode or empty placeholder implementations.