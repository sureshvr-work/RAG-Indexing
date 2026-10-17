# RAG-Indexing

A local Vector Search Architecture Lab comparing FAISS, PostgreSQL with pgvector, and Qdrant.

## Current setup

Dataset preparation, local embeddings, ingestion, exact filtered search, a local
FastAPI/Streamlit dashboard, and temporary HNSW tuning experiments are ready across
FAISS, PostgreSQL with pgvector, and Qdrant.

Create your Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

With Docker Desktop running, copy the local settings and choose a password in `.env`:

```bash
cp .env.example .env
```

Start the databases and inspect their status:

```bash
docker compose up -d --wait
docker compose ps
```

Verify pgvector is enabled and Qdrant is ready:

```bash
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\\dx vector"'
curl --fail http://127.0.0.1:6333/readyz
```

PostgreSQL is available on `127.0.0.1:5432`. Qdrant's REST API and dashboard
are available at `http://127.0.0.1:6333` and `http://127.0.0.1:6333/dashboard`.
If a port is already occupied, change its value in `.env` before starting.
Compose waits for PostgreSQL's health check; Qdrant readiness is checked separately
with `/readyz` above.

Stop the services with `docker compose down`. Named volumes retain database data.
The PostgreSQL initialization script enables pgvector only when the data volume is
first created. Changing PostgreSQL credentials in `.env` later does not update an
existing database. FAISS runs in Python and does not need a container.

Images are pinned to pgvector `0.8.7` with PostgreSQL `17`, and Qdrant `1.19.2`.
Sources: [pgvector Docker instructions](https://github.com/pgvector/pgvector#docker)
and [Qdrant release](https://github.com/qdrant/qdrant/releases/tag/v1.19.2).

## Prepare the document sample

The downloaded BEIR SciFact corpus contains 5,183 scientific abstracts. Select
1,000 complete records with a fixed seed:

```bash
python scripts/prepare_dataset.py
```

Outputs (ignored by Git):

- `datasets/processed/scifact_1000/documents.jsonl`: selected records with stable
  numeric `id`, original string `document_id`, title, abstract, combined `content`,
  and original metadata.
- `datasets/processed/scifact_1000/manifest.json`: seed, corpus and sample counts,
  hashes, sampling method, source, and corpus licensing attribution.

Each embedding input will be the title followed by the abstract. There is no
chunking or invented category, publication year, or region. The model's input
length and truncation will be checked during embedding generation.

Sampling sorts source IDs before selecting records, so a fixed seed produces the
same sample even if the source file is reordered. Re-running replaces the two
sample files; do not change the sample after embedding without regenerating its
embeddings. Source documents remain unchanged. Optional arguments:

```bash
python scripts/prepare_dataset.py --count 1000 --seed 42
python -m pytest tests/test_prepare_dataset.py -q
```

The SciFact corpus license is documented as ODC-By 1.0 by the
[original authors](https://github.com/allenai/scifact/blob/master/LICENSE.md).
This step does not generate embeddings or insert records into databases.

## Generate local embeddings

Prepare the sample first, then run the embedding workflow:

```bash
python scripts/prepare_dataset.py
python scripts/generate_embeddings.py
```

`config/settings.yaml` pins `sentence-transformers/all-MiniLM-L6-v2` to an immutable
model revision and uses CPU inference with four threads and batches of 32. The
first embedding run downloads the free public model into `.cache/huggingface`.
No API credentials or paid inference services are used.

Outputs in `datasets/processed/scifact_1000/embeddings/` (ignored by Git):

- `vectors.npy`: a normalized float32 array with shape `(1000, 384)`.
- `ids.npy`: matching int64 document IDs. Vector row `i`, ID row `i`, and prepared
  document row `i` refer to the same source record.
- `token_lengths.json`: full input token counts and truncation flags per ID.
- `manifest.json`: model revision, configuration, document and artifact hashes,
  package versions, dimensions, normalization, and measured generation times.

The model's limit is 256 tokens including special tokens. Longer title + abstract
inputs are truncated only for encoding; the stored title and abstract remain
complete. The verified default seed-42 sample has 710 of 1,000 inputs above this
limit. These are one-vector-per-document experiments using the leading text of
longer inputs, rather than embeddings of every token in every abstract.

Existing matching artifacts are validated and reused without loading the model:

```bash
python scripts/generate_embeddings.py --offline
```

To generate vectors for a different sample or configuration, select a new output
directory using `--output-dir`. The script rejects mismatched or corrupt artifacts
and publishes the local output directory only after the complete set validates.
This protects local files from interrupted writes; backend index versioning will
be a separate later milestone. Identical inputs and configuration can still show
small floating-point differences across hardware or library versions, so backend
comparisons should reuse the saved vectors rather than regenerate them.

The vector array alone uses approximately 1.46 MiB. Model weights and inference
buffers require additional memory. Load the saved arrays without pickle:

```python
import numpy as np

base = "datasets/processed/scifact_1000/embeddings"
vectors = np.load(f"{base}/vectors.npy", allow_pickle=False)
ids = np.load(f"{base}/ids.npy", allow_pickle=False)
print(vectors.shape, ids.shape)  # (1000, 384), (1000,)
```

Run the integrity tests with `python -m pytest tests -q`. The full workflow has also
been verified with the real model and cached weights. The next commands load the
saved vectors into the search backends without regenerating them.

Model references: [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and [Sentence Transformers API](https://sbert.net/docs/package_reference/sentence_transformer/model.html).

## Ingest and search the three backends

With your virtual environment activated and Docker services running:

```bash
python scripts/ingest.py
python scripts/search.py "cancer immunotherapy" --top-k 5
python scripts/search.py "brain magnetic resonance imaging" --top-k 5 --json
```

Ingestion validates the saved artifacts, then loads the same 1,000 stable IDs,
384-dimensional vectors, and complete document records into all three backends.
Repeating ingestion updates existing IDs without duplicating documents. Search
embeds the query once with the same pinned model, shares its vector across the
backends, and returns cosine similarity scores (higher means more similar).
Search uses cached model weights offline; generate the embeddings first to populate
the cache. Full JSON results include source text, metadata, query truncation status,
and measured timings.

Choose a single backend when needed:

```bash
python scripts/search.py "antibiotic resistance" --backends pgvector --top-k 10
```

These are exact-search baselines:

- FAISS uses `IndexIDMap2(IndexFlatIP)` on normalized vectors, with documents stored
  separately on disk. A dedicated local Python process keeps FAISS's native runtime
  separate from PyTorch. The installed macOS wheels crashed when sharing a process;
  the related [upstream issue](https://github.com/pytorch/pytorch/issues/149201)
  describes the OpenMP runtime conflict.
- PostgreSQL uses a `vector(384)` column and exact cosine distance scans, without an
  approximate vector index. A primary key preserves the stable document IDs.
- Qdrant uses a cosine collection, document payloads, HNSW disabled (`m=0`), and
  explicit `exact=True` searches.

The artifact hashes and embedding configuration determine a shared namespace.
For the current sample it is `scifact_e7493f6f25bafca0aa4f587f01bfa003`: a PostgreSQL
table, Qdrant collection, and FAISS directory under `datasets/indexes/faiss/`.
New artifacts produce a different namespace, preventing different model profiles
or document samples from being mixed. This does not implement the planned staged
publication and rollback workflow. Ingestion is sequential across backends;
Qdrant batches become visible as they are written. Retry ingestion after an
interruption to finish the same dataset.

In DBeaver's SQL editor, inspect PostgreSQL with:

```sql
SELECT namespace, identity FROM public.vector_lab_datasets;

SELECT count(*)
FROM public.scifact_e7493f6f25bafca0aa4f587f01bfa003;

SELECT id, title, vector_dims(embedding) AS dimensions
FROM public.scifact_e7493f6f25bafca0aa4f587f01bfa003
ORDER BY id
LIMIT 5;
```

Refresh the schema after ingestion. In Qdrant's dashboard at
`http://127.0.0.1:6333/dashboard`, open the collection with the same namespace.
FAISS persists an index plus a document mapping and integrity manifest; it has
no database UI. Each save creates a new local snapshot and switches an active
pointer. Old snapshots are retained. Its reported storage bytes cover the active
snapshot only; PostgreSQL's reported bytes include its table, primary key, and
TOAST storage. Qdrant storage bytes are left unset. These values are not equivalent
cross-backend disk or memory measurements.

Verify exact retrieval against NumPy cosine calculations over the saved vectors:

```bash
python scripts/verify_stores.py
python -m pytest tests -q
RAG_LAB_INTEGRATION=1 python -m pytest tests -q
```

Verification checks IDs/counts, retrieved document payloads, and top-10 results for
three text queries plus three saved-document vectors. Equal-score cutoff ties and
small floating-point differences are allowed. Integration tests use disposable,
uniquely named PostgreSQL tables and Qdrant collections, and remove those test
resources afterward. Without the integration flag, database tests are skipped.

Each search-call timing includes client transport and result decoding; FAISS also
includes local process communication. Model loading and query embedding are outside
the backend search timing, and embedding time is reported separately. Individual
calls are not P50/P95 benchmark statistics.

The observed truncation of 710 inputs is an embedding-model selection concern:
input capacity and output dimensions are separate properties. A longer-context
model or token-aware chunking may preserve more source content, but needs semantic
retrieval evaluation and a new embedding run. Exact agreement between backends
checks search correctness over the saved vectors; it cannot recover the text that
was omitted during embedding or demonstrate relevance quality.

## Inspect the saved FAISS index

FAISS runs as a local library in this project. Inspect the active snapshot with:

```bash
python scripts/inspect_faiss.py
python scripts/inspect_faiss.py --id 18670
```

The first command displays the actual index type, dimensions, vector count,
readiness, file size, and sample IDs, titles, metadata, and vector previews.
The second shows one complete document and its full 384-dimensional vector.
This read-only command checks the saved file hashes and ID mapping, and runs
without loading an embedding model or connecting to Docker databases.

Under `datasets/indexes/faiss/<namespace>/`, `active.json` names the current
snapshot folder. Inside that folder, `index.faiss` is the binary vector index,
`documents.json` is the readable document/metadata mapping, and `manifest.json`
records the identity and integrity hashes. Open the JSON files in your editor;
inspect the binary index through the Python command above. `is_trained=True` on
this Flat index means it is ready to accept vectors; no training was performed.
If multiple dataset namespaces exist, select one using `--namespace`.

## Metadata update and filter-test data

`metadata` contains the original source metadata plus any explicitly added lab
fields. These synthetic fields are unrelated to the papers' scientific content
or actual publication dates:

- `lab_metadata_source`: `synthetic_filter_test`.
- `lab_group`: `alpha`, `beta`, `gamma`, or `delta`.
- `lab_year`: integer values from 2022 through 2025 (test values).
- `lab_review_status`: `pending`, `reviewed`, or `approved`.

Generate deterministic labels for all documents, or patch one document:

```bash
python scripts/update_metadata.py --generate-test-data
python scripts/update_metadata.py --id 18670 --patch '{"lab_review_status":"approved"}'
```

The command applies updates to all three backends. PostgreSQL merges JSONB fields;
Qdrant merges fields inside the point's `metadata` payload; FAISS updates the local
document mapping and persists a snapshot. The vector contents stay unchanged.
Patches merge top-level metadata keys; replacing a nested object replaces that
whole value. This command does not remove keys or modify titles/abstracts.

Desired updates are saved atomically in
`datasets/processed/scifact_1000/metadata_updates.json`, ignored by Git. The original
`documents.jsonl`, embedding manifests, vectors, and dataset namespace stay intact.
Ingestion, search, and verification load this overlay, so re-ingestion preserves
updates and verification compares against the updated documents. The command
checks all backend ID sets before saving the overlay, then updates the backends
sequentially. Interrupted updates can leave temporarily different metadata; retry:

```bash
python scripts/update_metadata.py --apply-saved
```

Generating test data again with the same seed reproduces the same lab labels;
it resets these lab fields while preserving other metadata keys. The current
1,000-document sample has groups alpha=259, beta=248, gamma=228, delta=265.

Inspect or filter these values directly in PostgreSQL:

```sql
SELECT metadata->>'lab_group' AS lab_group, count(*)
FROM public.scifact_e7493f6f25bafca0aa4f587f01bfa003
GROUP BY metadata->>'lab_group'
ORDER BY lab_group;

SELECT id, title, metadata
FROM public.scifact_e7493f6f25bafca0aa4f587f01bfa003
WHERE metadata->>'lab_group' = 'alpha'
  AND (metadata->>'lab_year')::int >= 2024
ORDER BY id;
```

Qdrant stores these fields under payload paths such as `metadata.lab_group`.
The search CLI currently performs unfiltered vector retrieval. The dashboard/API
support shared exact-search filters for these fields. Metadata index experiments
and filtered ANN tuning are later steps.

## Index creation in the current implementation

`ingest_bundle()` calls `create_index()` before `insert()`. This shared method
initializes or validates the backend's storage; it does not imply that every
backend has an approximate vector index.

- FAISS constructs an empty `IndexIDMap2(IndexFlatIP(dimensions))`, then immediately
  adds vectors and IDs. Flat search needs no training. Reopening loads the persisted
  index and document mapping.
- PostgreSQL creates the table and its automatic B-tree primary-key index on `id`.
  It has no HNSW or IVFFlat vector index; vector searches scan the rows exactly.
- Qdrant creates the cosine collection with `m=0` to disable HNSW construction.
  Searches request `exact=True`. No payload indexes have been created.

Embedding model settings and dimensions come from `config/settings.yaml`.
The index algorithms and exact-search settings currently live in the adapter code:
`app/stores/_faiss_native.py`, `pgvector_store.py`, and `qdrant_store.py`.
The `lab_ui` YAML section sets default search and HNSW experiment parameters.
The UI can build temporary HNSW experiments on all three backends. Changing those
defaults does not rebuild or replace the persisted exact stores. Additional index
algorithms and persistent index configuration remain later steps. Vector indexes accelerate similarity search; metadata indexes accelerate
filter conditions and are configured separately.

## Run the local dashboard

Activate your environment, keep Docker Desktop running, and start both application
services with one command:

```bash
source .venv/bin/activate
docker compose up -d --wait
python scripts/run_ui.py
```

Open **http://127.0.0.1:8501**. API documentation is at
**http://127.0.0.1:8000/docs**. The launcher waits for API readiness and stops both
services on Ctrl+C. Both bind to localhost. Existing processes on these ports are
left untouched; choose different ports if needed:

```bash
python scripts/run_ui.py --api-port 8001 --ui-port 8502
```

The dashboard provides four working pages:

- **Overview:** actual backend readiness and index details, dataset/model facts,
  and synthetic metadata distributions.
- **Search playground:** embed a query once, select backends and Top K, apply
  group/status/year filters, and inspect results side by side. Each successful
  result is checked against NumPy exact cosine search over the filtered dataset.
  A failed backend is reported while other selected backends can still return.
- **Data explorer:** browse the prepared dataset and saved metadata updates, page
  through matching records, and inspect a document/vector from the prepared files
  or any actual backend. It shows full abstracts, metadata, token truncation,
  vector norms and values, and allows JSON downloads.
- **Index tuning:** choose a backend, query, Top K, repetitions, M,
  efConstruction, and efSearch. Confirm the temporary build, then inspect real
  build time, recorded latency samples, P50/P95, Recall@K, and session history.
  History can be downloaded as JSON; it is otherwise scoped to the browser session.

Groups and review statuses support OR within each field, with AND across fields.
The optional year range is inclusive. FAISS filters the document mapping before
ranking a temporary Flat index of eligible vectors; PostgreSQL applies SQL
conditions before exact ranking; Qdrant uses native payload filters. No metadata
indexes have been added to the baseline stores.

Tuning uses all 1,000 saved vectors without filters. FAISS builds an in-memory
HNSW index inside its isolated worker using one thread. PostgreSQL creates a
uniquely named temporary experiment table and HNSW index, checks the query plan
actually uses it, and removes the table/registry row afterward. Qdrant creates a
uniquely named collection, waits for its vectors to be indexed and the collection
to be green, searches with HNSW enabled, and deletes the collection afterward.
Cleanup also runs on handled errors. A hard process termination can leave remote
experiment resources with names beginning `experiment_`; it does not replace the
baseline namespace. Builds are serialized within one API process.

The exact persisted indexes and embedding files stay intact. The tuning sliders
control actual HNSW experiments; IVF/PQ, filtered ANN, and persistent index
publication are not implemented in this dashboard version. Default UI settings
come from the `lab_ui` section of `config/settings.yaml`, loaded by the API;
restart the application after changing the defaults.

Query timings include client transport and result decoding, including FAISS
process communication. Embedding/model loading and experiment setup are excluded.
FAISS build time measures vector addition, PostgreSQL measures CREATE INDEX, and
Qdrant measures point upload through indexed readiness; these scopes differ.
One warmup query is excluded, and the UI reports only the recorded observations.
With 2–20 repeats, percentiles are exploratory. Recall@K compares IDs with exact
neighbors over saved vectors, not scientific relevance; equivalent cutoff neighbors are accepted by the API
metrics within the documented tolerance. CPU, disk, and memory are shared on your Mac;
these experiments provide no resource isolation.

To run the services separately:

```bash
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
# In another terminal, with the environment activated:
python -m streamlit run app/ui/dashboard.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

The Streamlit HTTP client uses `RAG_LAB_API_URL` when set, otherwise
`http://127.0.0.1:8000`. The API owns model inference and backend operations;
Streamlit renders the returned data and controls.

## Search flow diagram

Open `docs/search-flow.html` directly in a browser, or use **View search flow
diagram** in the dashboard sidebar. With the API running it is available at
**http://127.0.0.1:8000/search-flow**. The page is self-contained and works offline;
it includes both entry routes, backend-specific search calls, common results,
route highlighting, light/dark themes, and a print view.

## Per-search precision and recall

The Search playground and Index tuning pages display two explicitly separate
reference sets for every result:

- **Labeled relevance:** SciFact's positive query/document relevance judgments.
  Select **SciFact labeled query**, choose `test` or `train`, and select a saved
  query. The selector includes queries with at least one relevant document in our
  sample. The current sample covers 58 labeled queries from the test split.
- **Exact-neighbor agreement:** NumPy cosine top-K over the searchable vectors.
  This measures index accuracy and is available for free-text queries as well.
  It does not measure whether documents are scientifically relevant.

For both references, Precision@K is matching reference documents among the first
K results divided by requested K. Missing result slots count against precision.
Relevance Recall@K divides matching relevant documents by the positive judgments
available in the 1,000-document sample after metadata filters. The UI reports
relevant documents excluded by sampling or filters; this is not full-corpus recall.
Neighbor recall divides matching exact neighbors by min(K, eligible document count),
accepting equivalent cutoff scores within a tolerance of 1e-5. Empty reference sets
are N/A, and a known reference set with zero retrieved matches scores zero.

Free-text queries have no relevance scores unless their text exactly matches a
labeled query in the selected split (surrounding whitespace is ignored). Unknown
queries show N/A with a reason. Changing a selected query's text cannot reuse its
query ID's labels. Positive qrels are treated as relevant and other results as
non-relevant for scoring; no new relevance labels are generated.

The CLI includes both metrics in text and JSON output:

```bash
python scripts/search.py "ADAR1 binds to Dicer to cleave pre-miRNA." --query-id 49 --qrels-split test --top-k 5
python scripts/search.py "cancer immunotherapy" --top-k 5 --json
```

In a verified query-49 run, each exact backend retrieved the one available judged
relevant document among five results: relevance Precision@5=20% and Recall@5=100%,
while exact-neighbor precision and recall were both 100%. API responses and JSON
downloads include the reference scope, denominators, matching/returned counts, and
availability reasons. The API query catalog is available at
`GET /evaluation/queries?split=test`.

Relevance-label format and evaluation conventions:
[BEIR evaluation example](https://github.com/beir-cellar/beir#-quick-example).

## Project specification

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
