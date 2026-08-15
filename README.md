# VectorDB

A lightweight vector database implemented from scratch using
HNSW-based approximate nearest-neighbor search.

## Why I Built This

Instead of using an existing vector database such as Chroma,
FAISS, or Pinecone, this project implements the core storage,
indexing, retrieval, persistence, and lifecycle mechanisms directly.


## Roadmap

### Basic V1 — Complete

- [x] Vector storage
- [x] Cosine similarity search
- [x] HNSW approximate nearest-neighbor indexing
- [x] CRUD operations
- [x] Metadata filtering
- [x] Logical document ID ↔ HNSW index ID mapping
- [x] Update-safe index ID allocation
- [x] Per-document vector validation
- [x] Partial-success batch ingestion
- [x] Persistent database state
- [x] Soft deletion
- [x] HNSW index rebuild / compaction
- [x] PDF ingestion pipeline
- [x] Automated test suite
- [x] Performance benchmarking

### Advanced V1 — In Progress

- [ ] Metadata indexes for faster filter evaluation
- [ ] Pre-filtering during ANN traversal
- [ ] Efficient filtered top-k retrieval
- [ ] Filter-aware candidate selection
- [ ] Query planning for filtered search
- [ ] Recall evaluation for ANN search
- [ ] Search performance evaluation across different `ef` values

### Advanced V2 — Planned

- [ ] Hybrid lexical + vector retrieval
- [ ] BM25 + ANN retrieval
- [ ] Reranking
- [ ] Improved persistence layer
- [ ] Larger-scale benchmarks
- [ ] Memory and storage optimization
- [ ] Concurrent query support

### Future

- [ ] WAL / crash recovery
- [ ] Configurable index capacity
- [ ] Sharding
- [ ] Distributed indexing
- [ ] API server
- [ ] Authentication and multi-user isolation
- [ ] ## Architecture

PDF / Text
    ↓
Chunking
    ↓
Sentence Transformer
    ↓
VectorDB
    ├── Document Storage
    ├── Metadata Storage
    ├── HNSW Index
    ├── ID Mapping
    └── Persistence
         ↓
      Top-K Search

## Features

- HNSW approximate nearest-neighbor search
- Cosine similarity
- CRUD operations
- Metadata filtering
- Document ↔ HNSW ID mapping
- Individual vector validation
- Partial-success batch ingestion
- Bulk vector insertion
- Persistent database state
- Index reconstruction after reload
- Soft deletion
- Index rebuild / compaction
- Search edge-case handling
- PDF ingestion pipeline


## Performance

Benchmarked locally on 384-dimensional vectors using HNSW
with `M=16`, `ef=50`, and `k=10`.

| Dataset | Insert Throughput | Avg Search | P95 Search | P99 Search |
|--------:|------------------:|-----------:|-----------:|-----------:|
| 1K      | 19,118 vec/s      | 0.166 ms   | 0.231 ms   | 0.429 ms   |
| 5K      | 5,374 vec/s       | 0.599 ms   | 0.912 ms   | 1.214 ms   |
| 10K     | 3,123 vec/s       | 1.226 ms   | 2.350 ms   | 2.628 ms   |

Benchmark configuration:

- Vector dimension: 384
- Top-k: 10
- Queries: 100
- HNSW `M`: 16
- HNSW `ef`: 50

Search latency increases as the index grows, while insertion
throughput decreases due to the additional HNSW graph construction
work.
## Core Design

### Logical vs Physical IDs

The database maintains two identifiers:

doc_id → logical document storage

hnsw_id → physical ANN index label

Mappings:

doc_id → hnsw_id
hnsw_id → doc_id

This allows updates and index rebuilds without
changing logical document identity.

## Batch Ingestion

Each document is independently validated.

Invalid records are rejected while valid records
continue through the batch.

Example:

4 records
→ 3 valid
→ 1 rejected
→ 3 vectors inserted

## Persistence

The database persists:

- documents
- vectors
- metadata
- HNSW ID mappings
- next index ID
- index configuration

The HNSW index is reconstructed during loading.

## Index Compaction

HNSW deletion is handled as a soft deletion.

Therefore deleted entries can remain in the physical index.

`rebuild_index()` reconstructs the index using only
active vectors and rebuilds the ID mappings.



## Testing

VectorDB currently has 18 automated tests covering:

- document insertion and validation
- vector validation
- batch ingestion
- ANN search
- metadata filtering
- updates
- deletion
- persistence
- index rebuilding / compaction

```text
18 passed in 0.17s
## Example

```python
db = VectorDB()

db.add(
    "Python is used for data engineering.",
    vector,
    {"source": "notes.pdf"}
)

results = db.search(
    query_vector,
    k=5
)

for document, similarity, metadata in results:
    print(similarity, document, metadata)
