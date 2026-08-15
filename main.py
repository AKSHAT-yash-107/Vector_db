from vectordb import VectorDB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

db = VectorDB()
db.load("vector_db.json")

print("\n1. INITIAL SEARCH")

query_vector = model.encode(
    "How does Spark process data?"
)

results = db.search(
    query_vector,
    k=3
)

print("Results:", len(results))


print("\n2. UPDATE")

active_doc_id = next(iter(db.index_ids))

old_index_id = db.index_ids[active_doc_id]

db.update(
    active_doc_id,
    "UPDATED DOCUMENT: Spark performs distributed data processing.",
    model.encode(
        "UPDATED DOCUMENT: Spark performs distributed data processing."
    ),
    {"source": "lifecycle_test", "page": 1}
)

new_index_id = db.index_ids[active_doc_id]

print("Document ID:", active_doc_id)
print("Old HNSW ID:", old_index_id)
print("New HNSW ID:", new_index_id)


print("\n3. SEARCH AFTER UPDATE")

results = db.search(
    model.encode("Spark distributed data processing"),
    k=3
)

print("Results:", len(results))


print("\n4. DELETE")

delete_doc_id = active_doc_id

db.delete(delete_doc_id)

print(
    "Deleted:",
    db.document[delete_doc_id] is None
)


print("\n5. REBUILD")

rebuild_result = db.rebuild_index()

print(rebuild_result)


print("\n6. SEARCH AFTER REBUILD")

results = db.search(
    model.encode("Spark distributed data processing"),
    k=3
)

print("Results:", len(results))


print("\n7. SAVE")

db.save("lifecycle_test.json")


print("\n8. RELOAD")

new_db = VectorDB()
new_db.load("lifecycle_test.json")

print(
    "Documents:",
    len(new_db.document)
)

print(
    "HNSW count:",
    new_db.index.get_current_count()
)

print(
    "Mappings:",
    len(new_db.index_ids),
    len(new_db.hnsw_to_doc)
)


print("\n9. FINAL SEARCH")

results = new_db.search(
    query_vector,
    k=3
)

print("Results:", len(results))

print("\n===== BASIC V1 LIFECYCLE COMPLETE =====")