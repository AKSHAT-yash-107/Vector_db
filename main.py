from sentence_transformers import SentenceTransformer

from vectordb import VectorDB

db = VectorDB()
db.load()
model = SentenceTransformer("all-MiniLM-L6-v2")
query = "What can I use for analyzing data?"

query_vector = model.encode(query)

results = db.search(query_vector, k=3)

for document, score, metadata in results:
    print(f"{score:.4f} -> {document}")