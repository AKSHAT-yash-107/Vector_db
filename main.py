from sentence_transformers import SentenceTransformer
from vectordb import VectorDB

model = SentenceTransformer("all-MiniLM-L6-v2")

db = VectorDB()
db.load()

query = "What can I use for analyzing data?"

query_vector = model.encode(query)

results = db.search(query_vector, k=2)

for document, score, metadata in results:
    print(f"{score:.4f} -> {document}")
    print(f"Metadata: {metadata}")