import hnswlib
import numpy as np
from vectordb import VectorDB
from sentence_transformers import SentenceTransformer



dimension = 384
db=VectorDB()
db.load()
index = hnswlib.Index(## this creates index which are based on cosine similarit and have 384 dimensions
    space="cosine",
    dim=dimension
)

index.init_index(
    max_elements=10000,## max element the index can hold
    ef_construction=200,## effort by which hsnw create graph
    M=16## max graph connection allowed per vector
)
vectors = np.array(db.vector).astype(np.float32)
ids = np.arange(len(vectors))

index.add_items(vectors, ids)

#print("Number of vectors:", index.get_current_count())
#print("HNSW index created!")


model = SentenceTransformer("all-MiniLM-L6-v2")
query = "What can I use for analyzing data?"

query_vector = model.encode(query)
query_vector = np.array(query_vector).astype(np.float32)

labels, distances = index.knn_query(
    query_vector,
    k=2
)
for i in range(len(labels[0])):

    idx = labels[0][i]
    distance = distances[0][i]

    similarity = 1 - distance

    document = db.document[idx]
    metadata = db.metadata[idx]

    print(f"{similarity:.4f} -> {document}")
    print(f"Metadata: {metadata}")

print("Labels:", labels)
print("Distances:", distances)