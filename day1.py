from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python is used for data analysis.",
    "SQL is used to query relational databases.",
    "Apache Spark processes large datasets.",
    "React is used to build web interfaces."
]
vectors=model.encode(documents)

def search(query, documents, vectors, k=2):

    query_vector = model.encode(query)
    query_vector = np.asarray(query_vector).reshape(-1)

    similarities = np.dot(vectors, query_vector) / (
        np.linalg.norm(vectors, axis=1)
        * np.linalg.norm(query_vector)
    )

    print("\nSIMILARITY SCORES:")

    for document, score in zip(documents, similarities):
        print(f"{score:.4f} -> {document}")

    indices = np.argsort(similarities)[::-1][:k]

    print("\nINDICES:", indices)

    return [(documents[i], similarities[i]) for i in indices]

query =  "What can I use for analyzing data?",

result= search(query,documents,vectors,k=2)
print(result)