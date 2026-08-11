from vectordb import VectorDB
from sentence_transformers import SentenceTransformer

db = VectorDB()


model=SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Python is used for data analysis.",
    "SQL is used to query relational databases.",
    "Apache Spark processes large datasets.",
    "React is used to build web interfaces."
]

for doc in documents:
    vector = model.encode(doc)
    db.add(doc,vector,{"source": "learning_notes.txt"})
    db.save()

query = "What can I use for analyzing data?"

query_vector = model.encode(query)

results = db.search(query_vector, k=2)

for document, score, metadata in results:
    print(f"{score:.4f} -> {document}")
    print(f"Metadata: {metadata}")