import json
import numpy as np

class VectorDB:
    def __init__(self):
        self.document = []
        self.vector = []
        self.metadata = []

    def add(self,documents,vector,metadata=None):
        self.document.append(documents)
        self.vector.append(np.asarray(vector))
        self.metadata.append(metadata)

    def save(self, filename="vector_db.json"):
        data = {
            "document": self.document,
            "vector": [vector.tolist() for vector in self.vector],
            "metadata": self.metadata
        }

        with open(filename, "w") as file:
            json.dump(data, file)

    def load(self, filename="vector_db.json"):
        with open(filename, "r") as file:
            data = json.load(file)

        self.document = data["document"]
        self.vector = [np.array(vector) for vector in data["vector"]]
        self.metadata = data["metadata"]

    def search(self, query_vector, k=2):
        vectors = np.array(self.vector)

        similarities = np.dot(vectors, query_vector) / (
                np.linalg.norm(vectors, axis=1)
                * np.linalg.norm(query_vector)
        )

        indices = np.argsort(similarities)[::-1][:k]

        return [
            (self.document[i], similarities[i], self.metadata[i])
            for i in indices
        ]
