import json
import numpy as np
import hnswlib
class VectorDB:
    def __init__(self):
        self.document = []
        self.vector = []
        self.metadata = []
        self.index=hnswlib.Index(
            space="cosine",
            dim=384
        )
        self.index.init_index(
            max_elements=10000,
            ef_construction=200,
            M=16
        )

    def add(self,documents,vector,metadata=None):
        vectors = np.array(vector,dtype=np.float32)
        idx=len(self.vector)
        self.document.append(documents)
        self.vector.append(np.asarray(vector))
        self.metadata.append(metadata)
        self.index.add_items(vectors,idx)

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
        self.vector= [
            np.array(vector, dtype=np.float32)
            for vector in data["vector"]
        ]
        self.metadata = data["metadata"]

        # Rebuild HNSW index
        self.index = hnswlib.Index(
            space="cosine",
            dim=384
        )

        self.index.init_index(
            max_elements=10000,
            ef_construction=200,
            M=16
        )

        vectors = np.array(self.vector, dtype=np.float32)
        ids = np.arange(len(vectors))

        self.index.add_items(vectors, ids)

    def search(self, query_vector, k=2):
        query_vector=np.asarray(
            query_vector,
            dtype=np.float32
        )
        labels,distances=self.index.knn_query(
            query_vector,
            k=k
        )

        result=[]
        for i in range (len(labels[0])):
            idx=labels[0][i]
            distance=distances[0][i]
            similarity=1-distance

            result.append((
                self.document[idx],
                similarity,
                self.metadata[idx]
            ))
        return result