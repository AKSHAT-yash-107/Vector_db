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

    def add(self, document, vector, metadata=None):
        vector = np.asarray(vector, dtype=np.float32)

        doc_id = len(self.document)

        self.document.append(document)
        self.vector.append(vector)
        self.metadata.append(metadata)

        self.index.add_items(vector, doc_id)

        return doc_id

    def save(self, filename="vector_db.json"):

        data = {
            "document": self.document,
            "vector": [
                vector.tolist() if vector is not None else None
                for vector in self.vector
            ],
            "metadata": self.metadata
        }

        with open(filename, "w") as file:
            json.dump(data, file)

    def load(self, filename="vector_db.json"):

        with open(filename, "r") as file:
            data = json.load(file)

        self.document = data["documents"]

        self.vector= [
            np.array(vector, dtype=np.float32)
            if vector is not None
            else None
            for vector in data["vectors"]
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

        active_vectors = []
        active_ids = []

        for i, vector in enumerate(self.vector):

            if vector is not None:
                active_vectors.append(vector)
                active_ids.append(i)

        if active_vectors:
            self.index.add_items(
                np.array(active_vectors, dtype=np.float32),
                np.array(active_ids)
            )

    def search(self, query_vector, k=2):
        query_vector=np.asarray(
            query_vector,
            dtype=np.float32
        )
        labels,distances=self.index.knn_query(
            query_vector,
            k=k
        )

        results = []

        for i in range(len(labels[0])):

            idx = labels[0][i]

            if self.document[idx] is None:
                continue

            distance = distances[0][i]
            similarity = 1 - distance

            results.append(
                (
                    self.document[idx],
                    similarity,
                    self.metadata[idx]
                )
            )

        return results

    def get(self, doc_id):
        if doc_id < 0 or doc_id >= len(self.document):
            return None
        return {
            "id": doc_id,
            "document": self.document[doc_id],
            "vector": self.vector[doc_id],
            "metadata": self.metadata[doc_id]
        }

    def delete(self, doc_id):

        if doc_id < 0 or doc_id >= len(self.document):
            return False

        self.index.mark_deleted(doc_id)

        self.document[doc_id] = None
        self.vector[doc_id] = None
        self.metadata[doc_id] = None

        return True