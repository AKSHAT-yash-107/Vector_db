import json
import numpy as np
import hnswlib
import os
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
        self.index.set_ef(50)#ef controls how many candidates HNSW considers during search.
        self.index_ids = {}
        self.next_index_id = 0

    def add(self, document, vector, metadata=None):

        vector = np.asarray(vector, dtype=np.float32)

        doc_id = len(self.document)
        index_id = self.next_index_id

        self.document.append(document)
        self.vector.append(vector)
        self.metadata.append(metadata)

        self.index.add_items(
            vector,
            index_id
        )

        self.index_ids[doc_id] = index_id
        self.next_index_id += 1

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
        if not os.path.exists(filename):
            return

        with open(filename, "r") as file:
            data = json.load(file)

        self.document = data["document"]

        self.vector= [
            np.array(vector, dtype=np.float32)
            if vector is not None
            else None
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

        active_vectors = []
        active_ids = []

        for i, vector in enumerate(self.vector):

            if vector is not None:
                active_vectors.append(vector)
                active_ids.append(i)
                self.index_ids[i] = i

        if active_vectors:
            self.index.add_items(
                np.array(active_vectors, dtype=np.float32),
                np.array(active_ids)
            )

        self.next_index_id = max(self.index_ids.values(), default=-1) + 1

    def search(self, query_vector, k=2, filter=None):

        self.index.set_ef(max(50, k))

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        labels, distances = self.index.knn_query(
            query_vector,
            k=k
        )

        results = []

        for i in range(len(labels[0])):

            idx = labels[0][i]

            if self.document[idx] is None:
                continue

            # Metadata filtering
            if filter is not None:

                match = True

                for key, value in filter.items():

                    if self.metadata[idx].get(key) != value:
                        match = False
                        break

                if not match:
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

    def update(self, doc_id, document, vector, metadata=None):

        if doc_id < 0 or doc_id >= len(self.document):
            return False

        if self.document[doc_id] is None:
            return False

        # Old HNSW entry
        old_index_id = self.index_ids[doc_id]

        print("Document ID:", doc_id)
        print("Index mapping:", self.index_ids)
        print("Old HNSW ID:", old_index_id)
        print("HNSW count:", self.index.get_current_count())

        self.index.mark_deleted(old_index_id)

        # Create new vector
        vector = np.asarray(vector, dtype=np.float32)

        # New HNSW ID
        new_index_id = self.next_index_id

        self.index.add_items(vector, new_index_id)

        # Update database record
        self.document[doc_id] = document
        self.vector[doc_id] = vector
        self.metadata[doc_id] = metadata

        # Update mapping
        self.index_ids[doc_id] = new_index_id

        # Move counter forward
        self.next_index_id += 1

        return True