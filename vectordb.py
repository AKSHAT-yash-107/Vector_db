import json
import numpy as np
import hnswlib
import os




class VectorDB:
    def __init__(self):
        self.document = []
        self.vector = []
        self.metadata = []
        self.max_elements = 10000
        self.dimension = 384
        self.index=hnswlib.Index(
            space="cosine",
            dim=self.dimension
        )
        self.index.init_index(
            max_elements=self.max_elements,
            ef_construction=200,
            M=16
        )
        self.index.set_ef(50)#ef controls how many candidates HNSW considers during search.
        self.index_ids = {}
        self.next_index_id = 0
        self.hnsw_to_doc = {}

    def add(self, document, vector, metadata=None):

        valid, error = self.valid_doc(
            document,
            vector,
            metadata
        )

        if not valid:
            raise ValueError(error)

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
        self.hnsw_to_doc[index_id] = doc_id

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
                self.hnsw_to_doc[i] = i

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

        filter_fn = None

        if filter is not None:
            def filter_fn(hnsw_id):
                return self.matches_filter(
                    hnsw_id,
                    filter
                )

        labels, distances = self.index.knn_query(
            query_vector,
            k=k,
            num_threads=1,
            filter=filter_fn
        )

        results = []

        for i in range(len(labels[0])):

            hnsw_id = labels[0][i]

            # HNSW ID -> Document ID
            doc_id = self.hnsw_to_doc[hnsw_id]

            if self.document[doc_id] is None:
                continue

            distance = distances[0][i]
            similarity = 1 - distance

            results.append(
                (
                    self.document[doc_id],
                    similarity,
                    self.metadata[doc_id]
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

        self.index.mark_deleted(old_index_id)

        vector = np.asarray(
            vector,
            dtype=np.float32
        )

        new_index_id = self.next_index_id

        self.index.add_items(
            vector,
            new_index_id
        )

        self.document[doc_id] = document
        self.vector[doc_id] = vector
        self.metadata[doc_id] = metadata

        self.index_ids[doc_id] = new_index_id
        self.hnsw_to_doc[new_index_id] = doc_id

        self.next_index_id += 1

        return True
    def matches_filter(self, hnsw_id, filter):

        doc_id = self.hnsw_to_doc[hnsw_id]

        metadata = self.metadata[doc_id]

        for key, value in filter.items():

            if metadata.get(key) != value:
                return False

        return True

    def valid_doc(self, document, vector, metadata):
        if document is None or not isinstance(document, str):
            return False, "Invalid document"
        if not document.strip():
            return False, "Document is empty"
        if vector is None:
            return False, "Vector is missing"
        try:
            vector = np.asarray(vector, dtype=np.float32)
        except (ValueError, TypeError):
            return False, "Vector could not be converted to a numeric array"
        if vector.ndim != 1:
            return False, "Vector must be 1-dimensional"
        if vector.shape[0] != self.dimension:
            return False, "Invalid vector dimension"
        if not np.all(np.isfinite(vector)):
            return False, "Vector contains NaN or Inf values"
        if np.linalg.norm(vector) == 0:
            return False, "Vector is a zero vector"
        if metadata is not None and not isinstance(metadata, dict):
            return False, "Metadata must be a dictionary"
        return True, None

    def add_batch(self, documents, vectors, metadata):

        # 1. Validate batch structure
        if not (
                len(documents)
                == len(vectors)
                == len(metadata)
        ):
            raise ValueError(
                "documents, vectors and metadata must have the same length"
            )

        valid_docs = []
        errors = []

        # 2. Validate every document independently
        for i in range(len(documents)):

            document = documents[i]
            vector = vectors[i]
            meta = metadata[i]

            valid, error = self.valid_doc(
                document,
                vector,
                meta
            )

            if not valid:
                errors.append({
                    "index": i,
                    "reason": error
                })
                continue

            vector = np.asarray(
                vector,
                dtype=np.float32
            )

            valid_docs.append(
                (document, vector, meta)
            )

        # 3. Nothing valid
        if not valid_docs:
            return {
                "added": 0,
                "failed": len(errors),
                "errors": errors,
                "added_ids": []
            }

        # 4. Number of valid vectors
        n = len(valid_docs)

        # 5. Check HNSW capacity
        if self.index.get_current_count() + n > self.max_elements:
            raise ValueError(
                "HNSW index capacity exceeded"
            )

        # 6. Build one NumPy matrix
        batch_vectors = np.asarray(
            [item[1] for item in valid_docs],
            dtype=np.float32
        )

        # 7. Allocate HNSW labels
        start_index_id = self.next_index_id

        new_index_ids = np.arange(
            start_index_id,
            start_index_id + n
        )

        # 8. ONE bulk HNSW insertion
        self.index.add_items(
            batch_vectors,
            new_index_ids
        )

        # 9. Update storage and mappings
        added_ids = []

        for (document, vector, meta), index_id in zip(
                valid_docs,
                new_index_ids
        ):
            # Same ID allocation model as add()
            doc_id = len(self.document)

            self.document.append(document)
            self.vector.append(vector)
            self.metadata.append(meta)

            # Convert NumPy integer → Python int
            index_id = int(index_id)

            # Both directions
            self.index_ids[doc_id] = index_id
            self.hnsw_to_doc[index_id] = doc_id

            added_ids.append(doc_id)

        # 10. Advance HNSW ID counter
        self.next_index_id += n

        return {
            "added": n,
            "failed": len(errors),
            "errors": errors,
            "added_ids": added_ids
        }