import numpy as np


def test_add_valid_document(db, vector):

    doc_id = db.add(
        "Python is used for data engineering.",
        vector,
        {"source": "test.pdf"}
    )

    assert doc_id == 0
    assert db.document[0] == "Python is used for data engineering."
    assert db.vector[0].shape == (384,)
    assert db.metadata[0]["source"] == "test.pdf"

    assert db.index_ids[0] == 0
    assert db.hnsw_to_doc[0] == 0


def test_invalid_vector_dimension(db):

    vector = np.ones(128, dtype=np.float32)

    valid, error = db.valid_doc(
        "test",
        vector,
        {}
    )

    assert valid is False
    assert "dimension" in error.lower()


def test_zero_vector_rejected(db):

    vector = np.zeros(384, dtype=np.float32)

    valid, error = db.valid_doc(
        "test",
        vector,
        {}
    )

    assert valid is False
    assert "zero" in error.lower()


def test_nan_vector_rejected(db):

    vector = np.ones(384, dtype=np.float32)
    vector[0] = np.nan

    valid, error = db.valid_doc(
        "test",
        vector,
        {}
    )

    assert valid is False
    assert "nan" in error.lower() or "inf" in error.lower()


def test_empty_document_rejected(db, vector):

    valid, error = db.valid_doc(
        "",
        vector,
        {}
    )

    assert valid is False
    assert "empty" in error.lower()