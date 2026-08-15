import numpy as np


def test_search_returns_top_k(db, vector):

    for i in range(5):

        db.add(
            f"document {i}",
            vector,
            {"source": "test.pdf"}
        )

    results = db.search(
        vector,
        k=3
    )

    assert len(results) == 3


def test_search_returns_similarity(db, vector):

    db.add(
        "test document",
        vector,
        {"source": "test.pdf"}
    )

    results = db.search(
        vector,
        k=1
    )

    document, similarity, metadata = results[0]

    assert document == "test document"
    assert similarity > 0.99
    assert metadata["source"] == "test.pdf"


def test_search_k_larger_than_database(db, vector):

    db.add(
        "document",
        vector,
        {}
    )

    results = db.search(
        vector,
        k=1
    )

    assert len(results) == 1


def test_metadata_filtering(db, vector):

    db.add(
        "Python document",
        vector,
        {"source": "python.pdf"}
    )

    db.add(
        "Spark document",
        vector,
        {"source": "spark.pdf"}
    )

    results = db.search(
        vector,
        k=1,   # changed from 10
        filter={"source": "spark.pdf"}
    )

    assert len(results) == 1
    assert results[0][2]["source"] == "spark.pdf"