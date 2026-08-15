import numpy as np


def test_batch_partial_acceptance(db, vector):

    documents = [
        "valid document 1",
        "",
        "valid document 2"
    ]

    vectors = [
        vector,
        vector,
        vector
    ]

    metadata = [
        {"source": "a"},
        {"source": "b"},
        {"source": "c"}
    ]

    result = db.add_batch(
        documents,
        vectors,
        metadata
    )

    assert result["added"] == 2
    assert result["failed"] == 1

    assert len(result["added_ids"]) == 2
    assert len(result["errors"]) == 1

    assert db.index.get_current_count() == 2


def test_batch_all_valid(db, vector):

    documents = [
        "document 1",
        "document 2",
        "document 3"
    ]

    vectors = [
        vector,
        vector,
        vector
    ]

    metadata = [
        {},
        {},
        {}
    ]

    result = db.add_batch(
        documents,
        vectors,
        metadata
    )

    assert result["added"] == 3
    assert result["failed"] == 0