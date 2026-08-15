def test_delete_document(db, vector):

    doc_id = db.add(
        "document",
        vector,
        {}
    )

    index_id = db.index_ids[doc_id]

    result = db.delete(doc_id)

    assert result is True

    assert db.document[doc_id] is None
    assert db.vector[doc_id] is None
    assert db.metadata[doc_id] is None

    assert doc_id not in db.index_ids
    assert index_id not in db.hnsw_to_doc


def test_delete_invalid_document(db):

    result = db.delete(999)

    assert result is False


def test_delete_twice(db, vector):

    doc_id = db.add(
        "document",
        vector,
        {}
    )

    assert db.delete(doc_id) is True
    assert db.delete(doc_id) is False