def test_update_changes_document(db, vector):

    doc_id = db.add(
        "old document",
        vector,
        {"version": 1}
    )

    old_index_id = db.index_ids[doc_id]

    new_vector = vector.copy()

    db.update(
        doc_id,
        "updated document",
        new_vector,
        {"version": 2}
    )

    new_index_id = db.index_ids[doc_id]

    assert db.document[doc_id] == "updated document"
    assert db.metadata[doc_id]["version"] == 2

    assert new_index_id != old_index_id

    assert db.hnsw_to_doc[new_index_id] == doc_id
    assert old_index_id not in db.hnsw_to_doc


def test_update_invalid_vector_does_not_modify_document(
    db,
    vector
):

    doc_id = db.add(
        "original",
        vector,
        {}
    )

    invalid_vector = vector[:100]

    try:
        db.update(
            doc_id,
            "should not update",
            invalid_vector,
            {}
        )
    except ValueError:
        pass

    assert db.document[doc_id] == "original"