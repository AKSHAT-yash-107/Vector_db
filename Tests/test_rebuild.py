def test_rebuild_compacts_deleted_entries(
    db,
    vector
):

    for i in range(5):

        db.add(
            f"document {i}",
            vector,
            {}
        )

    db.delete(0)
    db.delete(1)

    assert db.index.get_current_count() == 5

    result = db.rebuild_index()

    assert result["active_documents"] == 3
    assert result["hnsw_count"] == 3

    assert db.index.get_current_count() == 3

    # Logical IDs remain
    assert 0 not in db.index_ids
    assert 1 not in db.index_ids

    assert 2 in db.index_ids
    assert 3 in db.index_ids
    assert 4 in db.index_ids

    # Reverse mapping is consistent
    for doc_id, index_id in db.index_ids.items():

        assert db.hnsw_to_doc[index_id] == doc_id