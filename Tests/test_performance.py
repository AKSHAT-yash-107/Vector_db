from vectordb import VectorDB


def test_save_and_load(
    db,
    vector,
    tmp_path
):

    db.add(
        "document 1",
        vector,
        {"source": "a"}
    )

    db.add(
        "document 2",
        vector,
        {"source": "b"}
    )

    path = tmp_path / "vector_db.json"

    db.save(str(path))

    new_db = VectorDB()

    new_db.load(str(path))

    assert len(new_db.document) == 2
    assert new_db.index.get_current_count() == 2

    assert new_db.document == db.document
    assert new_db.index_ids == db.index_ids
    assert new_db.hnsw_to_doc == db.hnsw_to_doc

    assert new_db.next_index_id == db.next_index_id