from app import store


def test_query_ids_persist_across_restart():
    store.register_query("q-1", "question", "answer", [], None)
    store.reset()  # simulates a server restart (memory cleared, file kept)
    assert store.is_known("q-1")
    assert not store.is_known("never-seen")
