from scripts.index_documents import chunk_text


def test_empty_text_returns_empty():
    assert chunk_text("") == []


def test_short_text_is_single_chunk():
    chunks = chunk_text("hello world", size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_long_text_produces_multiple_chunks():
    text = " ".join(str(i) for i in range(600))
    chunks = chunk_text(text, size=512, overlap=64)
    assert len(chunks) > 1


def test_overlap_is_applied():
    # With size=50 overlap=10, second chunk starts at word index 40
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, size=50, overlap=10)
    first_last_10 = chunks[0].split()[-10:]
    second_first_10 = chunks[1].split()[:10]
    assert first_last_10 == second_first_10


def test_chunk_size_respected():
    text = " ".join("word" for _ in range(1000))
    chunks = chunk_text(text, size=100, overlap=0)
    assert all(len(c.split()) == 100 for c in chunks[:-1])
