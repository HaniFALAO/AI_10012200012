from src.preprocessing.chunking import _word_windows


def test_word_windows_generates_chunks() -> None:
    words = ["w"] * 1000
    chunks = _word_windows(words, chunk_size=400, overlap=80)
    assert len(chunks) >= 2
    assert all(len(c.split()) >= 40 for c in chunks)
