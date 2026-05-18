"""Index a directory of text files into Qdrant via BGE-M3."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.retriever import HybridRetriever

def chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/docs/")
    args = parser.parse_args()
    retriever = HybridRetriever()
    docs = []
    for path in Path(args.source).rglob("*.txt"):
        for chunk in chunk_text(path.read_text()):
            docs.append({"text": chunk, "source": str(path)})
    retriever.index(docs)
    print(f"Indexed {len(docs)} chunks from {args.source}")

if __name__ == "__main__":
    main()
