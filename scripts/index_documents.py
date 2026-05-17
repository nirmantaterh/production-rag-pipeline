"""Index documents into FAISS vector store."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.retriever import FAISSRetriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/docs/")
    parser.add_argument("--output", default="data/faiss_index")
    args = parser.parse_args()
    loader = DirectoryLoader(args.source, loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(docs)
    FAISSRetriever.build_index(chunks, save_path=args.output)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents -> {args.output}")

if __name__ == "__main__":
    main()
