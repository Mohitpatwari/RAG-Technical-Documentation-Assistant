"""
scripts/ingest.py
─────────────────
Standalone ingestion script.

Usage:
    python scripts/ingest.py                          # ingest data/corpus/
    python scripts/ingest.py --file path/to/file.md  # ingest a single file
    python scripts/ingest.py --url https://...        # ingest a URL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.ingestion.pipeline import ingest_corpus, ingest_file, ingest_url
from app.vectorstore import get_collection_count

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Docs Ingestion Script")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=Path, help="Single file to ingest (.md/.txt/.html)")
    group.add_argument("--url", type=str, help="URL to fetch and ingest")
    args = parser.parse_args()

    print("=== RAG Docs — Ingestion Script ===\n")

    if args.file:
        print(f"Ingesting file: {args.file}")
        n = ingest_file(args.file, show_progress=True)
        print(f"\n✓ Ingested {n} chunks from {args.file}")

    elif args.url:
        print(f"Ingesting URL: {args.url}")
        n = ingest_url(args.url, show_progress=True)
        print(f"\n✓ Ingested {n} chunks from {args.url}")

    else:
        corpus_dir = Path(settings.corpus_dir)
        manifest = corpus_dir / "manifest.json"
        print(f"Ingesting corpus from: {corpus_dir}")
        result = ingest_corpus(corpus_dir, manifest_path=manifest)
        print(
            f"\n✓ Ingested {result['docs_ingested']} docs, "
            f"{result['chunks_ingested']} chunks"
        )

    total = get_collection_count()
    print(f"\nTotal chunks in vector store: {total}")


if __name__ == "__main__":
    main()
