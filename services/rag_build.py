import json
import sys
from pathlib import Path

import chromadb
from google import genai

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.config import Config
from services.rag_builder import GeminiEmbeddingFunction, _normalize_crop_tag

DATA_SOURCE_DIR = Path(Config.rag_kb_dir)
DB_DIR = Path(Config.chroma_db_dir)
PROGRESS_FILE = Path(Config.data_dir) / "indexing_progress.json"
COLLECTION_NAME = Config.chroma_collection_name

genai_client = genai.Client(api_key=Config.gemini_api_key)
gemini_ef = GeminiEmbeddingFunction(client=genai_client)

chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=gemini_ef,
    metadata={"hnsw:space": "cosine"},
)


def load_progress():
    """Load the list of already indexed files."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                return set()
    return set()


def save_progress(indexed_files):
    """Save the list of indexed files to disk."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(indexed_files), f)


def build_rag_corpus():
    indexed_files = load_progress()
    print("--- BUILDING LOCAL RAG CORPUS (NEW SDK) ---")
    print(f"Total files already indexed: {len(indexed_files)}")

    if not DATA_SOURCE_DIR.exists():
        print(f"Error: Could not find folder {DATA_SOURCE_DIR}")
        return

    doc_id_counter = len(indexed_files)

    try:
        # 1. Iterate through each crop folder
        for crop_folder in sorted(DATA_SOURCE_DIR.iterdir()):
            if crop_folder.is_dir():
                crop_name = _normalize_crop_tag(crop_folder.name)

                # 2. Process every q_x.txt file
                for file_path in sorted(crop_folder.glob("*.txt")):
                    # Create a unique file identifier
                    file_id = f"{crop_name}/{file_path.name}"

                    if file_id in indexed_files:
                        continue  # Skip already processed files

                    try:
                        content = file_path.read_text(encoding="utf-8").strip()
                        if not content:
                            continue

                        print(f"Indexing: {file_id}...")

                        # 3. Add to ChromaDB
                        collection.add(
                            documents=[content],
                            metadatas=[{"crop": crop_name, "source": file_path.name}],
                            ids=[f"id_{doc_id_counter}"],
                        )

                        # Mark as indexed and update counter
                        indexed_files.add(file_id)
                        doc_id_counter += 1

                        # Save progress every file to ensure last known position is kept
                        save_progress(indexed_files)

                    except Exception as e:
                        print(f"  Failed to index {file_id}: {e}")

        print("\n" + "=" * 50)
        print(f"SUCCESS: Total indexed documents: {doc_id_counter}")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user (Ctrl+C).")
        print(f"Progress saved. {len(indexed_files)} files processed total.")
        print("You can run the script again to resume.")
        sys.exit(0)


if __name__ == "__main__":
    build_rag_corpus()
