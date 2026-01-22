import os
import json
import sys
from pathlib import Path
from google import genai
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from services.config import Config

# --- PATH RESOLUTION ---
SCRIPT_PATH = Path(__file__).resolve()
ROOT_DIR = SCRIPT_PATH.parents[1] 

# Paths
INPUT_FILE = ROOT_DIR / "data" / "queryanalysis" / "final_query" / "individual_queries.txt"
DB_DIR = ROOT_DIR / "data" / "chroma_db"
OUTPUT_FOLDER = ROOT_DIR / "data" / "queryanalysis" / "retrieval_results"


genai_client = genai.Client(api_key=Config.gemini_api_key)

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client, model_name="text-embedding-004"):
        self.client = client
        self.model_name = model_name

    def __call__(self, input_texts: Documents) -> Embeddings:
        result = self.client.models.embed_content(model=self.model_name, contents=input_texts)
        return [e.values for e in result.embeddings]

    def name(self) -> str:
        return "GeminiEmbeddingFunction"

# --- INITIALIZE DATABASE ---
gemini_ef = GeminiEmbeddingFunction(client=genai_client)
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

try:
    collection = chroma_client.get_collection(name="crop_knowledge_base", embedding_function=gemini_ef)
except chromadb.errors.NotFoundError:
    print(f"❌ Error: Collection not found in {DB_DIR}")
    sys.exit(1)

def run_optimized_retrieval():
    print(f"--- PROMPT 8: REFINED RETRIEVAL & GAP ANALYSIS ---")
    
    if not INPUT_FILE.exists():
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    # Get a list of actual crop tags in the DB for flexible mapping
    try:
        all_metadata = collection.get(include=['metadatas'])['metadatas']
        valid_db_crops = set(m.get('crop') for m in all_metadata)
    except Exception as e:
        print(f"⚠️ Could not fetch metadata census: {e}")
        valid_db_crops = set()

    queries = INPUT_FILE.read_text(encoding='utf-8').strip().splitlines()
    results_package = []

    for line in queries:
        if "|" not in line: continue
        
        parts = line.split("|")
        locked_crop = parts[0].strip().lower()
        atomic_query = parts[1].strip()

        # --- ROBUST METADATA MAPPING ---
        # 1. Normalize input by replacing spaces with underscores (e.g., 'green gram' -> 'green_gram')
        normalized_crop = locked_crop.replace(" ", "_")
        
        search_crop_tag = normalized_crop
        
        # 2. Check DB census for exact match or 'name_local' style suffixes (e.g., 'paddy_dhan')
        if normalized_crop not in valid_db_crops:
            for db_crop in valid_db_crops:
                if db_crop == normalized_crop or db_crop.startswith(f"{normalized_crop}_"):
                    search_crop_tag = db_crop
                    break
        # -------------------------------

        # 1. Targeted Metadata Search
        search_results = collection.query(
            query_texts=[atomic_query],
            n_results=3, 
            where={"crop": search_crop_tag} 
        )

        # 2. Status Check
        has_local_data = len(search_results['documents'][0]) > 0
        top_distance = search_results['distances'][0][0] if has_local_data else 1.0
        
        # Threshold set to 0.35 for high-precision matching
        status = "FOUND" if has_local_data and top_distance < 0.35 else "MISSING"

        # 3. Content Deduplication
        raw_evidence = search_results['documents'][0] if status == "FOUND" else []
        clean_evidence = list(dict.fromkeys(raw_evidence)) 

        # 4. Final Package Assembly (Distance Removed)
        package = {
            "query": atomic_query,
            "crop": locked_crop,
            "status": status,
            "evidence": clean_evidence
        }
        results_package.append(package)
        print(f"Processed: {atomic_query[:40]}... -> {status} (Used Tag: {search_crop_tag} | Score: {top_distance:.4f})")

    # Save finalized package
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_FOLDER / "retrieval_audit.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_package, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Retrieval Package Ready: {output_path}")

if __name__ == "__main__":
    run_optimized_retrieval()