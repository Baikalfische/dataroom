import os
import chromadb
import sys
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag.data_loader import DataLoader
from rag.embedder import MultimodalEmbedder

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_JSON_PATH = os.path.join(PROJECT_ROOT, "data/local_dataset.json")
IMAGE_BASE_PATH = os.path.join(PROJECT_ROOT, "data/figures")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "rag_config.yaml")

def build_database():
    """
    Builds the ChromaDB vector database based on configuration.
    This script performs a full rebuild each time it's run.
    """
    # 1. Load Configuration
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    database_type = config["vector_store_config"]["database_type"]
    print(f"--- 构建 {database_type.upper()} 级别数据库 ---")
    
    # 2. Initialize Embedder
    print("--- Step 1: Initializing Multimodal Embedder ---")
    embedder = MultimodalEmbedder()
    embedding_function = embedder.get_embedding_function()

    # 3. Get database configuration
    if database_type == "case_level":
        db_config = config["vector_store_config"]["case_level"]
        level = "case"
    else:
        db_config = config["vector_store_config"]["chunk_level"]
        level = "chunk"
    
    DB_PATH = os.path.join(PROJECT_ROOT, db_config["persist_directory"])
    COLLECTION_NAME = db_config["collection_name"]

    # 4. Initialize ChromaDB Client
    print(f"\n--- Step 2: Initializing ChromaDB at: {DB_PATH} ---")
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Delete the old collection if it exists to ensure a fresh build
    try:
        print(f"Checking for existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' for a fresh start.")
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found, creating a new one.")

    # Create a new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print("New collection created.")

    # 5. Initialize DataLoader
    print(f"\n--- Step 3: Initializing DataLoader with data from {DATA_JSON_PATH} ---")
    data_loader = DataLoader(data_path=DATA_JSON_PATH, image_base_path=IMAGE_BASE_PATH, level=level)

    # 6. Process and Add Data to ChromaDB
    print(f"\n--- Step 4: Processing data and generating embeddings ---")

    documents_to_add = []
    embeddings_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    counter = 0

    for doc_text, metadata, image in data_loader.load_data():
        case_id = metadata["case_id"]
        
        # 根据级别处理
        if level == "case":
            # Case级别：使用图片信息
            print(f"📸 Case {case_id}: {metadata.get('title', '')[:50]}...")
            embedding = embedding_function(texts=[doc_text], images=[image])
            unique_id = f"case_{case_id}"
        else:
            # Chunk级别：不使用图片信息
            print(f"📄 Chunk {counter} (Case {case_id}): {doc_text[:50]}...")
            embedding = embedding_function(texts=[doc_text], images=[None])
            unique_id = f"chunk_{case_id}_{counter}"
            counter += 1
        
        ids_to_add.append(unique_id)
        documents_to_add.append(doc_text)
        metadatas_to_add.append(metadata)
        embeddings_to_add.append(embedding)

    # Batch insert into ChromaDB for efficiency
    if documents_to_add:
        print(f"\n--- Step 5: Adding {len(documents_to_add)} items to the database ---")
        collection.add(
            embeddings=embeddings_to_add,
            documents=documents_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        print(f"\n✅ {database_type.upper()} 级别数据库构建完成!")
    else:
        print("⚠️ No documents were generated to add to the database.")
        
    # Verify count
    count = collection.count()
    print(f"Collection '{COLLECTION_NAME}' now contains {count} items.")
    print(f"Database is stored at: {DB_PATH}")


if __name__ == "__main__":
    build_database() 