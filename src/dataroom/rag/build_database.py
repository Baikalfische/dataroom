"""
Build and manage vector databases for PDF and CSV documents.
Implements dual database architecture with CLIP embeddings.
"""

import os
import yaml
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .embedder import DocumentEmbedder
from .chunks import DocumentChunker
from ..tools import parser

class VectorDatabaseManager:
    """
    Vector database manager supporting separate storage for PDF and CSV.
    Uses the CLIP embedding model for text vectorization.
    """
    
    def __init__(self):
        """Initialize the database manager."""
        # Load settings from YAML configuration file
        config_path = Path(__file__).parent / "rag_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        vector_config = self.config["vector_store_config"]
        
    # Initialize embedder and chunker
        self.embedder = DocumentEmbedder()
        self.chunker = DocumentChunker()
        
    # Initialize PDF database client
        pdf_db_path = vector_config["pdf_database"]["persist_directory"]
        self.pdf_client = chromadb.PersistentClient(
            path=pdf_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
    # Initialize CSV database client
        csv_db_path = vector_config["csv_database"]["persist_directory"]
        self.csv_client = chromadb.PersistentClient(
            path=csv_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
    # Create collections
        self.pdf_collection = self.pdf_client.get_or_create_collection(
            name=vector_config["pdf_database"]["collection_name"],
            metadata={"description": "PDF documents with page-level chunks"}
        )
        
        self.csv_collection = self.csv_client.get_or_create_collection(
            name=vector_config["csv_database"]["collection_name"], 
            metadata={"description": "CSV documents with row-level chunks"}
        )
        
        print("✅ VectorDatabaseManager initialized with CLIP embeddings")
        print(f"   📁 PDF Database: {pdf_db_path}")
        print(f"   📁 CSV Database: {csv_db_path}")
        print(f"   🧠 Embedding Model: CLIP (512 dimensions)")
    
    def add_pdf_document(self, 
                        file_path: str, 
                        document_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add PDF document to vector database - store using page_chunks directly."""
        try:
            print(f"📄 Processing PDF document: {file_path}")
            
            # Parse PDF document
            parse_result = parser.parse_pdf(file_path)
            if "error" in parse_result:
                return {"error": f"Failed to parse PDF: {parse_result['error']}"}
            
            # Get page chunks
            page_chunks = parse_result.get("page_chunks", [])
            if not page_chunks:
                return {"error": "No page chunks found in parse result"}
            
            # Generate document ID
            if not document_id:
                document_id = f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare document metadata
            doc_metadata = {
                "document_id": document_id,
                "filename": parse_result.get("filename", "unknown.pdf"),
                "file_type": "pdf",
                "file_path": file_path,
                "upload_time": datetime.now().isoformat(),
                "total_pages": len(page_chunks),
                **(metadata or {})
            }
            
            # Generate embedding for each page and store
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for page_chunk in page_chunks:
                # Generate page-level ID
                page_id = f"{document_id}_page_{page_chunk['metadata']['page']}"
                
                # Generate embedding vector
                text_content = page_chunk["text"]
                embedding = self.embedder.embed(text_content)
                
                # Prepare per-page metadata
                page_metadata = {
                    "document_id": document_id,
                    "filename": doc_metadata["filename"],
                    "page_number": page_chunk["metadata"]["page"],
                    "file_type": "pdf",
                    "chunk_type": "page",
                    "upload_time": doc_metadata["upload_time"]
                }
                
                ids.append(page_id)
                documents.append(text_content)
                embeddings.append(embedding)
                metadatas.append(page_metadata)
            
            # Batch add to PDF collection
            self.pdf_collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            result = {
                "status": "success",
                "document_id": document_id,
                "database": "pdf_db",
                "collection": "pdf_documents",
                "pages_added": len(page_chunks),
                "metadata": doc_metadata
            }
            
            print(f"✅ PDF document added to pdf_db: {len(page_chunks)} pages")
            return result
            
        except Exception as e:
            return {"error": f"Error adding PDF document: {str(e)}"}
    
    def add_csv_document(self, 
                        file_path: str, 
                        document_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add CSV document to vector database - split by rows using chunker."""
        try:
            print(f"📊 Processing CSV document: {file_path}")
            
            # Parse CSV document
            parse_result = parser.parse_csv(file_path)
            if "error" in parse_result:
                return {"error": f"Failed to parse CSV: {parse_result['error']}"}
            
            # Generate document ID
            if not document_id:
                document_id = f"csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare document metadata
            doc_metadata = {
                "document_id": document_id,
                "filename": parse_result.get("filename", "unknown.csv"),
                "file_type": "csv",
                "file_path": file_path,
                "upload_time": datetime.now().isoformat(),
                "rows": parse_result.get("rows", 0),
                "columns": parse_result.get("columns", 0),
                **(metadata or {})
            }
            print(doc_metadata)
            # Get markdown content
            markdown_content = parse_result.get("markdown_content", "")
            if not markdown_content:
                return {"error": "No markdown content found in parse result"}
            
            # Chunk content
            chunks = self.chunker.chunk_csv_markdown(markdown_content, doc_metadata)
            
            if not chunks:
                return {"error": "No chunks generated from CSV"}
            
            # Generate embeddings
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.embedder.embed(texts)
            
            # Prepare ChromaDB data
            ids = [chunk["chunk_id"] for chunk in chunks]
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Add to CSV collection
            self.csv_collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            result = {
                "status": "success",
                "document_id": document_id,
                "database": "csv_db",
                "collection": "csv_documents",
                "chunks_added": len(chunks),
                "metadata": doc_metadata
            }
            
            print(f"✅ CSV document added to csv_db: {len(chunks)} chunks")
            return result
            
        except Exception as e:
            return {"error": f"Error adding CSV document: {str(e)}"}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pdf_count = self.pdf_collection.count()
        csv_count = self.csv_collection.count()
        
        vector_config = self.config["vector_store_config"]
        
        return {
            "pdf_database": {
                "path": vector_config["pdf_database"]["persist_directory"],
                "collection_name": vector_config["pdf_database"]["collection_name"],
                "total_chunks": pdf_count,
                "description": "PDF documents with page-level chunks"
            },
            "csv_database": {
                "path": vector_config["csv_database"]["persist_directory"],
                "collection_name": vector_config["csv_database"]["collection_name"], 
                "total_chunks": csv_count,
                "description": "CSV documents with row-level chunks"
            },
            "embedding_model": "CLIP (512 dimensions)",
            "total_chunks": pdf_count + csv_count
        }
    
    def reset_databases(self):
        """Reset all databases."""
        vector_config = self.config["vector_store_config"]
        
        # Reset PDF database
        self.pdf_client.delete_collection(vector_config["pdf_database"]["collection_name"])
        self.pdf_collection = self.pdf_client.get_or_create_collection(
            name=vector_config["pdf_database"]["collection_name"],
            metadata={"description": "PDF documents with page-level chunks"}
        )
        
        # Reset CSV database
        self.csv_client.delete_collection(vector_config["csv_database"]["collection_name"])
        self.csv_collection = self.csv_client.get_or_create_collection(
            name=vector_config["csv_database"]["collection_name"],
            metadata={"description": "CSV documents with row-level chunks"}
        )
        
        print("🔄 Databases reset successfully")
        print(f"   📁 PDF Database: {vector_config['pdf_database']['persist_directory']}")
        print(f"   📁 CSV Database: {vector_config['csv_database']['persist_directory']}")
        
    def list_documents(self) -> Dict[str, Any]:
        """List existing files in both collections with simple statistics.

        Because Chroma doesn't provide a direct distinct query, we fetch all ids in batches (pagination)
        and aggregate by filename in metadata. For very large datasets, consider maintaining a persistent index.
        """
        result: Dict[str, Any] = {"pdf": {}, "csv": {}}
        try:
            page_size = 500  # Max items per batch

            # ---- PDF ----
            pdf_total = self.pdf_collection.count()
            offset = 0
            while offset < pdf_total:
                batch = self.pdf_collection.get(include=["metadatas"], limit=page_size, offset=offset)
                for meta in batch.get("metadatas", []):
                    if not meta:
                        continue
                    fname = meta.get("filename", "unknown.pdf")
                    page = meta.get("page_number") or meta.get("page")
                    info = result["pdf"].setdefault(fname, {"pages": set()})
                    if page is not None:
                        info["pages"].add(page)
                offset += page_size
            for fname, info in result["pdf"].items():
                info["total_pages"] = len(info["pages"]) or None
                info.pop("pages", None)

            # ---- CSV ----
            csv_total = self.csv_collection.count()
            offset = 0
            while offset < csv_total:
                batch = self.csv_collection.get(include=["metadatas"], limit=page_size, offset=offset)
                for meta in batch.get("metadatas", []):
                    if not meta:
                        continue
                    fname = meta.get("filename", "unknown.csv")
                    row = meta.get("row")
                    info = result["csv"].setdefault(fname, {"rows": set()})
                    if row is not None:
                        info["rows"].add(row)
                offset += page_size
            for fname, info in result["csv"].items():
                info["total_rows"] = len(info["rows"]) or None
                info.pop("rows", None)

            result["summary"] = {
                "pdf_files": len(result["pdf"]),
                "csv_files": len(result["csv"])
            }
        except Exception as e:
            return {"error": f"list_documents failed: {e}"}
        return result

if __name__ == "__main__":
    # Build databases from data folder directly
    print("🚀 Building vector databases from data folder...")

    db_manager = VectorDatabaseManager()
    data_dir = Path("data")

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please create 'data' folder and put your documents there.")
        exit(1)

    # Process PDF files
    pdf_files = list(data_dir.glob("*.pdf"))
    if pdf_files:
        print(f"📄 Found {len(pdf_files)} PDF files")
        for file_path in pdf_files:
            print(f"   Processing: {file_path.name}")
            result = db_manager.add_pdf_document(str(file_path))
            if "error" not in result:
                print(f"   ✅ Added {result['pages_added']} pages")
            else:
                print(f"   ❌ Failed: {result['error']}")

    # Process CSV files
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        print(f"📊 Found {len(csv_files)} CSV files")
        for file_path in csv_files:
            print(f"   Processing: {file_path.name}")
            result = db_manager.add_csv_document(str(file_path))
            if "error" not in result:
                print(f"   ✅ Added {result['chunks_added']} chunks")
            else:
                print(f"   ❌ Failed: {result['error']}")

    # Show final statistics
    stats = db_manager.get_collection_stats()
    print(f"\n✅ Database build completed!")
    print(f"📄 PDF Database: {stats['pdf_database']['total_chunks']} pages")
    print(f"📊 CSV Database: {stats['csv_database']['total_chunks']} chunks")
    print(f"📚 Total: {stats['total_chunks']} chunks")
