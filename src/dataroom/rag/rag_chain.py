"""
RAG Retrieval Chain - Real estate document Q&A system.
Supports intelligent retrieval and traceability for PDF and CSV documents.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from chromadb.config import Settings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from .embedder import DocumentEmbedder
from .build_database import VectorDatabaseManager

class RealEstateRAGChain:
    """
    Real estate document RAG retrieval chain.
    Supports intelligent retrieval and traceability for PDF and CSV documents.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RAG retrieval chain.
        
        Args:
            config_path: Path to configuration file.
        """
        if config_path is None:
            config_path = str(Path(__file__).parent / "rag_config.yaml")

        print("🚀 Initializing Real Estate RAG Chain...")

        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        model_config = self.config["model_config"]
        vector_config = self.config["vector_store_config"]
        prompt_config = model_config["prompt_template"]

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_config["lm_model"]["name"],
            temperature=model_config["lm_model"]["temperature"],
            max_output_tokens=model_config["lm_model"]["max_length"],
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Initialize embedder
        self.embedder = DocumentEmbedder()

        # Initialize database manager
        self.db_manager = VectorDatabaseManager()

        # Retrieval parameters
        self.pdf_k = vector_config["pdf_database"]["search_kwargs"]["k"]
        self.csv_k = vector_config["csv_database"]["search_kwargs"]["k"]

        # Initialize prompt template (created directly from configuration)
        self.prompt = PromptTemplate(
            template=prompt_config["template"],
            input_variables=prompt_config["input_variables"]
        )

        print("✅ Real Estate RAG Chain initialized successfully")
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the RAG retrieval pipeline.
        
        Args:
            inputs: Input dictionary containing a "query" key.
            
        Returns:
            Retrieval results and generated answer.
        """
        query = inputs.get("query")
        
        if not query:
            raise ValueError("Query must be provided in inputs")
        
        print(f"🔍 Searching for: {query}")

        # Generate query embedding
        query_embedding = self.embedder.embed(query)

        # Stage 1: PDF retrieval
        print("🔍 Stage 1: PDF retrieval")
        pdf_results = self._pdf_retrieval(query_embedding)

        # Stage 2: CSV retrieval
        print("🔍 Stage 2: CSV retrieval")
        csv_results = self._csv_retrieval(query_embedding)

        # Merge retrieval results
        all_documents = []
        all_metadatas = []

        # Process PDF results
        if pdf_results['documents'] and pdf_results['documents'][0]:
            for doc, metadata in zip(
                pdf_results['documents'][0],
                pdf_results['metadatas'][0]
            ):
                filename = metadata.get("filename", "unknown.pdf")
                page_number = metadata.get("page_number", "unknown")
                formatted_doc = f"Document: {filename}, Page {page_number}\n{doc}"
                all_documents.append(formatted_doc)
                all_metadatas.append(metadata)

        # Process CSV results
        if csv_results['documents'] and csv_results['documents'][0]:
            for doc, metadata in zip(
                csv_results['documents'][0],
                csv_results['metadatas'][0]
            ):
                filename = metadata.get("filename", "unknown.csv")
                row = metadata.get("row", "unknown")
                formatted_doc = f"Data: {filename}, Row {row}\n{doc}"
                all_documents.append(formatted_doc)
                all_metadatas.append(metadata)

        if not all_documents:
            return {
                "answer": "Sorry, no relevant document content was found to answer your question.",
                "context": "",
                "pdf_results": pdf_results,
                "csv_results": csv_results
            }

        # Build context
        context = "\n\n---\n\n".join(all_documents)

        print(f"Number of context documents: {len(all_documents)}")
        print(f"Merged context length: {len(context)}")

        # Generate answer
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "query": query
        })

        # Print debug info before returning
        self._print_debug_info(query, pdf_results, csv_results, all_documents)

        return {
            "answer": response.content,
            "context": context,
            "pdf_results": pdf_results,
            "csv_results": csv_results
        }
    
    def _pdf_retrieval(self, query_embedding: List[float]) -> Dict[str, Any]:
        """Retrieve PDF documents."""
        return self.db_manager.pdf_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.pdf_k,
            include=['documents', 'metadatas', 'distances']
        )
    
    def _csv_retrieval(self, query_embedding: List[float]) -> Dict[str, Any]:
        """Retrieve CSV documents."""
        return self.db_manager.csv_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.csv_k,
            include=['documents', 'metadatas', 'distances']
        )
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.db_manager.get_collection_stats()

    def _print_debug_info(self, query, pdf_results, csv_results, all_documents):
        """Display debug information for the RAG retrieval."""
        print(f"\n{'='*60}")
        print(f" RAG Retrieval Debug Info")
        print(f"{'='*60}")

        # Query information
        print(f" Query: '{query}'")

        # PDF retrieval results
        print(f"\n--- PDF Retrieval Results ---")
        if pdf_results['documents'] and pdf_results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                pdf_results['documents'][0],
                pdf_results['metadatas'][0],
                pdf_results['distances'][0]
            )):
                similarity = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                print(f"📄 PDF {i+1}:")
                print(f"    📏 Distance: {distance:.4f}")
                print(f"    Similarity: {similarity:.4f} ({similarity*100:.1f}%)")
                print(f"    File: {metadata.get('filename', 'unknown')}")
                print(f"    Page: {metadata.get('page_number', 'unknown')}")
                print(f"     Content preview: {doc[:100]}...")
                print()
        else:
            print("❌ No PDF results found")

        # CSV retrieval results
        print(f"\n--- CSV Retrieval Results ---")
        if csv_results['documents'] and csv_results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                csv_results['documents'][0],
                csv_results['metadatas'][0],
                csv_results['distances'][0]
            )):
                similarity = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                print(f"📊 CSV {i+1}:")
                print(f"    📏 Distance: {distance:.4f}")
                print(f"    Similarity: {similarity:.4f} ({similarity*100:.1f}%)")
                print(f"    File: {metadata.get('filename', 'unknown')}")
                print(f"    Row: {metadata.get('row', 'unknown')}")
                print(f"     Content preview: {doc[:100]}...")
                print()
        else:
            print("❌ No CSV results found")

        print(f"📚 Total retrieved documents: {len(all_documents)}")
        print(f"{'='*60}\n")

