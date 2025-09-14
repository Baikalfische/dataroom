"""
RAG Tool for Real Estate Document Dataroom
Enables agents to search and query PDF and CSV documents
"""

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, Any
import sys
from pathlib import Path

# Use relative import; no need to modify sys.path

from ..rag.rag_chain import RealEstateRAGChain

class RealEstateRAGInput(BaseModel):
    """Input schema for the Real Estate RAG Tool."""
    query: str = Field(
        description="The user's natural language question about real estate documents, contracts, or property data."
    )

class RealEstateRAGTool(BaseTool):
    """
    A tool that performs Retrieval-Augmented Generation on real estate documents.
    It can answer questions based on PDF contracts, CSV property data, and other real estate documents.
    """
    name: str = "real_estate_document_search"
    description: str = (
        "Use this tool to search and answer questions about real estate documents. "
        "It can retrieve information from PDF contracts, CSV property data, and other documents. "
        "Provide a text 'query' describing what you're looking for. "
        "The tool will search both PDF and CSV documents and provide answers with proper citations."
    )
    args_schema: Type[BaseModel] = RealEstateRAGInput

    # The RAG chain is initialized once and reused
    rag_chain: RealEstateRAGChain = Field(default_factory=RealEstateRAGChain)

    def _run(self, query: str) -> Any:
        """Executes the RAG query on real estate documents."""
        if not self.rag_chain:
            raise RuntimeError("RAG system is not initialized.")
            
        print(f"🔍 Executing Real Estate RAG Tool...")
        print(f"   Query: {query}")

        try:
            # Invoke RAG chain for retrieval
            result = self.rag_chain.invoke({"query": query})
            
            answer = result["answer"]
            context = result.get("context", "")
            pdf_results = result.get("pdf_results", {})
            csv_results = result.get("csv_results", {})
            
            # Extract citation information
            citations = []
            
            # Process PDF citations
            if pdf_results.get('metadatas') and pdf_results['metadatas'][0]:
                for metadata in pdf_results['metadatas'][0]:
                    if metadata:
                        citations.append({
                            "type": "pdf",
                            "filename": metadata.get("filename", "unknown.pdf"),
                            "page_number": metadata.get("page_number", "unknown")
                        })
            
            # Process CSV citations
            if csv_results.get('metadatas') and csv_results['metadatas'][0]:
                for metadata in csv_results['metadatas'][0]:
                    if metadata:
                        citations.append({
                            "type": "csv", 
                            "filename": metadata.get("filename", "unknown.csv"),
                            "row": metadata.get("row", "unknown")
                        })
            
            # Create response metadata
            metadata = {
                "tool_name": self.name,
                "query": query,
                "status": "completed",
                "answer_length": len(answer),
                "context_length": len(context),
                "source": "real_estate_document_database",
                "citations": citations,
                "pdf_results_count": len(pdf_results.get('documents', [[]])[0]) if pdf_results.get('documents') else 0,
                "csv_results_count": len(csv_results.get('documents', [[]])[0]) if csv_results.get('documents') else 0
            }
            
            print("✅ Real Estate RAG Tool execution finished successfully.")
            print(f"📄 Found {metadata['pdf_results_count']} PDF results")
            print(f"📊 Found {metadata['csv_results_count']} CSV results")
            print(f"📝 Generated answer with {len(answer)} characters")
            
            # Return answer and metadata
            return answer, metadata
        
        except Exception as e:
            print(f"❌ Real Estate RAG Tool execution failed: {e}")
            error_metadata = {
                "tool_name": self.name,
                "query": query,
                "status": "failed",
                "error": str(e)
            }
            return f"Real estate document search failed with error: {str(e)}", error_metadata

    async def _arun(self, query: str) -> Any:
        """Asynchronously executes the RAG query."""
        # In a real high-concurrency scenario, rag_chain.invoke should be made asynchronous.
        # For now we wrap the sync version.
        return self._run(query=query)

    def get_database_stats(self) -> dict:
        """Get database statistics information."""
        try:
            return self.rag_chain.get_database_stats()
        except Exception as e:
            return {"error": f"Failed to get database stats: {str(e)}"}

