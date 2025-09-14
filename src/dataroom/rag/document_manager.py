"""
Document Manager for dynamic document operations.
Handles uploading, updating, and deleting documents in vector databases.
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from .build_database import VectorDatabaseManager
from ..tools.parser import parse_pdf, parse_csv

class DocumentManager:
    """
    Document Manager - provides dynamic document operation interfaces.
    Supports uploading, updating, and deleting documents.
    """
    
    def __init__(self):
        """Initialize the document manager."""
        self.db_manager = VectorDatabaseManager()
        print("📚 DocumentManager initialized")
    
    def upload_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Upload a document - automatically detect file type and process it.
        
        Args:
            file_path: Path to the document.
            metadata: Optional additional metadata.
            
        Returns:
            Processing result.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        file_extension = file_path.suffix.lower()
        
    # Choose processing method based on file type
        if file_extension == '.pdf':
            return self.upload_pdf(file_path, metadata)
        elif file_extension == '.csv':
            return self.upload_csv(file_path, metadata)
        else:
            return {"error": f"Unsupported file type: {file_extension}"}
    
    def upload_pdf(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Upload a PDF document."""
        try:
            print(f"📄 Uploading PDF: {Path(file_path).name}")
            
            # Check whether the document already exists
            existing_result = self._check_document_exists(file_path, "pdf")
            if existing_result["exists"]:
                return {"error": f"Document already exists: {existing_result['document_id']}"}
            
            # Add to database
            result = self.db_manager.add_pdf_document(file_path, metadata=metadata)
            
            if "error" not in result:
                print(f"✅ PDF uploaded successfully: {result['pages_added']} pages")
            else:
                print(f"❌ PDF upload failed: {result['error']}")
            
            return result
            
        except Exception as e:
            return {"error": f"Error uploading PDF: {str(e)}"}
    
    def upload_csv(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Upload a CSV document."""
        try:
            print(f"📊 Uploading CSV: {Path(file_path).name}")
            
            # Check whether the document already exists
            existing_result = self._check_document_exists(file_path, "csv")
            if existing_result["exists"]:
                return {"error": f"Document already exists: {existing_result['document_id']}"}
            
            # Add to database
            result = self.db_manager.add_csv_document(file_path, metadata=metadata)
            
            if "error" not in result:
                print(f"✅ CSV uploaded successfully: {result['chunks_added']} chunks")
            else:
                print(f"❌ CSV upload failed: {result['error']}")
            
            return result
            
        except Exception as e:
            return {"error": f"Error uploading CSV: {str(e)}"}
    
    def update_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update a document - delete it first and then re-add.
        
        Args:
            file_path: Path to the document.
            metadata: Optional additional metadata.
            
        Returns:
            Update result.
        """
        try:
            print(f"🔄 Updating document: {Path(file_path).name}")
            
            # First delete existing document
            delete_result = self.delete_document_by_path(file_path)
            if "error" in delete_result and "not found" not in delete_result["error"].lower():
                return delete_result
            
            # Re-upload
            return self.upload_document(file_path, metadata)
            
        except Exception as e:
            return {"error": f"Error updating document: {str(e)}"}
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete document by document ID.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Deletion result.
        """
        try:
            print(f"��️ Deleting document: {document_id}")
            
            # Delete from PDF collection
            pdf_result = self._delete_from_collection(self.db_manager.pdf_collection, document_id)
            
            # Delete from CSV collection
            csv_result = self._delete_from_collection(self.db_manager.csv_collection, document_id)
            
            if pdf_result["deleted"] or csv_result["deleted"]:
                return {
                    "status": "success",
                    "document_id": document_id,
                    "deleted_from_pdf": pdf_result["deleted"],
                    "deleted_from_csv": csv_result["deleted"]
                }
            else:
                return {"error": f"Document not found: {document_id}"}
                
        except Exception as e:
            return {"error": f"Error deleting document: {str(e)}"}
    
    def delete_document_by_path(self, file_path: str) -> Dict[str, Any]:
        """
        Delete document by file path.
        
        Args:
            file_path: File path.
            
        Returns:
            Deletion result.
        """
        filename = Path(file_path).name
        
    # Look in PDF collection
        pdf_docs = self.db_manager.pdf_collection.get(
            where={"filename": filename},
            include=['metadatas']
        )
        
    # Look in CSV collection
        csv_docs = self.db_manager.csv_collection.get(
            where={"filename": filename},
            include=['metadatas']
        )
        
        if pdf_docs['ids']:
            document_id = pdf_docs['metadatas'][0]['document_id']
            return self.delete_document(document_id)
        elif csv_docs['ids']:
            document_id = csv_docs['metadatas'][0]['document_id']
            return self.delete_document(document_id)
        else:
            return {"error": f"Document not found: {filename}"}
    
    def list_documents(self) -> Dict[str, Any]:
        """
        List all documents.
        
        Returns:
            Document list.
        """
        try:
            # Retrieve PDF documents
            pdf_docs = self.db_manager.pdf_collection.get(
                include=['metadatas']
            )
            
            # Retrieve CSV documents
            csv_docs = self.db_manager.csv_collection.get(
                include=['metadatas']
            )
            
            # Process PDF documents
            pdf_documents = {}
            for i, metadata in enumerate(pdf_docs['metadatas']):
                doc_id = metadata['document_id']
                if doc_id not in pdf_documents:
                    pdf_documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata['filename'],
                        "file_type": "pdf",
                        "total_pages": 0,
                        "upload_time": metadata.get('upload_time', 'unknown')
                    }
                pdf_documents[doc_id]["total_pages"] += 1
            
            # Process CSV documents
            csv_documents = {}
            for i, metadata in enumerate(csv_docs['metadatas']):
                doc_id = metadata['document_id']
                if doc_id not in csv_documents:
                    csv_documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata['filename'],
                        "file_type": "csv",
                        "total_rows": 0,
                        "upload_time": metadata.get('upload_time', 'unknown')
                    }
                csv_documents[doc_id]["total_rows"] += 1
            
            return {
                "status": "success",
                "pdf_documents": list(pdf_documents.values()),
                "csv_documents": list(csv_documents.values()),
                "total_pdf": len(pdf_documents),
                "total_csv": len(csv_documents),
                "total_documents": len(pdf_documents) + len(csv_documents)
            }
            
        except Exception as e:
            return {"error": f"Error listing documents: {str(e)}"}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics information."""
        return self.db_manager.get_collection_stats()
    
    def _check_document_exists(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Check if a document already exists."""
        filename = Path(file_path).name
        collection = self.db_manager.pdf_collection if file_type == "pdf" else self.db_manager.csv_collection
        
        existing_docs = collection.get(
            where={"filename": filename},
            include=['metadatas']
        )
        
        if existing_docs['ids']:
            return {
                "exists": True,
                "document_id": existing_docs['metadatas'][0]['document_id']
            }
        else:
            return {"exists": False}
    
    def _delete_from_collection(self, collection, document_id: str) -> Dict[str, Any]:
        """Delete document from specified collection."""
        try:
            # Find all related records
            docs = collection.get(
                where={"document_id": document_id},
                include=['metadatas']
            )
            
            if docs['ids']:
                collection.delete(ids=docs['ids'])
                return {"deleted": True, "count": len(docs['ids'])}
            else:
                return {"deleted": False, "count": 0}
                
        except Exception as e:
            return {"error": f"Error deleting from collection: {str(e)}"}

