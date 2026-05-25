import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .vector_db import vector_db
import tempfile

class IngestionService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def process_pdf(self, file_path: str, filename: str, document_id: int):
        try:
            # 1. Purge any existing vector indices to maintain absolute context isolation
            vector_db.clear_all()

            # 2. Purge existing SQL metadata from the DB except the current document ID
            from ..db.base import SessionLocal
            from ..db.models import DocumentMetadata
            
            db = SessionLocal()
            try:
                db.query(DocumentMetadata).filter(DocumentMetadata.id != document_id).delete()
                db.commit()
            except Exception as db_err:
                db.rollback()
                import logging
                logging.getLogger(__name__).error(f"Error purging document metadata: {db_err}")
            finally:
                db.close()

            # Load PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            if not pages:
                raise ValueError("PDF is empty or could not be parsed.")

            # Extract chunks
            chunks_to_upsert = []
            for i, page in enumerate(pages):
                page_content = page.page_content
                if not page_content.strip():
                    continue # Skip empty pages (scanned or image-only without OCR)
                
                # Split page content into smaller chunks
                page_chunks = self.text_splitter.split_text(page_content)
                
                for chunk_text in page_chunks:
                    chunks_to_upsert.append({
                        "text": chunk_text,
                        "page_number": i + 1,
                        "filename": filename
                    })

            if not chunks_to_upsert:
                raise ValueError("No text content found in the PDF. It might be a scanned document without OCR.")

            # Upsert to Vector DB
            vector_db.upsert_chunks(document_id, chunks_to_upsert)
            
            return len(pages), len(chunks_to_upsert)

        except Exception as e:
            # Raise exception to be handled by the caller (API)
            raise e

ingestion_service = IngestionService()
