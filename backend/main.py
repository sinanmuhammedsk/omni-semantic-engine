from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
import tempfile

from .core.config import settings
from .db.base import get_db, init_db
from .db.models import DocumentMetadata, ProcessingStatus
from .schemas.schemas import DocumentResponse, QueryRequest, QueryResponse
from .services.ingestion import ingestion_service
from .services.llm import llm_service

app = FastAPI(title="Enterprise RAG Pipeline")

@app.on_event("startup")
def startup_event():
    init_db()

def process_document_task(document_id: int, file_path: str, filename: str):
    db = next(get_db())
    doc_metadata = db.query(DocumentMetadata).filter(DocumentMetadata.id == document_id).first()
    
    if not doc_metadata:
        return

    try:
        doc_metadata.status = ProcessingStatus.PROCESSING
        db.commit()

        page_count, chunk_count = ingestion_service.process_pdf(file_path, filename, document_id)

        doc_metadata.status = ProcessingStatus.COMPLETED
        doc_metadata.page_count = page_count
        doc_metadata.extra_metadata = {"chunk_count": chunk_count}
        db.commit()
    except Exception as e:
        import traceback
        doc_metadata.status = ProcessingStatus.FAILED
        doc_metadata.error_message = traceback.format_exc()
        db.commit()
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Clear all previous documents and vector memory before processing the new upload
    try:
        db.query(DocumentMetadata).delete()
        db.commit()
        from .services.vector_db import vector_db
        vector_db.clear_all()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear existing documents before upload: {str(e)}")

    doc_metadata = DocumentMetadata(
        filename=file.filename,
        status=ProcessingStatus.PENDING
    )
    db.add(doc_metadata)
    db.commit()
    db.refresh(doc_metadata)

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(process_document_task, doc_metadata.id, temp_file_path, file.filename)

    return doc_metadata

@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)):
    return db.query(DocumentMetadata).all()

@app.delete("/documents")
async def clear_documents(db: Session = Depends(get_db)):
    try:
        db.query(DocumentMetadata).delete()
        db.commit()
        from .services.vector_db import vector_db
        vector_db.clear_all()
        return {"status": "success", "message": "All documents and vector memory cleared."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_pipeline(request: QueryRequest):
    # Returns a stream of text chunks
    return StreamingResponse(
        llm_service.stream_response(request.query, top_k=request.top_k),
        media_type="text/plain"
    )

@app.post("/retrieve")
async def retrieve_chunks(request: QueryRequest):
    try:
        from .services.vector_db import vector_db
        matching_chunks = vector_db.search(request.query, top_k=request.top_k)
        return {"chunks": matching_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
