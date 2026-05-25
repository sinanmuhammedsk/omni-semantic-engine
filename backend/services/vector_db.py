from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer
from ..core.config import settings
from typing import List, Dict, Any, Optional
import uuid
import os
import logging

logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self):
        self._client: Optional[QdrantClient] = None
        self._encoder: Optional[SentenceTransformer] = None

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return self._encoder

    @property
    def client(self):
        if self._client is None:
            if settings.QDRANT_PATH == ":memory:":
                try:
                    self._client = QdrantClient(":memory:")
                    self._ensure_collection()
                except Exception as e:
                    logger.error(f"Error connecting to in-memory Qdrant: {e}")
            else:
                # Ensure storage directory exists
                if not os.path.exists(settings.QDRANT_PATH):
                    os.makedirs(settings.QDRANT_PATH, exist_ok=True)
                
                try:
                    self._client = QdrantClient(path=settings.QDRANT_PATH)
                    self._ensure_collection()
                except Exception as e:
                    print(f"Error connecting to Qdrant: {e}")
                    # Fallback to memory if local folder is locked (for dev/test)
                    print("Falling back to in-memory storage...")
                    self._client = QdrantClient(":memory:")
                    self._ensure_collection()
        return self._client

    def _ensure_collection(self):
        try:
            collections = self._client.get_collections().collections
            exists = any(c.name == settings.QDRANT_COLLECTION_NAME for c in collections)
            
            if not exists:
                vector_size = self.encoder.get_sentence_embedding_dimension()
                self._client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=rest_models.VectorParams(
                        size=vector_size,
                        distance=rest_models.Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"Collection check failed: {e}")

    def upsert_chunks(self, document_id: int, chunks: List[Dict[str, Any]]):
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.encoder.encode(texts).tolist()

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            points.append(rest_models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "text": chunk['text'],
                    "page_number": chunk['page_number'],
                    "filename": chunk['filename'],
                    "chunk_id": f"doc_{document_id}_ch_{i}"
                }
            ))

        self.client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.encoder.encode(query).tolist()
        
        # Try standard search method
        try:
            results = self.client.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k
            )
        except AttributeError:
            # Fallback for older or specific versions of local client
            logger.error("QdrantClient.search not found, attempting query_points")
            results = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=query_vector,
                limit=top_k
            ).points
        
        return [
            {
                "text": hit.payload["text"],
                "page_number": hit.payload["page_number"],
                "filename": hit.payload["filename"],
                "chunk_id": hit.payload["chunk_id"],
                "score": hit.score
            }
            for hit in results
        ]

    def clear_all(self):
        try:
            self.client.delete_collection(settings.QDRANT_COLLECTION_NAME)
            self._ensure_collection()
            logger.info("Cleared Qdrant collection")
        except Exception as e:
            logger.error(f"Error clearing Qdrant collection: {e}")
            raise e

vector_db = VectorDBService()
