from config import QDRANT_HOST,QDRANT_COLLECTION,QDRANT_PORT, DENSE_DIM,TOP_K_DENSE
from app.ingestion.chunker import Chunk
from app.utils.logger import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,VectorParams,SparseVectorParams,SparseIndexParams,PointStruct,SparseVector,NamedVector,NamedSparseVector,QueryRequest
)
from typing import List,Dict,Any
import uuid

DENSE_VECTOR_NAME="dense"
SPARSE_VECTOR_NAME="sparse"

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host= QDRANT_HOST,
            port= QDRANT_PORT,
            timeout = 30,
        )
        self.collection = QDRANT_COLLECTION
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection in existing:
            logger.info(f"Collection '{self.collection}' already exists, skipping create")
            return

        logger.info(f"Creating collection '{self.collection}'")
        self.client.create_collection(
            collection_name      = self.collection,
            vectors_config       = {
                DENSE_VECTOR_NAME: VectorParams(
                    size     = DENSE_DIM,
                    distance = Distance.COSINE,
                    on_disk  = False,
                )
            },
            sparse_vectors_config = {
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index = SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info(f"Collection '{self.collection}' created — dense({DENSE_DIM}d) + sparse")

    def upsert_chunks(
        self,
        chunks:List[Chunk],
        embeddings:List[Dict[str, Any]],
        batch_size:int = 64,
    ) -> int:
        assert len(chunks) == len(embeddings), \
            f"chunks({len(chunks)}) and embeddings({len(embeddings)}) must match"

        points = []
        for chunk, emb in zip(chunks, embeddings):
            sparse_raw= emb["sparse"]
            sparse_indices = [int(k) for k in sparse_raw.keys()]
            sparse_values= [float(v) for v in sparse_raw.values()]

            point = PointStruct(
                id     = str(uuid.uuid4()),
                vector = {
                    DENSE_VECTOR_NAME: emb["dense"].tolist(),
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices = sparse_indices,
                        values= sparse_values,
                    ),
                },
                payload = chunk.to_dict(),
            )
            points.append(point)

        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            self.client.upsert(
                collection_name = self.collection,
                points          = batch,
                wait            = True,
            )
            total += len(batch)
            logger.info(f"Upserted {total}/{len(points)} points")

        logger.info(f"Upsert complete: {total} points in '{self.collection}'")
        return total

    def dense_search(
        self,
        query_dense: list,
        top_k:       int = TOP_K_DENSE,
    ) -> List[Dict]:
        raw = self.client.query_points(
            collection_name = self.collection,
            query           = query_dense,
            using           = DENSE_VECTOR_NAME,
            limit           = top_k,
            with_payload    = True,
        )
        return [{"score": r.score, "payload": r.payload} for r in raw.points]

    def sparse_search(
        self,
        query_sparse: dict,
        top_k:        int = TOP_K_DENSE,
    ) -> List[Dict]:
        indices = [int(k) for k in query_sparse.keys()]
        values= [float(v) for v in query_sparse.values()]

        raw = self.client.query_points(
            collection_name = self.collection,
            query           = SparseVector(indices=indices, values=values),
            using           = SPARSE_VECTOR_NAME,
            limit           = top_k,
            with_payload    = True,
        )
        return [{"score": r.score, "payload": r.payload} for r in raw.points]

    def collection_info(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {
            "vectors_count": info.points_count,
            "status":str(info.status),
        }