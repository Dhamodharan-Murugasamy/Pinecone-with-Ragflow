import logging
import time
import os
import json
import pinecone

from rag.utils import singleton
from rag.settings import PAGERANK_FLD
from rag.utils.doc_store_conn import DocStoreConnection, MatchExpr, MatchDenseExpr, MatchTextExpr

ATTEMPT_TIME = 2
logger = logging.getLogger('ragflow.pinecone_conn')


@singleton
class PineconeConnection(DocStoreConnection):
    def __init__(self):
        self.index = None
        logger.info(f"Connecting to Pinecone...")
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("Pinecone index name must be provided in the environment variables.")
        self.index = pinecone.Index(index_name)
        logger.info(f"Pinecone connected to index: {index_name}")

    """
    Database operations
    """

    def dbType(self) -> str:
        return "pinecone"

    def health(self) -> dict:
        # Pinecone does not have a direct health endpoint
        return {"type": "pinecone", "status": "connected" if self.index else "disconnected"}

    """
    CRUD operations
    """

    def insert(self, documents: list[dict], indexName: str = None, knowledgebaseId: str = None) -> list[str]:
        """Inserts documents into the Pinecone index."""
        items = []
        for doc in documents:
            assert "id" in doc and "vector" in doc
            metadata = {k: v for k, v in doc.items() if k not in ["id", "vector"]}
            items.append((doc["id"], doc["vector"], metadata))

        for _ in range(ATTEMPT_TIME):
            try:
                self.index.upsert(vectors=items)
                return [doc["id"] for doc in documents]
            except Exception as e:
                logger.warning(f"Pinecone.insert got exception: {str(e)}")
                time.sleep(3)
        raise Exception("Pinecone.insert failed after multiple attempts.")

    def delete(self, condition: dict, indexName: str = None, knowledgebaseId: str = None) -> int:
        """Deletes items based on conditions."""
        if "id" in condition:
            ids = condition["id"] if isinstance(condition["id"], list) else [condition["id"]]
            self.index.delete(ids=ids)
            return len(ids)
        raise NotImplementedError("Delete by condition other than ID is not supported in Pinecone.")

    def search(
        self, selectFields: list[str], highlightFields: list[str], condition: dict,
        matchExprs: list[MatchExpr], orderBy: None, offset: int, limit: int,
        indexNames: str | list[str], knowledgebaseIds: list[str],
        aggFields: list[str] = [], rank_feature: dict | None = None
    ) -> list[dict]:
        """
        Performs a vector search using Pinecone.
        """
        query_vector = None
        filters = {}

        for expr in matchExprs:
            if isinstance(expr, MatchDenseExpr):
                query_vector = expr.embedding_data
            elif isinstance(expr, MatchTextExpr):
                # Pinecone does not directly support text matching.
                logger.warning("Text matching is not natively supported in Pinecone; ignoring MatchTextExpr.")
        
        if not query_vector:
            raise ValueError("A MatchDenseExpr with a query vector is required for Pinecone search.")

        if condition:
            filters = {k: {"$eq": v} for k, v in condition.items()}

        try:
            results = self.index.query(
                vector=query_vector,
                filter=filters,
                top_k=limit,
                include_metadata=True,
            )
            return [
                {
                    "id": match["id"],
                    **match.get("metadata", {}),
                    "score": match["score"]
                }
                for match in results.get("matches", [])
            ]
        except Exception as e:
            logger.exception(f"Pinecone.search failed: {e}")
            return []

    def get(self, chunkId: str, indexName: str = None, knowledgebaseIds: list[str] = None) -> dict | None:
        """Fetch a single document by its ID."""
        try:
            result = self.index.fetch(ids=[chunkId])
            if result and result.get("vectors"):
                return result["vectors"][chunkId]
            return None
        except Exception as e:
            logger.exception(f"Pinecone.get({chunkId}) failed: {e}")
            return None

    """
    Helper functions
    """

    def getTotal(self, res):
        return len(res)

    def getChunkIds(self, res):
        return [d["id"] for d in res]

    def getFields(self, res, fields: list[str]) -> dict[str, dict]:
        return {r["id"]: {f: r.get(f) for f in fields} for r in res}

    def getHighlight(self, res, keywords: list[str], fieldnm: str):
        raise NotImplementedError("Highlighting is not supported in Pinecone.")

    def getAggregation(self, res, fieldnm: str):
        raise NotImplementedError("Aggregations are not supported in Pinecone.")
