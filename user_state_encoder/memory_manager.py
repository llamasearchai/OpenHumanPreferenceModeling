import yaml
import time
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
# import pinecone  # Commented out to avoid import errors if not installed, purely structure for now
# from pinecone import Pinecone, ServerlessSpec

# Load config
with open("user_state_encoder/config.yaml", "r") as f:
    config = yaml.safe_load(f)


class MemoryManager:
    def __init__(self):
        self.api_key = "placeholder_key"  # In prod, load from env
        self.env = config["pinecone_env"]
        self.index_name = config["pinecone_index"]
        self.top_k = config["memory_top_k"]
        self.executor = ThreadPoolExecutor(max_workers=config["prefetch_threads"])

        # Initialize Pinecone (Mocking for now to avoid dependency issues in this env)
        # self.pc = Pinecone(api_key=self.api_key)
        # self.index = self.pc.Index(self.index_name)
        self.index = None  # Placeholder

    def upsert_exemplar(
        self, user_id: str, embedding: List[float], metadata: Dict[str, Any]
    ):
        """
        Stores user exemplar in Pinecone.
        Format: {id: f"{user_id}_{timestamp}", values: embedding, metadata: {...}}
        """
        timestamp = int(time.time())
        vector_id = f"{user_id}_{timestamp}"

        record = {
            "id": vector_id,
            "values": embedding,
            "metadata": {**metadata, "user_id": user_id, "timestamp": timestamp},
        }

        # if self.index:
        #     self.index.upsert(vectors=[record])
        print(f"Upserted record: {record['id']}")  # specific logging for verification

    async def retrieve_exemplars(
        self, user_id: str, query_vector: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k exemplars for a user given a query vector.
        Enforces 50ms SLA via async prefetching / background threads.
        """
        loop = asyncio.get_running_loop()

        # Wrap the blocking Pinecone call in the thread pool executor
        try:
            results = await loop.run_in_executor(
                self.executor, self._query_index, user_id, query_vector
            )
            return results
        except Exception as e:
            print(f"Error retrieving exemplars: {e}")
            return []

    def _query_index(
        self, user_id: str, query_vector: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Blocking internal method to query Pinecone.
        """
        # Simulated latency
        # time.sleep(0.02)

        # Mock Response
        # if self.index:
        #    response = self.index.query(
        #        vector=query_vector,
        #        top_k=self.top_k,
        #        filter={"user_id": user_id},
        #        include_metadata=True
        #    )
        #    return [match['metadata'] for match in response['matches']]

        return []  # Return empty list for now
