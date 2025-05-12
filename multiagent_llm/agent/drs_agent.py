from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from overrides import override
from qdrant_client import qdrant_client

from .agent import Agent


class DRS_Agent(Agent):
    def __init__(self, source: str, k=5):
        super().__init__()
        self.__source = source
        documents = SimpleDirectoryReader(self.__source).load_data()
        client = qdrant_client.QdrantClient(
            host="localhost",
            port=6333
        )
        vector_store = QdrantVectorStore(client=client, collection_name=source.split("/")[-1])
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        self.__retriever = index.as_retriever(similarity_top_k=k)

    @override
    def act(self, state: str) -> str:
        aid = self.__retriever.retrieve()
        out = "Context information is below.\n---------------------\n"
        for item in aid:
            out += item.text + "\n"
        out += "---------------------\n"
        out += "Given the context information, answer the prompt.\nQuery: " + state
        return out
