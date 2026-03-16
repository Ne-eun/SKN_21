from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from qdrant_client.models import Distance
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "criminal"
QDRANT_API_KEY = os.getenv("QDRANT__SERVICE__API_KEY")


def get_embeddings():
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")
    dense = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=500,
        max_retries=3,
    )
    size = VECTOR_SIZE

    return sparse, dense, size



def get_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


def get_vector_store():
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


def create_collection(documents):
    client = get_client(COLLECTION_NAME)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    sparse, dense, size = get_embeddings()
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=(
            'dense': VectorsConfig(
                size=size,
                distance=Distance.COSINE
            )
        ),
        sparse_vectors_config=(
            'sparse': SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        ),
    )

    return QdrantVectorStore(
        client = get_client(),
        collection_name=COLLECTION_NAME,
        embedding=dense,
        sparse_embedding=sparse,
        size=size,
        distance=Distance.COSINE,
        force_recreate=True,
    )