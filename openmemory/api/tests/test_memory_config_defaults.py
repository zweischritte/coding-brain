"""
Tests for OpenMemory default memory configuration.
"""

import pytest

from app.utils import memory as memory_utils
from app.routers import config as config_router


VECTOR_STORE_ENV_VARS = [
    "CHROMA_HOST",
    "CHROMA_PORT",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "WEAVIATE_CLUSTER_URL",
    "WEAVIATE_HOST",
    "WEAVIATE_PORT",
    "REDIS_URL",
    "PG_HOST",
    "PG_PORT",
    "MILVUS_HOST",
    "MILVUS_PORT",
    "ELASTICSEARCH_HOST",
    "ELASTICSEARCH_PORT",
    "OPENSEARCH_HOST",
    "OPENSEARCH_PORT",
    "FAISS_PATH",
]


@pytest.fixture(autouse=True)
def clear_vector_store_env(monkeypatch):
    for key in VECTOR_STORE_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    yield


def test_default_memory_config_uses_bge_m3():
    config = memory_utils.get_default_memory_config()

    embedder = config["embedder"]
    assert embedder["provider"] == "ollama"
    assert embedder["config"]["model"] == "bge-m3"
    assert embedder["config"]["embedding_dims"] == 1024

    vector_store = config["vector_store"]
    assert vector_store["provider"] == "qdrant"
    assert vector_store["config"]["collection_name"] == "openmemory_bge_m3"
    assert vector_store["config"]["embedding_model_dims"] == 1024


def test_default_memory_config_uses_qwen3():
    config = memory_utils.get_default_memory_config()

    llm = config["llm"]
    assert llm["provider"] == "ollama"
    assert llm["config"]["model"] == "qwen3:8b"
    assert llm["config"]["ollama_base_url"].startswith("http")


def test_config_defaults_use_bge_m3():
    config = config_router.get_default_configuration()

    embedder = config["mem0"]["embedder"]
    assert embedder["provider"] == "ollama"
    assert embedder["config"]["model"] == "bge-m3"
    assert embedder["config"]["embedding_dims"] == 1024

    llm = config["mem0"]["llm"]
    assert llm["provider"] == "ollama"
    assert llm["config"]["model"] == "qwen3:8b"
