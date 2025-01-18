#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
from datetime import date
from enum import IntEnum, Enum
import logging
from typing import Optional

import pinecone
from rag.nlp import search
from graphrag import search as kg_search
from api.utils import get_base_config, decrypt_database_config
from api.constants import RAG_FLOW_SERVICE_NAME

from rag.utils.doc_store_conn import DocStoreConnection

# Logging configuration for debugging
logging.basicConfig(level=logging.DEBUG)

LIGHTEN = int(os.environ.get("LIGHTEN", "0"))

# LLM settings
LLM = None
LLM_FACTORY = None
LLM_BASE_URL = None
CHAT_MDL = ""
EMBEDDING_MDL = ""
RERANK_MDL = ""
ASR_MDL = ""
IMAGE2TEXT_MDL = ""
API_KEY = None
PARSERS = None
HOST_IP = None
HOST_PORT = None
SECRET_KEY = None

# Database settings
DATABASE_TYPE = os.getenv("DB_TYPE", "mysql")
DATABASE = decrypt_database_config(name=DATABASE_TYPE)

# Authentication settings
AUTHENTICATION_CONF = None
CLIENT_AUTHENTICATION = None
HTTP_APP_KEY = None
GITHUB_OAUTH = None
FEISHU_OAUTH = None

# Document engine
DOC_ENGINE = None
docStoreConn = None

# Retrieval engines
retrievaler = None
kg_retrievaler = None


class PineconeDocStore(DocStoreConnection):
    def __init__(self, index: pinecone.Index):
        self.index = index

    def upsert(self, vector_id, vector_values):
        self.index.upsert([(vector_id, vector_values)])

    def query(self, vector_values, top_k):
        return self.index.query(vector_values, top_k=top_k)

    def describe_index_stats(self):
        return self.index.describe_index_stats()

    def fetch(self, ids, namespace=None):
        return self.index.fetch(ids=ids, namespace=namespace)
    
    def createIdx(self, index_name, **kwargs):
        # Pinecone automatically creates indexes; raise an error if unsupported.
        raise NotImplementedError("Index creation is not supported for Pinecone via this interface.")

    def dbType(self):
        return "Pinecone"

    def delete(self, ids, namespace=None):
        self.index.delete(ids=ids, namespace=namespace)

    def deleteIdx(self, index_name):
        # Pinecone does not support deleting indexes directly from an instance.
        raise NotImplementedError("Index deletion must be managed via Pinecone's dashboard or API.")

    def get(self, ids, namespace=None):
        return self.index.fetch(ids=ids, namespace=namespace)

    def getAggregation(self, query):
        # Aggregations might not be directly supported in Pinecone.
        raise NotImplementedError("Aggregation queries are not supported for Pinecone.")

    def getChunkIds(self, namespace=None):
        raise NotImplementedError("Chunk IDs retrieval is not supported.")

    def getFields(self, namespace=None):
        raise NotImplementedError("Field metadata retrieval is not supported.")

    def getHighlight(self, query):
        raise NotImplementedError("Highlighting is not supported for Pinecone.")

    def getTotal(self):
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)

    def health(self):
        # Pinecone doesn't provide a direct health check API.
        return {"status": "healthy"}  # Replace with actual health-check logic if needed.

    def indexExist(self, index_name):
        # Verify if the index exists.
        return index_name in pinecone.list_indexes()

    def insert(self, records):
        raise NotImplementedError("Bulk insertion is not directly supported.")

    def search(self, query_vector, top_k=10, namespace=None):
        return self.index.query(vector=query_vector, top_k=top_k, namespace=namespace)

    def sql(self, query):
        raise NotImplementedError("SQL is not supported for Pinecone.")

    def update(self, vector_id, vector_values, namespace=None):
        self.index.upsert([(vector_id, vector_values)], namespace=namespace)


class Dealer:
    def __init__(self, dataStore: DocStoreConnection):
        self.dataStore = dataStore


def init_settings():
    global LLM, LLM_FACTORY, LLM_BASE_URL, LIGHTEN, DATABASE_TYPE, DATABASE
    LIGHTEN = int(os.environ.get("LIGHTEN", "0"))
    DATABASE_TYPE = os.getenv("DB_TYPE", "mysql")
    DATABASE = decrypt_database_config(name=DATABASE_TYPE)
    LLM = get_base_config("user_default_llm", {})
    LLM_FACTORY = LLM.get("factory", "Tongyi-Qianwen")
    LLM_BASE_URL = LLM.get("base_url")

    global CHAT_MDL, EMBEDDING_MDL, RERANK_MDL, ASR_MDL, IMAGE2TEXT_MDL
    if not LIGHTEN:
        default_llm = {
            "Tongyi-Qianwen": {"chat_model": "qwen-plus", "embedding_model": "text-embedding-v2", "image2text_model": "qwen-vl-max", "asr_model": "paraformer-realtime-8k-v1"},
            "OpenAI": {"chat_model": "gpt-3.5-turbo", "embedding_model": "text-embedding-ada-002", "image2text_model": "gpt-4-vision-preview", "asr_model": "whisper-1"},
            "Azure-OpenAI": {"chat_model": "gpt-35-turbo", "embedding_model": "text-embedding-ada-002", "image2text_model": "gpt-4-vision-preview", "asr_model": "whisper-1"},
            "BAAI": {"chat_model": "", "embedding_model": "BAAI/bge-large-zh-v1.5", "image2text_model": "", "asr_model": "", "rerank_model": "BAAI/bge-reranker-v2-m3"},
        }

        if LLM_FACTORY:
            CHAT_MDL = default_llm[LLM_FACTORY]["chat_model"] + f"@{LLM_FACTORY}"
            ASR_MDL = default_llm[LLM_FACTORY]["asr_model"] + f"@{LLM_FACTORY}"
            IMAGE2TEXT_MDL = default_llm[LLM_FACTORY]["image2text_model"] + f"@{LLM_FACTORY}"
        EMBEDDING_MDL = default_llm["BAAI"]["embedding_model"] + "@BAAI"
        RERANK_MDL = default_llm["BAAI"]["rerank_model"] + "@BAAI"

    global API_KEY, PARSERS, HOST_IP, HOST_PORT, SECRET_KEY
    API_KEY = LLM.get("api_key", "")
    PARSERS = LLM.get("parsers", "naive:General,qa:Q&A,...")

    HOST_IP = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("host", "127.0.0.1")
    HOST_PORT = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("http_port")
    SECRET_KEY = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("secret_key", str(date.today()))

    global AUTHENTICATION_CONF, CLIENT_AUTHENTICATION, HTTP_APP_KEY, GITHUB_OAUTH, FEISHU_OAUTH
    AUTHENTICATION_CONF = get_base_config("authentication", {})
    CLIENT_AUTHENTICATION = AUTHENTICATION_CONF.get("client", {}).get("switch", False)
    HTTP_APP_KEY = AUTHENTICATION_CONF.get("client", {}).get("http_app_key")
    GITHUB_OAUTH = get_base_config("oauth", {}).get("github")
    FEISHU_OAUTH = get_base_config("oauth", {}).get("feishu")

    global DOC_ENGINE, docStoreConn, retrievaler, kg_retrievaler
    DOC_ENGINE = os.environ.get("DOC_ENGINE", "pinecone").lower()

    if DOC_ENGINE == "pinecone":
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        pinecone_env = os.environ.get("PINECONE_ENVIRONMENT")
        pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

        if not all([pinecone_api_key, pinecone_env, pinecone_index_name]):
            raise EnvironmentError("Pinecone configuration variables are missing.")

        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        pinecone_index = pinecone.Index(pinecone_index_name)
        docStoreConn = PineconeDocStore(pinecone_index)

    elif DOC_ENGINE == "infinity":
        docStoreConn = rag.utils.infinity_conn.InfinityConnection()

    else:
        raise Exception(f"Unsupported doc engine: {DOC_ENGINE}")

    retrievaler = search.Dealer(docStoreConn)
    kg_retrievaler = kg_search.KGSearch(docStoreConn)


class CustomEnum(Enum):
    @classmethod
    def valid(cls, value):
        try:
            cls(value)
            return True
        except ValueError:
            return False

    @classmethod
    def values(cls):
        return [member.value for member in cls.__members__.values()]

    @classmethod
    def names(cls):
        return [member.name for member in cls.__members__.values()]


class RetCode(IntEnum, CustomEnum):
    SUCCESS = 0
    NOT_EFFECTIVE = 10
    EXCEPTION_ERROR = 100
    ARGUMENT_ERROR = 101
    DATA_ERROR = 102
    OPERATING_ERROR = 103
    CONNECTION_ERROR = 105
    RUNNING = 106
    PERMISSION_ERROR = 108
    AUTHENTICATION_ERROR = 109
    UNAUTHORIZED = 401
    SERVER_ERROR = 500
    FORBIDDEN = 403
    NOT_FOUND = 404
