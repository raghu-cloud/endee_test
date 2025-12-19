import pytest
import requests
import logging
import sys
import os
import time
from dotenv import load_dotenv
from endee.endee_client import Endee
from endee_langchain import EndeeVectorStore
from endee.exceptions import APIException
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings 
import builtins

load_dotenv()
ENDEE_API_TOKEN = getattr(builtins, "ENDEE_API_KEY", None)
if ENDEE_API_TOKEN == None:
    ENDEE_API_TOKEN = os.getenv("ENDEE_API_TOKEN")


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # or DEBUG

if not logger.hasHandlers():  # Prevent duplicate handlers
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TestCreateVectorStore:
    @classmethod
    def setup_class(cls):
        cls.nd = Endee(token=ENDEE_API_TOKEN)

        cls.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        cls.dimension = 384
        index_lst = cls.nd.list_indexes()
        if len(index_lst['indixes'])>0:
            for index in index_lst['indixes']:
                cls.nd.delete_index(index['name'])
        cls.cleanup_indexes = []


    @classmethod
    def teardown_class(cls):
        # Runs once after all tests in this class
        index_lst = cls.nd.list_indexes()
        if len(index_lst['indixes'])>0:
            for index in index_lst['indixes']:
                cls.nd.delete_index(index['name'])
        logger.info("Create vector store tests done")

    def setup_method(self):
        """Setup before each test"""
        self.test_index_name = "langchain_test_normal_index"
        time.sleep(2)

    def teardown_method(self):
        """Cleanup after each test"""
        index_lst = self.nd.list_indexes()
        print("Length", len(index_lst['indixes']))
        if len(index_lst['indixes'])>0:
            for index_name in self.cleanup_indexes:
                print("Indixes to be deleted", self.cleanup_indexes)
                try:
                    self.nd.delete_index(index_name)
                    logger.info(f"Deleted index: {index_name}")
                    print("DELETED")
                except Exception as e:
                    logger.warning(f"Failed to delete index '{index_name}': {e}")
            self.cleanup_indexes.clear()


    def test_create_vector_store_from_params_missing_parameters(self):
        """Test that EndeeVectorStore.from_params() raises correct errors when required parameters are missing."""

        # 1. Missing 'embedding'
        with pytest.raises(TypeError) as exc_info:
            EndeeVectorStore.from_params(
                api_token=ENDEE_API_TOKEN,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine",
            )
        assert "missing 1 required positional argument: 'embedding'" in str(exc_info.value)

        # 2. Missing 'api_token'
        with pytest.raises(TypeError) as exc_info:
            EndeeVectorStore.from_params(
                embedding=self.embedding_model,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine",
            )
        assert "missing 1 required positional argument: 'api_token'" in str(exc_info.value)

        # 3. Missing 'index_name'
        with pytest.raises(TypeError) as exc_info:
            EndeeVectorStore.from_params(
                embedding=self.embedding_model,
                api_token=ENDEE_API_TOKEN,
                dimension=128,
                space_type="cosine",
            )
        assert "missing 1 required positional argument: 'index_name'" in str(exc_info.value)

        # 4. Missing 'dimension'
        with pytest.raises(ValueError) as exc_info:
            EndeeVectorStore.from_params(
                embedding=self.embedding_model,
                api_token=ENDEE_API_TOKEN,
                index_name=self.test_index_name,
                space_type="cosine",
            )
        assert "Dimension must be explicitly provided when creating a new index" in str(exc_info.value)

        # 5. Invalid space_type
        with pytest.raises(ValueError) as exc_info:
            EndeeVectorStore.from_params(
                embedding=self.embedding_model,
                api_token=ENDEE_API_TOKEN,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="invalid_space",
            )
        assert "Invalid space type: invalid_space. Must be 'cosine', 'l2', or 'ip'" in str(exc_info.value)


        # 6. Missing 'space_type' → should NOT raise an error (defaults to "cosine")
        vector_store = EndeeVectorStore.from_params(
            embedding=self.embedding_model,
            api_token=ENDEE_API_TOKEN,
            index_name=self.test_index_name,
            dimension=self.dimension,
        )
        # Assert store is created
        assert vector_store is not None

        if vector_store is not None:
            self.cleanup_indexes.append(self.test_index_name)


    def test_create_vector_store_invalid_parameters(self):
        """Test invalid parameter values for EndeeVectorStore.from_params()."""

        # 1. Invalid dimension (0 or negative)
        with pytest.raises(APIException) as exc_info:
            EndeeVectorStore.from_params(
                embedding=self.embedding_model,
                api_token=ENDEE_API_TOKEN,
                index_name=self.test_index_name,
                dimension=0,
                space_type="cosine",
            )
        assert "Dimension must be between 2 and 4096" in str(exc_info.value)

        with pytest.raises(APIException) as exc_info:
            EndeeVectorStore.from_params(
                embedding=self.embedding_model,
                api_token=ENDEE_API_TOKEN,
                index_name=self.test_index_name,
                dimension=-10,
                space_type="cosine",
            )
        assert "Dimension must be between 2 and 4096" in str(exc_info.value)

        # 2. Invalid space_type
        with pytest.raises(ValueError) as exc_info:
            EndeeVectorStore.from_params(
                embedding=self.embedding_model,
                api_token=ENDEE_API_TOKEN,
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="invalid_metric",
            )
        assert "Invalid space type" in str(exc_info.value)


        # 3. Invalid index_name
        with pytest.raises(ValueError) as exc_info:
            EndeeVectorStore.from_params(
                embedding=self.embedding_model,
                api_token=ENDEE_API_TOKEN,
                index_name=" ",
                dimension=self.dimension,
                space_type="cosine",
            )
        assert "Invalid index name" in str(exc_info.value)

        # 4. Invalid API token
        with pytest.raises(requests.exceptions.ConnectionError) as exc_info:
            EndeeVectorStore.from_params(
                embedding=self.embedding_model,
                api_token="invalid_api_token",
                index_name=self.test_index_name,
                dimension=self.dimension,
                space_type="cosine",
            )
        assert "Max retries exceeded with url" in str(exc_info.value)

    def test_create_vector_store_valid_parameters(self):
        """Test creating Endee vector store with valid parameters."""

        vector_store = EndeeVectorStore.from_params(
            embedding=self.embedding_model,
            api_token=ENDEE_API_TOKEN,
            index_name=self.test_index_name,
            dimension=self.dimension,
            space_type="cosine",
        )

        # Ensure vector store object was created
        assert vector_store is not None

        # Validate internal Endee index attributes
        assert vector_store._endee_index.name == self.test_index_name
        assert vector_store._endee_index.dimension == self.dimension
        assert vector_store._endee_index.space_type == "cosine"

        # Ensure the index exists in Endee backend
        index = self.nd.get_index(
            name=self.test_index_name,
        )
        assert index is not None

        # Check backend index metadata
        index_info = index.describe()

        assert index_info["dimension"] == self.dimension
        assert index_info["space_type"] == "cosine"



