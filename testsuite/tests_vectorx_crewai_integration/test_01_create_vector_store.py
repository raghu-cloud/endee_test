import pytest
import logging
import sys
import os
import time
from dotenv import load_dotenv
from vecx.vectorx import VectorX
from testsuite.vecx_crewai.base import VectorXVectorStore
from testsuite.vecx_crewai.hugging_face import HuggingFaceEmbedder
import builtins


load_dotenv()
VECTORX_API_TOKEN = getattr(builtins, "VECTORX_API_KEY", None)
if VECTORX_API_TOKEN == None:
    VECTORX_API_TOKEN = os.getenv("VECTORX_API_TOKEN")


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
        cls.vx = VectorX(token=VECTORX_API_TOKEN)
        cls.encryption_key = cls.vx.generate_key()
        cls.embedder_config = {
            "provider": "custom",  # Tell CrewAI to use your embedder
            "config": {
                "embedder": HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
            }
        }
        cls.dimension = 384
        index_lst = cls.vx.list_indexes()
        if len(index_lst['indixes'])>0:
            for index in index_lst['indixes']:
                cls.vx.delete_index(index['name'])
        cls.cleanup_indexes = []

    @classmethod
    def teardown_class(cls):
        # Runs once after all tests in this class
        index_lst = cls.vx.list_indexes()
        if len(index_lst['indixes'])>0:
            for index in index_lst['indixes']:
                cls.vx.delete_index(index['name'])
        logger.info("Create vector store tests done")


    def setup_method(self):
        """Setup before each test"""
        self.test_index_name = "crewai_test_normal_index"
        time.sleep(2)

    def teardown_method(self):
        """Cleanup after each test"""
        index_lst = self.vx.list_indexes()
        print("Length", len(index_lst['indixes']))
        if len(index_lst['indixes'])==5:
            for index_name in self.cleanup_indexes:
                print("Indixes to be deleted", self.cleanup_indexes)
                try:
                    self.vx.delete_index(index_name)
                    logger.info(f"Deleted index: {index_name}")
                    print("DELETED")
                except Exception as e:
                    logger.warning(f"Failed to delete index '{index_name}': {e}")
            self.cleanup_indexes.clear()

    def test_create_vector_store_missing_parameters(self):
        """Test that missing required parameters raise TypeError with correct messages."""
        # Test missing 'type'
        with pytest.raises(TypeError) as exc_info:
            # Initialize the VectorX vector store
            memory_storage = VectorXVectorStore(
                api_token=VECTORX_API_TOKEN,
                embedder_config=self.embedder_config,
                encryption_key=self.encryption_key
            )
        assert "missing 1 required positional argument: 'type'"  in str(exc_info.value)


        # Test missing 'api_token'
        with pytest.raises(ValueError) as exc_info:
            # Initialize the VectorX vector store
            memory_storage = VectorXVectorStore(
                type="test_crewai_index1",
                embedder_config=self.embedder_config,
                encryption_key=self.encryption_key
            )
        assert "API token must be provided"  in str(exc_info.value)


        # Test missing 'embedder_config'
        with pytest.raises(ValueError) as exc_info:
            # Initialize the VectorX vector store
            memory_storage = VectorXVectorStore(
                type="test_crewai_index2",
                api_token=VECTORX_API_TOKEN,
                encryption_key=self.encryption_key
            )
        assert "Please provide an embedder configuration"  in str(exc_info.value)

        # Test missing 'encryption_key'
        with pytest.raises(ValueError) as exc_info:
            # Initialize the VectorX vector store
            memory_storage = VectorXVectorStore(
                type="test_crewai_index3",
                api_token=VECTORX_API_TOKEN,
                embedder_config=self.embedder_config,
            )
        assert "Encryption key must be provided"  in str(exc_info.value)


    def test_create_vector_store_all_parameters_provided(self):
        memory_storage = VectorXVectorStore(
            type=self.test_index_name,
            api_token=VECTORX_API_TOKEN,
            embedder_config=self.embedder_config,
            encryption_key=self.encryption_key
        )

        assert memory_storage.type == self.test_index_name


