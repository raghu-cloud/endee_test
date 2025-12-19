import pytest
import logging
import sys
import os
import time
from dotenv import load_dotenv
from endee.endee import Endee
from endee_crewai import EndeeVectorStore
import builtins


load_dotenv()
ENDEE_API_TOKEN = getattr(builtins, "ENDEE_API_KEY", None)
if ENDEE_API_TOKEN == None:
    ENDEE_API_TOKEN = os.getenv("ENDEE_API_TOKEN")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")


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
        cls.embedder_config = {
            "provider": "cohere",  # Tell CrewAI to use your embedder
            "config": {
                "model_name": "small",
                "api_key": COHERE_API_KEY,

            }
        }
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
        self.test_index_name = "crewai_test_normal_index"
        time.sleep(2)

    def teardown_method(self):
        """Cleanup after each test"""
        index_lst = self.nd.list_indexes()
        print("Length", len(index_lst['indixes']))
        if len(index_lst['indixes'])==5:
            for index_name in self.cleanup_indexes:
                print("Indixes to be deleted", self.cleanup_indexes)
                try:
                    self.nd.delete_index(index_name)
                    logger.info(f"Deleted index: {index_name}")
                    print("DELETED")
                except Exception as e:
                    logger.warning(f"Failed to delete index '{index_name}': {e}")
            self.cleanup_indexes.clear()

    def test_create_vector_store_missing_parameters(self):
        """Test behavior when required parameters are missing."""

        # 1. Missing 'type' → Python TypeError
        with pytest.raises(TypeError) as exc_info:
            EndeeVectorStore(
                api_token=ENDEE_API_TOKEN,
                embedder_config=self.embedder_config,
            )
        assert "missing 1 required positional argument: 'type'" in str(exc_info.value)

        # 2. Missing 'api_token' → Expect: "API token must be provided if endee_index is not provided"
        with pytest.raises(ValueError) as exc_info:
            EndeeVectorStore(
                type="test_crewai_index1",
                embedder_config=self.embedder_config,
            )
        assert "API token must be provided if endee_index is not provided" in str(exc_info.value)

        # 3. Missing embedder_config → CrewAI throws TypeError ('NoneType' is not subscriptable)
        with pytest.raises(TypeError) as exc_info:
            EndeeVectorStore(
                type="test_crewai_index2",
                api_token=ENDEE_API_TOKEN,
            )
        assert "'NoneType' object is not subscriptable" in str(exc_info.value)

    def test_create_vector_store_all_parameters_provided(self):
        memory_storage = EndeeVectorStore(
            type=self.test_index_name,
            api_token=ENDEE_API_TOKEN,
            embedder_config=self.embedder_config,
        )

        assert memory_storage.type == self.test_index_name
