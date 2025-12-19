import os
import sys
import pytest
import logging
import numpy as np
from dotenv import load_dotenv
from endee import Endee
from config.test_config import TestConfig
import builtins

load_dotenv()  # Load .env vars

ENDEE_API_TOKEN = getattr(builtins, "ENDEE_API_KEY", None)
if ENDEE_API_TOKEN == None:
    ENDEE_API_TOKEN = os.getenv("ENDEE_API_TOKEN")

timestamp = getattr(builtins, "TEST_RUN_TIMESTAMP", None)

# Setup logger to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TestDeleteVectorsAndIndexes:
    @classmethod
    def setup_class(cls):
        """Initialize Endee client and test indexes."""
        cls.nd = Endee(token=ENDEE_API_TOKEN)
        # with open(f"config/pipeline_mode_bool_{timestamp}.txt", "r") as f:
        #     cls.pipeline_mode = f.read().strip().lower() == "true"

        # Load or generate encryption key based on pipeline mode
        # if cls.pipeline_mode:
        #     try:
        #         with open(f"config/tmp_encryption_key_{timestamp}.txt", "r") as f:
        #             cls.encryption_key = f.read().strip()
        #         logger.info("Loaded encryption key from file (pipeline mode).")
        #     except FileNotFoundError:
        #         cls.encryption_key = None
        #         logger.warning("Encryption key file missing.")
        # else:
        #     cls.encryption_key = cls.nd.generate_key()
        #     logger.info("Generated encryption key.")

        # Define index configs
        cls.index_configs = [
            {"name": TestConfig.TEST_UPSERT_INDEX1, "dimension": 5, "encryption": False, "space_type": "cosine"},
            {"name": TestConfig.TEST_UPSERT_INDEX2, "dimension": 768, "encryption": False, "space_type": "l2"},
            # {"name": TestConfig.TEST_UPSERT_INDEX3, "dimension": 5, "encryption": True, "space_type": "ip"},
            # {"name": TestConfig.TEST_UPSERT_INDEX4, "dimension": 768, "encryption": True, "space_type": "cosine"},
        ]

        cls.test_indexes = [{"name": c["name"]} for c in cls.index_configs]

        # If not pipeline mode, clean and setup indexes
        # if not cls.pipeline_mode:
        logger.info("Cleaning and creating test indexes.")

        # Delete leftover indexes
        existing_indexes = cls.nd.list_indexes().get('indixes', [])
        for index in existing_indexes:
            cls.nd.delete_index(index['name'])
            logger.info(f"Deleted leftover index: {index['name']}")

        # Create indexes and upsert vectors
        for config in cls.index_configs:
            kwargs = {
                "name": config["name"],
                "dimension": config["dimension"],
                "space_type": config["space_type"],
            }
            # if config["encryption"]:
            #     kwargs["key"] = cls.encryption_key

            cls.nd.create_index(**kwargs)
            logger.info(f"Created index: {config['name']}")

            index = cls.nd.get_index(config["name"])
            num_vectors = 2000
            vectors = [TestConfig.generate_vector(str(i), config["dimension"], config["space_type"]) for i in range(num_vectors)]

            # Insert vectors in batches of 1000 (no info logging per batch)
            for i in range(0, num_vectors, 1000):
                batch = vectors[i:i + 1000]
                index.upsert(batch)

        logger.info("Indexes and vectors setup complete.")

    @classmethod
    def teardown_class(cls):
        """Cleanup test indexes if pipeline mode."""
        # if cls.pipeline_mode:
        # if not cls.pipeline_mode:
        logger.info("Cleanup: Deleting test indexes.")
        for idx in cls.test_indexes:
            try:
                cls.nd.delete_index(idx["name"])
                logger.info(f"Deleted index {idx['name']} on teardown.")
            except Exception as e:
                logger.warning(f"Could not delete index {idx['name']}: {e}")
            # os.remove(f"config/tmp_encryption_key_{timestamp}.txt")
        # os.remove(f"config/pipeline_mode_bool_{timestamp}.txt")
        logger.info("Teardown complete.")

        """
        Clean up test indexes after all tests.
        """
        # indexes = cls.nd.list_indexes()
        # for idx in indexes.get("indixes", []):
        #     if idx["name"].startswith(TestConfig.TEST_INDEX_PREFIX):
        #         cls.nd.delete_index(idx["name"])
        # logger.info("Deleted all test indexes after test run.")

    def get_test_index(self, name):
        """Fetch index."""
        try:
            return self.nd.get_index(name=name)
        except Exception as e:
            logger.warning(f"Failed to get index '{name}': {e}")
            raise

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        (TestConfig.TEST_UPSERT_INDEX1, 5, "cosine"),
        (TestConfig.TEST_UPSERT_INDEX2, 768, "l2"),
    ])
    def test_delete_vectors_with_invalid_filter(self, index_attr, dimension, space_type):
        '''Deleting vectors with invalid filter should not affect vector count'''
        index = self.get_test_index(index_attr)
        before = index.describe().get("count", 0)
        index.delete_with_filter([{"invalid_field": {"$eq": "invalid_value"}}])
        after = index.describe().get("count", 0)
        logger.info(f"Invalid filter delete on '{index_attr}' — before: {before}, after: {after}")
        assert before == after, f"Vector count changed after invalid delete filter on index {index_attr}"

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        (TestConfig.TEST_UPSERT_INDEX1, 5, "cosine"),
        (TestConfig.TEST_UPSERT_INDEX2, 768, "l2"),
    ])
    def test_delete_nonexistent_vector(self, index_attr, dimension, space_type):
        '''Deleting a nonexistent vector should raise a "not found" error'''
        index = self.get_test_index(index_attr)
        with pytest.raises(Exception) as exc_info:
            index.delete_vector("nonexistent_vector_id_12345")
        assert "not found" in str(exc_info.value).lower(), "Expected 'not found' error message"

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        (TestConfig.TEST_UPSERT_INDEX1, 5, "cosine"),
        (TestConfig.TEST_UPSERT_INDEX2, 768, "l2"),
    ])
    def test_delete_vector(self, index_attr, dimension, space_type):
        '''Upsert a single vector, fetch, delete, and confirm removal'''
        index = self.get_test_index(index_attr)
        test_id_suffix = f"del_{TestConfig.get_unique_id()}"
        vector_data = TestConfig.generate_vector(
            id_suffix=test_id_suffix,
            dim=dimension,
            title="Temp vector for delete test",
            space_type=space_type
        )
        test_id = vector_data["id"]
        index.upsert([vector_data])
        fetched = index.get_vector(test_id)
        assert fetched["id"] == test_id
        index.delete_vector(test_id)
        with pytest.raises(Exception):
            index.get_vector(test_id)

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        (TestConfig.TEST_UPSERT_INDEX1, 5, "cosine"),
        (TestConfig.TEST_UPSERT_INDEX2, 768, "l2"),
    ])
    def test_delete_multiple_vectors(self, index_attr, dimension, space_type):
        '''Batch upsert multiple vectors, then delete them one-by-one'''
        index = self.get_test_index(index_attr)
        num_test_vectors = 5
        vectors = [
            TestConfig.generate_vector(
                id_suffix=TestConfig.get_unique_id(),
                dim=dimension,
                title="Batch Delete Test Vector",
                space_type=space_type
            ) for _ in range(num_test_vectors)
        ]
        index.upsert(vectors)
        for vec in vectors:
            vec_id = vec["id"]
            try:
                assert index.get_vector(vec_id)["id"] == vec_id
            except Exception:
                continue
            index.delete_vector(vec_id)
            with pytest.raises(Exception):
                index.get_vector(vec_id)

    def test_delete_vectors_in_nonexistent_index(self):
        '''Attempting to delete vectors in a nonexistent index should raise error'''
        with pytest.raises(Exception) as exc_info:
            idx = self.nd.get_index("nonexistent_index_123456")
            idx.delete_with_filter({})
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        (TestConfig.TEST_UPSERT_INDEX1, 5, "cosine"),
        (TestConfig.TEST_UPSERT_INDEX2, 768, "l2"),
    ])
    def test_delete_test_indexes(self, index_attr, dimension, space_type):
        '''Delete index and verify deletion (backend does not raise on describe)'''

        # Step 1: Try deleting
        try:
            self.nd.delete_index(index_attr)
            logger.info(f"Deleted index: {index_attr}")
        except Exception as e:
            # Index may already be gone; this is fine
            if "not found" in str(e).lower():
                logger.warning(f"Index already missing: {index_attr}")
            else:
                raise

        # Step 2: Fetch index object (client-side only)
        idx = self.nd.get_index(index_attr)

        # Step 3: Describe should NOT raise — but should give zero elements
        desc = idx.describe()

        assert desc.get("count", None) in (0, None), \
            f"Expected empty describe response after deletion, got: {desc}"



    # def test_fetch_existing_indexes(self):
    #     """Check all test indexes exist."""
    #     for idx in self.test_indexes:
    #         try:
    #             index = self.get_test_index(idx["name"], idx["encrypted"])
    #             assert index is not None, f"Index not found: {idx['name']}"
    #         except Exception:
    #             logger.warning(f"Index missing or failed to fetch: {idx['name']}")

    # def test_describe_index_vectors(self):
    #     """Describe indexes and assert vector count is non-negative."""
    #     for idx in self.test_indexes:
    #         try:
    #             index = self.get_test_index(idx["name"], idx["encrypted"])
    #             count = index.describe().get("count", None)
    #             if not isinstance(count, int) or count < 0:
    #                 logger.warning(f"Invalid vector count for index {idx['name']}: {count}")
    #             else:
    #                 logger.info(f"Index: {idx['name']} | Vector Count: {count}")
    #         except Exception as e:
    #             logger.warning(f"Describe failed for '{idx['name']}': {e}")

    # def test_fetch_nonexistent_index(self):
    #     """Ensure fetching nonexistent index raises error."""
    #     with pytest.raises(Exception) as exc_info:
    #         self.nd.get_index("nonexistent_index_123456")
    #     assert "not found" in str(exc_info.value).lower(), "Expected 'not found' error"

    # def test_describe_nonexistent_index(self):
    #     """Describing nonexistent index raises error."""
    #     with pytest.raises(Exception) as exc_info:
    #         idx = self.nd.get_index("nonexistent_index_123456")
    #         idx.describe()
    #     assert "not found" in str(exc_info.value).lower(), "Expected 'not found' error on describe"

    # def test_fetch_index_with_wrong_encryption_key(self):
    #     """Fetching encrypted index with wrong key fails."""
    #     for idx in self.test_indexes:
    #         if idx["encrypted"]:
    #             with pytest.raises(Exception) as exc_info:
    #                 self.nd.get_index(name=idx["name"], key="wrong_key_123")
    #             logger.info(f"Expected failure for wrong key on '{idx['name']}': {exc_info.value}")

    # def test_describe_index_with_wrong_encryption_key(self):
    #     """Describing encrypted index with wrong key fails."""
    #     for idx in self.test_indexes:
    #         if idx["encrypted"]:
    #             with pytest.raises(Exception) as exc_info:
    #                 index = self.nd.get_index(name=idx["name"], key="wrong_key_123")
    #                 index.describe()
    #             logger.info(f"Expected failure describing '{idx['name']}' with wrong key: {exc_info.value}")
