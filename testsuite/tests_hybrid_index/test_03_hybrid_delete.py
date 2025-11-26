import os
import sys
import time
import pytest
import logging
import numpy as np
from dotenv import load_dotenv
from vecx.vectorx import VectorX
from config.test_config import TestConfig
import builtins

load_dotenv()

VECTORX_API_TOKEN = getattr(builtins, "VECTORX_API_KEY", None)
if VECTORX_API_TOKEN is None:
    VECTORX_API_TOKEN = os.getenv("VECTORX_API_TOKEN")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TestHybridDelete:
    @classmethod
    def _get_dim_from_index_name(cls, index_name):
        """Helper function to map index names to dimensions"""
        if 'test_idx_5' in index_name:
            return 5
        elif 'test_idx_768' in index_name:
            return 768
        else:
            return 5  # Fallback if no specific dimension is identified

    @classmethod
    def setup_class(cls):
        cls.vx = VectorX(token=VECTORX_API_TOKEN)
        cls.encryption_key = cls.vx.generate_key()

        # Prepare sample vectors for both dimensions
        cls.vectors_5 = [
            {
                "id": "lifecycle_1",
                "dense_vector": np.random.rand(5).tolist(),
                "sparse_vector": {"indices": [1, 10, 30], "values": [0.8, 0.6, 0.3]},
                "meta": {"title": "Intro to Lifecycle", "lifecycle_stage": "stage_1"}
            },
            {
                "id": "lifecycle_2",
                "dense_vector": np.random.rand(5).tolist(),
                "sparse_vector": {"indices": [5, 20, 25], "values": [0.5, 0.7, 0.6]},
                "meta": {"title": "Middle Stage", "lifecycle_stage": "stage_2"}
            },
            {
                "id": "lifecycle_3",
                "dense_vector": np.random.rand(5).tolist(),
                "sparse_vector": {"indices": [15, 30, 45], "values": [0.9, 0.6, 0.5]},
                "meta": {"title": "End Stage", "lifecycle_stage": "stage_3"}
            }
        ]
        
        cls.vectors_768 = [
            {
                "id": "lifecycle_1",
                "dense_vector": np.random.rand(768).tolist(),
                "sparse_vector": {"indices": [1, 10, 30], "values": [0.8, 0.6, 0.3]},
                "meta": {"title": "Intro to Lifecycle", "lifecycle_stage": "stage_1"}
            },
            {
                "id": "lifecycle_2",
                "dense_vector": np.random.rand(768).tolist(),
                "sparse_vector": {"indices": [5, 20, 25], "values": [0.5, 0.7, 0.6]},
                "meta": {"title": "Middle Stage", "lifecycle_stage": "stage_2"}
            },
            {
                "id": "lifecycle_3",
                "dense_vector": np.random.rand(768).tolist(),
                "sparse_vector": {"indices": [15, 30, 45], "values": [0.9, 0.6, 0.5]},
                "meta": {"title": "End Stage", "lifecycle_stage": "stage_3"}
            }
        ]

    @pytest.mark.parametrize("dimension", [5, 768])  # Parametrize for both dimensions
    @pytest.mark.parametrize("encryption", [True, False])  # Parametrize for both encryption states
    def test_create_and_delete_vectors(self, dimension, encryption):
        # Generate dynamic index name with timestamp and unique ID for each test run
        test_index_name = f"{TestConfig.TEST_INDEX_PREFIX}_{TestConfig.get_timestamp()}_{TestConfig.get_unique_id()}_dim_{dimension}_enc_{'yes' if encryption else 'no'}"
        
        # Select vectors based on dimension
        vectors = self.vectors_5 if dimension == 5 else self.vectors_768
        try:
            # Create index with or without encryption
            if encryption:
                idx = self.vx.create_hybrid_index(
                    name=test_index_name,
                    dimension=dimension,
                    key=self.encryption_key
                )
                index = self.vx.get_hybrid_index(test_index_name, key=self.encryption_key)
            else:
                idx = self.vx.create_hybrid_index(
                    name=test_index_name,
                    dimension=dimension
                )
                index = self.vx.get_hybrid_index(test_index_name)

            # Insert vectors into the index
            index.upsert(vectors)

            # Test: Ensure vectors are inserted
            for vector in vectors:
                vec = index.get_vector(vector["id"])
                assert vec["id"] == vector["id"]
                logger.info(f"Vector '{vector['id']}' exists after insertion.")

            # Test: Ensure vectors can be deleted and others remain unaffected
            index.delete_vector("lifecycle_2")
            for remaining_id in ["lifecycle_1", "lifecycle_3"]:
                vector = index.get_vector(remaining_id)
                assert vector["id"] == remaining_id
                logger.info(f"Vector '{remaining_id}' still exists after deletion.")

            # Test: Ensure deleted vector is not in search results
            dense_query = [0.6, 0.7, 0.8, 0.9, 0.2] + [0.2] * (dimension - 5)
            sparse_query = {
                "indices": [15, 30, 60],
                "values": [0.9, 0.7, 0.5]
            }
            results = index.search(
                dense_vector=dense_query,
                sparse_vector=sparse_query,
                sparse_top_k=10,
                dense_top_k=10,
                include_vectors=False
            )
            ids_in_results = [r["id"] for r in results]
            assert "lifecycle_2" not in ids_in_results, "Deleted vector appeared in search results"
            logger.info(f"Deleted vector 'lifecycle_2' is not in search results for dimension {dimension}.")

        finally:
            # Cleanup after the test
            try:
                self.vx.delete_hybrid_index(test_index_name)
                logger.info(f"Deleted hybrid index: {test_index_name}")
            except Exception as e:
                logger.warning(f"Failed to delete hybrid index: {e}")
            else:
                # If it's a normal index, use delete_index method
                try:
                    self.vx.delete_index(test_index_name)
                    logger.info(f"Deleted index: {test_index_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete index: {e}")

    @pytest.mark.parametrize("dimension", [5, 768])  # Parametrize for both dimensions
    def test_other_vectors_unaffected(self, dimension):
        """Ensure other vectors are still present after deletion"""
        test_index_name = f"{TestConfig.TEST_INDEX_PREFIX}_{TestConfig.get_timestamp()}_{TestConfig.get_unique_id()}_dim_{dimension}"

        vectors = self.vectors_5 if dimension == 5 else self.vectors_768
        try:
            # Create index (without encryption for simplicity)
            idx = self.vx.create_hybrid_index(
                name=test_index_name,
                dimension=dimension
            )
            index = self.vx.get_hybrid_index(test_index_name)
            index.upsert(vectors)

            # Delete lifecycle_2 first (assuming test requires deletion)
            index.delete_vector("lifecycle_2")
            for remaining_id in ["lifecycle_1", "lifecycle_3"]:
                vector = index.get_vector(remaining_id)
                assert vector["id"] == remaining_id
                logger.info(f"Vector '{remaining_id}' still exists after deletion.")

        finally:
            # Cleanup
            try:
                self.vx.delete_hybrid_index(test_index_name)
                logger.info(f"Deleted hybrid index: {test_index_name}")
            except Exception as e:
                logger.warning(f"Failed to delete hybrid index: {e}")
            else:
                # If it's a normal index, use delete_index method
                try:
                    self.vx.delete_index(test_index_name)
                    logger.info(f"Deleted index: {test_index_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete index: {e}")

    @pytest.mark.parametrize("dimension", [5, 768])  # Parametrize for both dimensions
    def test_deleted_vector_not_in_search(self, dimension):
        """Ensure deleted vector doesn't appear in search results"""
        test_index_name = f"{TestConfig.TEST_INDEX_PREFIX}_{TestConfig.get_timestamp()}_{TestConfig.get_unique_id()}_dim_{dimension}"

        vectors = self.vectors_5 if dimension == 5 else self.vectors_768
        try:
            # Create index (without encryption for simplicity)
            idx = self.vx.create_hybrid_index(
                name=test_index_name,
                dimension=dimension
            )
            index = self.vx.get_hybrid_index(test_index_name)
            index.upsert(vectors)

            # Delete lifecycle_2 before searching
            index.delete_vector("lifecycle_2")

            dense_query = [0.6, 0.7, 0.8, 0.9, 0.2] + [0.2] * (dimension - 5)
            sparse_query = {
                "indices": [15, 30, 60],
                "values": [0.9, 0.7, 0.5]
            }

            results = index.search(
                dense_vector=dense_query,
                sparse_vector=sparse_query,
                sparse_top_k=10,
                dense_top_k=10,
                include_vectors=False
            )

            ids_in_results = [r["id"] for r in results]
            assert "lifecycle_2" not in ids_in_results, "Deleted vector appeared in search results"
            logger.info(f"Deleted vector 'lifecycle_2' is not in search results for dimension {dimension}.")

        finally:
            # Cleanup
            try:
                self.vx.delete_hybrid_index(test_index_name)
                logger.info(f"Deleted hybrid index: {test_index_name}")
            except Exception as e:
                logger.warning(f"Failed to delete hybrid index: {e}")
            else:
                # If it's a normal index, use delete_index method
                try:
                    self.vx.delete_index(test_index_name)
                    logger.info(f"Deleted index: {test_index_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete index: {e}")

    @pytest.mark.parametrize("dimension", [5, 768])  # Parametrize for both dimensions
    def test_delete_nonexistent_vector_raises(self, dimension):
        """Deleting nonexistent hybrid vector raises an exception or handles it gracefully"""
        test_index_name = f"{TestConfig.TEST_INDEX_PREFIX}_{TestConfig.get_timestamp()}_{TestConfig.get_unique_id()}_dim_{dimension}"

        vectors = self.vectors_5 if dimension == 5 else self.vectors_768
        try:
            # Create index (without encryption for simplicity)
            idx = self.vx.create_hybrid_index(
                name=test_index_name,
                dimension=dimension
            )
            index = self.vx.get_hybrid_index(test_index_name)
            index.upsert(vectors)

            # Try deleting a nonexistent vector
            try:
                index.delete_vector("nonexistent_id_999")
                logger.info("Deletion of nonexistent vector silently succeeded.")
            except Exception as exc:
                assert "not found" in str(exc).lower(), f"Expected 'not found' error, got: {str(exc)}"

        finally:
            # Cleanup
            try:
                self.vx.delete_hybrid_index(test_index_name)
                logger.info(f"Deleted hybrid index: {test_index_name}")
            except Exception as e:
                logger.warning(f"Failed to delete hybrid index: {e}")
            else:
                # If it's a normal index, use delete_index method
                try:
                    self.vx.delete_index(test_index_name)
                    logger.info(f"Deleted index: {test_index_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete index: {e}")

    def test_cleanup_index_and_verify_removal(self):
        """Delete the test index and verify it’s removed"""
        test_index_name = f"{TestConfig.TEST_INDEX_PREFIX}_{TestConfig.get_timestamp()}_{TestConfig.get_unique_id()}_dim_5"
        self.vx.create_hybrid_index(
            name=test_index_name,
            dimension=5
        )
        self.vx.delete_hybrid_index(test_index_name)
        
        with pytest.raises(Exception):
            self.vx.get_hybrid_index(test_index_name)
        logger.info("Hybrid index successfully deleted.")
