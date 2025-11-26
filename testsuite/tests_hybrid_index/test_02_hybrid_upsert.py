import pytest
import os
import sys
import time
import random
import json
import numpy as np
from dotenv import load_dotenv
from vecx.vectorx import VectorX
from config.test_config import TestConfig
import logging
import builtins

# Load environment variables
load_dotenv()
VECTORX_API_TOKEN = getattr(builtins, "VECTORX_API_KEY", None) or os.getenv("VECTORX_API_TOKEN")

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TestUpsertHybridVectors:
    @classmethod
    def setup_class(cls):
        cls.vx = VectorX(token=VECTORX_API_TOKEN)
        cls.encryption_key = cls.vx.generate_key()

        # Define index names here
        cls.idx_name_no_enc_5 = f"{TestConfig.TEST_UPSERT_INDEX1}_dim5"
        cls.idx_name_enc_5 = f"{TestConfig.TEST_UPSERT_INDEX3}_dim5"
        cls.idx_name_no_enc_768 = f"{TestConfig.TEST_UPSERT_INDEX1}_dim768"
        cls.idx_name_enc_768 = f"{TestConfig.TEST_UPSERT_INDEX3}_dim768"

        # Cleanup existing indexes
        cls.cleanup_existing_indexes()

        # Create fresh hybrid indexes for testing (5 and 768 dimensions)
        cls.create_index_if_not_exists(cls.idx_name_no_enc_5, dimension=5)
        cls.create_index_if_not_exists(cls.idx_name_enc_5, dimension=5, encryption_key=cls.encryption_key)
        cls.create_index_if_not_exists(cls.idx_name_no_enc_768, dimension=768)
        cls.create_index_if_not_exists(cls.idx_name_enc_768, dimension=768, encryption_key=cls.encryption_key)

    @classmethod
    def cleanup_existing_indexes(cls):
        """Delete existing indexes before setup."""
        index_lst = cls.vx.list_indexes()
        indexes = index_lst.get('indixes', [])
        for index in indexes:
            index_name = index.get('name')
            if not index_name:
                continue
            try:
                if 'vocab_size' in index:
                    cls.vx.delete_hybrid_index(index_name)
                    logger.info(f"Deleted hybrid index (setup): {index_name}")
                else:
                    cls.vx.delete_index(index_name)
                    logger.info(f"Deleted regular index (setup): {index_name}")
            except Exception as delete_err:
                logger.error(f"Failed to delete index during setup '{index_name}': {delete_err}")

    @classmethod
    def create_index_if_not_exists(cls, index_name, dimension, encryption_key=None):
        """Create index if it doesn't already exist."""
        index_lst = cls.vx.list_indexes()
        existing_index_names = [index.get("name") for index in index_lst.get("indixes", [])]

        if index_name not in existing_index_names:
            logger.info(f"Creating index: {index_name} with dimension {dimension}")
            cls.vx.create_hybrid_index(
                name=index_name,
                key=encryption_key,
                dimension=dimension,
                space_type="cosine",
                vocab_size=30522
            )
        else:
            logger.info(f"Index already exists: {index_name}")

    @classmethod
    def teardown_class(cls):
        # Cleanup after tests
        cls.cleanup_existing_indexes()

    def setup_method(self):
        """Fetch hybrid index handles before each test"""
        self.index_no_enc_5 = self.vx.get_hybrid_index(self.idx_name_no_enc_5)
        self.index_enc_5 = self.vx.get_hybrid_index(self.idx_name_enc_5, key=self.encryption_key)
        self.index_no_enc_768 = self.vx.get_hybrid_index(self.idx_name_no_enc_768)
        self.index_enc_768 = self.vx.get_hybrid_index(self.idx_name_enc_768, key=self.encryption_key)

    def _generate_hybrid_vector(self, id_suffix, dim=5):
        """Helper: create hybrid vector with both dense and sparse parts"""
        dense = np.random.rand(dim).tolist()
        sparse = {
            "indices": random.sample(range(100), 4),
            "values": [random.random() for _ in range(4)],
        }
        return {
            "id": f"vec_{id_suffix}",
            "dense_vector": dense,
            "sparse_vector": sparse,
            "meta": {"title": f"Hybrid Vector {id_suffix}"},
        }

    def _safe_get_vector(self, index, vec_id):
        """Helper: safely retrieve a vector"""
        result = index.get_vector(vec_id)
        return json.loads(result) if isinstance(result, str) else result

    @pytest.mark.parametrize("index_attr, encrypted, dimension", [
        ("index_no_enc_5", False, 5),
        ("index_enc_5", True, 5),
        ("index_no_enc_768", False, 768),
        ("index_enc_768", True, 768),
    ])
    def test_upsert_and_retrieve_vector(self, index_attr, encrypted, dimension):
        """Test upserting and retrieving a hybrid vector"""
        index = getattr(self, index_attr)
        hybrid_vec = self._generate_hybrid_vector(f"basic_dim{dimension}", dim=dimension)
        result = index.upsert([hybrid_vec])
        assert "success" in result.lower()

        time.sleep(0.5)
        retrieved = self._safe_get_vector(index, hybrid_vec["id"])
        assert retrieved["id"] == hybrid_vec["id"]
        assert "dense_vector" in retrieved
        assert "sparse_vector" in retrieved
        assert retrieved["meta"]["title"] == hybrid_vec["meta"]["title"]

    @pytest.mark.parametrize("index_attr, encrypted, dimension", [
        ("index_enc_5", True, 5),
        ("index_enc_768", True, 768),
    ])
    def test_encryption_roundtrip_sparse_vector(self, index_attr, encrypted, dimension):
        """Test encryption and decryption of sparse vector"""
        index = getattr(self, index_attr)
        vec = self._generate_hybrid_vector(f"enc_test_dim{dimension}", dim=dimension)
        sparse = vec["sparse_vector"]

        enc_indices, enc_values = index.vxlib.encrypt_sparse_vector(sparse["indices"], sparse["values"])
        dec_indices, dec_values = index.vxlib.decrypt_sparse_vector(enc_indices, enc_values)

        assert sparse["indices"] == dec_indices
        assert pytest.approx(sum(sparse["values"]), rel=1e-6) == pytest.approx(sum(dec_values), rel=1e-6)

    @pytest.mark.parametrize("index_attr, encrypted, dimension", [
        ("index_no_enc_5", False, 5),
        ("index_enc_5", True, 5),
        ("index_no_enc_768", False, 768),
        ("index_enc_768", True, 768),
    ])
    def test_upsert_invalid_dimension(self, index_attr, encrypted, dimension):
        """Test that wrong dimension vector raises error"""
        index = getattr(self, index_attr)
        vec = self._generate_hybrid_vector(f"wrong_dim{dimension}", dim=dimension-1)
        with pytest.raises(Exception) as exc:
            index.upsert([vec])
        assert "dimension" in str(exc.value).lower()


    @pytest.mark.parametrize("index_attr, encrypted, dim", [
        ("index_no_enc_5", False, 5),
        ("index_enc_5", True, 5),
        ("index_no_enc_768", False, 768),
        ("index_enc_768", True, 768),
    ])
    def test_upsert_empty_id(self, index_attr, encrypted, dim):
        """Test that vector with empty ID is rejected"""
        index = getattr(self, index_attr)
        vec = self._generate_hybrid_vector("empty_id", dim=dim)
        vec["id"] = ""  # Empty ID
        with pytest.raises(Exception):
            index.upsert([vec])

    @pytest.mark.parametrize("index_attr, encrypted, dim", [
        ("index_no_enc_5", False, 5),
        ("index_enc_5", True, 5),
        ("index_no_enc_768", False, 768),
        ("index_enc_768", True, 768),
    ])
    def test_upsert_and_update_vector(self, index_attr, encrypted, dim):
        """Test updating an existing hybrid vector's metadata"""
        index = getattr(self, index_attr)
        vec = self._generate_hybrid_vector("update", dim=dim)
        vec["meta"]["title"] = "Original"
        index.upsert([vec])

        # Update with new title
        vec["meta"]["title"] = "Updated"
        result = index.upsert([vec])
        assert "success" in result.lower()

        # Fetch and assert the update
        fetched = self._safe_get_vector(index, vec["id"])
        assert fetched["meta"]["title"] == "Updated"

    @pytest.mark.parametrize("index_attr, encrypted, dim", [
        ("index_no_enc_5", False, 5),
        ("index_enc_5", True, 5),
        ("index_no_enc_768", False, 768),
        ("index_enc_768", True, 768),
    ])
    def test_large_batch_upsert_hybrid_vectors(self, index_attr, encrypted, dim):
        """Test upserting large batch of hybrid vectors in chunks"""
        index = getattr(self, index_attr)
        total = 1000
        batch_size = 250
        all_vectors = [self._generate_hybrid_vector(i, dim=dim) for i in range(total)]

        for i in range(0, total, batch_size):
            batch = all_vectors[i:i + batch_size]
            res = index.upsert(batch)
            assert "success" in res.lower()
            time.sleep(0.3)

        # Validate a few random vectors
        for i in [0, total // 2, total - 1]:
            got = self._safe_get_vector(index, f"vec_{i}")
            assert got["id"] == f"vec_{i}"
            assert "sparse_vector" in got
            assert "dense_vector" in got
