import pytest
import os
import sys
import time
from vecx.vectorx import VectorX
from dotenv import load_dotenv
import logging
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


class TestCreateHybridIndex:
    @classmethod
    def setup_class(cls):
        cls.vx = VectorX(token=VECTORX_API_TOKEN)
        cls.encryption_key = cls.vx.generate_key()
        index_lst = cls.vx.list_indexes()
        indexes = index_lst.get('indixes', [])
        for index in indexes:
            index_name = index.get('name')
            if not index_name:
                continue
            if 'vocab_size' in index:
                cls.vx.delete_hybrid_index(index_name)
            else:
                cls.vx.delete_index(index_name)
        cls.cleanup_indexes = []

    @classmethod
    def teardown_class(cls):
        index_lst = cls.vx.list_indexes()
        indexes = index_lst.get('indixes', [])
        for index in indexes:
            index_name = index.get('name')
            if not index_name:
                continue
            if 'vocab_size' in index:
                cls.vx.delete_hybrid_index(index_name)
            else:
                cls.vx.delete_index(index_name)

    def setup_method(self):
        self.cleanup_indexes = []
        # Updated index naming and dimensions
        self.test_index_name = f"{TestConfig.TEST_INDEX_PREFIX}_{TestConfig.get_timestamp()}_{TestConfig.get_unique_id()}_new"
        time.sleep(1)

    def teardown_method(self):
        index_lst = self.vx.list_indexes()
        indexes = index_lst.get('indixes', [])
        for index_name in self.cleanup_indexes:
            idx_info = next((i for i in indexes if i['name'] == index_name), None)
            if not idx_info:
                continue
            if 'vocab_size' in idx_info:
                self.vx.delete_hybrid_index(index_name)
            else:
                self.vx.delete_index(index_name)
        self.cleanup_indexes.clear()

    def test_invalid_index_name(self):
        invalid_names = [
            "invalid name",    # space
            "invalid@name",    # special char
            "x" * 49,          # too long (>48 chars)
            "",                # empty string
            None,              # NoneType
            123,               # int
            ["list"],          # list
            {"name": "dict"},  # dict
        ]

        for name in invalid_names:
            with pytest.raises((ValueError, TypeError)) as exc:
                self.vx.create_hybrid_index(
                    name=name,
                    dimension=512,  # Updated dimension
                    space_type="cosine",
                    vocab_size=30522
                )
            err_msg = str(exc.value).lower()
            assert ("invalid index name" in err_msg) or ("expected string" in err_msg)

    @pytest.mark.parametrize("dimension", [5, 768])
    @pytest.mark.parametrize("space_type", ['cosine', 'l2', 'ip'])
    @pytest.mark.parametrize("vocab_size", [30522])
    @pytest.mark.parametrize("encryption", [True, False])
    def test_create_hybrid_index_combinations(self, dimension, space_type, vocab_size, encryption):
        index_name = self.test_index_name
        self.cleanup_indexes.append(index_name)

        # Check if index already exists, delete if so
        existing_indexes = self.vx.list_indexes()
        for idx in existing_indexes.get('indixes', []):
            if idx['name'] == index_name:
                logger.info(f"Index {index_name} exists before creation, deleting it first.")
                if 'vocab_size' in idx:
                    self.vx.delete_hybrid_index(index_name)
                else:
                    self.vx.delete_index(index_name)
                break

        try:
            if encryption:
                result = self.vx.create_hybrid_index(
                    name=index_name,
                    dimension=dimension,
                    space_type=space_type,
                    vocab_size=vocab_size,
                    key=self.encryption_key
                )
            else:
                result = self.vx.create_hybrid_index(
                    name=index_name,
                    dimension=dimension,
                    space_type=space_type,
                    vocab_size=vocab_size
                )

            assert isinstance(result, str) and ("success" in result.lower() or "created" in result.lower())

            # Get the index instance (with key if encrypted)
            if encryption:
                idx_instance = self.vx.get_hybrid_index(name=index_name, key=self.encryption_key)
            else:
                idx_instance = self.vx.get_hybrid_index(name=index_name)

            # Basic property asserts
            assert idx_instance.name == index_name
            assert idx_instance.dimension == dimension
            assert idx_instance.space_type == space_type
            assert idx_instance.vocab_size == vocab_size
            assert (idx_instance.key is not None) == encryption

        except Exception as e:
            logger.error(f"Failed hybrid index create test: {e}")
            raise
