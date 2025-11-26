import pytest
import os
import sys
import time
import math
import logging
from dotenv import load_dotenv
from vecx.vectorx import VectorX
from vecx.exceptions import APIException
from config.test_config import TestConfig

# Load environment variables
load_dotenv()
VECTORX_API_TOKEN = os.getenv("VECTORX_API_TOKEN")

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    logger.addHandler(handler)

class TestHybridSearch:
    @classmethod
    def _get_dim_from_index_name(cls, index_name):
        if index_name in [TestConfig.TEST_UPSERT_INDEX1, TestConfig.TEST_UPSERT_INDEX3]:
            return 5
        elif index_name in [TestConfig.TEST_UPSERT_INDEX2, TestConfig.TEST_UPSERT_INDEX4]:
            return 768
        else:
            return 5

    @classmethod
    def setup_class(cls):
        cls.vx = VectorX(token=VECTORX_API_TOKEN)
        cls.encryption_key = cls.vx.generate_key()
        cls.cleanup_indexes = []

        index_lst = cls.vx.list_indexes()
        indexes = index_lst.get('indixes', [])

        for index in indexes:
            index_name = index.get('name')
            if not index_name:
                continue
            cls.cleanup_indexes.append(index_name)
            try:
                if 'vocab_size' in index:
                    cls.vx.delete_hybrid_index(index_name)
                else:
                    cls.vx.delete_index(index_name)
                logger.info(f"Deleted index {index_name}")
            except Exception as e:
                logger.error(f"Failed to delete index {index_name}: {e}")

        cls.test_index_name_no_enc = f"{TestConfig.TEST_INDEX_PREFIX}_{TestConfig.get_timestamp()}_{TestConfig.get_unique_id()}_search_no_enc"
        cls.test_index_name_enc = f"{TestConfig.TEST_INDEX_PREFIX}_{TestConfig.get_timestamp()}_{TestConfig.get_unique_id()}_search_enc"
        cls.dimension_no_enc = cls._get_dim_from_index_name(cls.test_index_name_no_enc)
        cls.dimension_enc = cls._get_dim_from_index_name(cls.test_index_name_enc)

        logger.info(f"Creating hybrid index: {cls.test_index_name_no_enc}")
        cls.vx.create_hybrid_index(
            name=cls.test_index_name_no_enc,
            dimension=cls.dimension_no_enc,
            space_type="cosine",
            vocab_size=30522
        )

        logger.info(f"Creating hybrid index: {cls.test_index_name_enc}")
        cls.vx.create_hybrid_index(
            name=cls.test_index_name_enc,
            dimension=cls.dimension_enc,
            space_type="cosine",
            vocab_size=30522,
            key=cls.encryption_key
        )

        for attempt in range(10):
            try:
                cls.hybrid_no_enc = cls.vx.get_hybrid_index(cls.test_index_name_no_enc)
                cls.hybrid_enc = cls.vx.get_hybrid_index(cls.test_index_name_enc, key=cls.encryption_key)
                if cls.hybrid_no_enc and cls.hybrid_enc:
                    break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {e}")
            time.sleep(1)
        else:
            raise RuntimeError("Failed to fetch hybrid indexes.")

        cls.docs = [
            {
                "id": f"doc_{i}",
                "dense_vector": [0.6, 0.7, 0.8, 0.9, 0.2] + [0.2] * (cls.dimension_no_enc - 5),
                "sparse_vector": {"indices": [15, 30, 60], "values": [0.9, 0.7, 0.5]},
                "meta": {"title": f"Doc {i}"}
            }
            for i in range(10)
        ]

        logger.info("Upserting documents into indexes")
        cls.hybrid_no_enc.upsert(cls.docs)
        cls.hybrid_enc.upsert(cls.docs)

        query_dense = [0.6, 0.7, 0.8, 0.9, 0.2] + [0.2] * (cls.dimension_no_enc - 5)
        query_sparse = {"indices": [15, 30, 60], "values": [0.9, 0.7, 0.5]}

        for attempt in range(15):
            try:
                r1 = cls.hybrid_no_enc.search(dense_vector=query_dense, sparse_vector=query_sparse, sparse_top_k=1, dense_top_k=1)
                r2 = cls.hybrid_enc.search(dense_vector=query_dense, sparse_vector=query_sparse, sparse_top_k=1, dense_top_k=1)
                if r1 and r2:
                    break
            except Exception as e:
                logger.warning(f"Search attempt {attempt+1} failed: {e}")
            time.sleep(1)
        else:
            raise RuntimeError("Documents not searchable.")

    @classmethod
    def teardown_class(cls):
        index_lst = cls.vx.list_indexes()
        indexes = index_lst.get('indixes', [])
        for index in indexes:
            index_name = index.get('name')
            if not index_name:
                continue
            try:
                if 'vocab_size' in index:
                    cls.vx.delete_hybrid_index(index_name)
                else:
                    cls.vx.delete_index(index_name)
                logger.info(f"Deleted index {index_name}")
            except Exception as ex:
                logger.error(f"Failed to delete index {index_name}: {ex}")
        logger.info("All test indexes cleaned up.")

    def setup_method(self):
        self.index_no_enc = self.hybrid_no_enc
        self.index_enc = self.hybrid_enc
        self.query_dense = [0.6, 0.7, 0.8, 0.9, 0.2]
        self.query_sparse = {"indices": [15, 30, 60], "values": [0.9, 0.7, 0.5]}

    def make_valid_dense(self, dim):
        return [float(i + 1) / 10.0 for i in range(dim)]

    @pytest.mark.parametrize("index_attr, encrypted", [
        ("index_no_enc", False),
        ("index_enc", True),
    ])
    def test_hybrid_search_basic(self, index_attr, encrypted):
        index = getattr(self, index_attr)
        results = index.search(
            dense_vector=self.query_dense,
            sparse_vector=self.query_sparse,
            sparse_top_k=5,
            dense_top_k=5,
            include_vectors=False,
            rrf_k=10
        )
        assert isinstance(results, list)
        assert len(results) > 0
        for res in results:
            assert "id" in res
            assert "meta" in res
            assert "rrf_score" in res
            assert "dense_rank" in res and "sparse_rank" in res
            assert "dense_vector" not in res

    def test_invalid_query(self):
        with pytest.raises(TypeError):
            self.index_no_enc.search(
                sparse_vector=self.query_sparse,
                sparse_top_k=5,
                dense_top_k=5
            )
        params = dict(dense_vector=self.query_dense, sparse_vector=self.query_sparse, sparse_top_k=-1, dense_top_k=5)
        try:
            res = self.index_no_enc.search(**params)
            assert isinstance(res, list)
        except APIException:
            pass

    @pytest.mark.parametrize("index_attr, encrypted", [
        ("index_no_enc", False),
        ("index_enc", True),
    ])
    def test_retrieve_encrypted_vector_after_search(self, index_attr, encrypted):
        idx = getattr(self, index_attr)
        vec = idx.get_vector("doc_1")
        assert vec["id"] == "doc_1"
        assert "sparse_vector" in vec and "dense_vector" in vec

    @pytest.mark.parametrize("include_vectors", [True, False])
    def test_include_vectors_flag(self, include_vectors):
        index = self.index_enc
        assert index is not None

        try:
            dummy_dense = self.make_valid_dense(len(self.query_dense))
            assert all(math.isfinite(x) for x in dummy_dense)

            results = index.search(
                dense_vector=dummy_dense,
                sparse_vector=self.query_sparse,
                sparse_top_k=3,
                dense_top_k=3,
                include_vectors=include_vectors
            )
        except Exception as e:
            logger.error(f"Error during search with include_vectors={include_vectors}: {e}")
            pytest.fail(str(e))

        logger.info(f"Search results with include_vectors={include_vectors}: {results}")
        assert results is not None
        assert isinstance(results, list)

        for r in results:
            if include_vectors:
                # Check for key returned when include_vectors=True
                assert "vector" in r, f"Expected 'vector' key in result but got: {r}"
            else:
                # Ensure vector data not present
                assert "vector" not in r, f"'vector' should not be present when include_vectors=False: {r}"

    @pytest.mark.parametrize("index_attr, encrypted", [
        ("index_no_enc", False),
        ("index_enc", True),
    ])
    def test_sparse_only_search(self, index_attr, encrypted):
        index = getattr(self, index_attr)
        index_name = self.test_index_name_enc if encrypted else self.test_index_name_no_enc

        assert index is not None, f"{index_attr} is None"

        # Confirm index exists
        try:
            index_list = self.vx.list_indexes()
            index_names = [i.get('name') for i in index_list.get('indixes', [])] if index_list and 'indixes' in index_list else []
            assert index_name in index_names, f"{index_name} not found"
            logger.info(f"Index {index_name} is available.")
        except Exception as e:
            logger.error(f"Error listing indexes: {e}")
            pytest.fail(str(e))

        try:
            dummy_dense = self.make_valid_dense(len(self.query_dense))
            assert all(math.isfinite(x) for x in dummy_dense), "Non-finite values in dense_vector"

            results = index.search(
                dense_vector=dummy_dense,
                sparse_vector=self.query_sparse,
                sparse_top_k=5,
                dense_top_k=0,
                include_vectors=False
            )
        except Exception as e:
            logger.error(f"Error during search for {index_attr}: {e}")
            pytest.fail(f"Error during search for {index_attr}: {e}")

        logger.info(f"Sparse-only search results for {index_attr}: {results}")
        assert results is not None
        assert isinstance(results, list)

        for r in results:
            assert "sparse_rank" in r and r["sparse_rank"] is not None

    @pytest.mark.parametrize("index_attr, encrypted", [
        ("index_no_enc", False),
        ("index_enc", True),
    ])
    def test_dense_only_search(self, index_attr, encrypted):
        index = getattr(self, index_attr)
        assert index is not None, f"{index_attr} is None"

        try:
            dummy_dense = self.make_valid_dense(len(self.query_dense))
            assert all(math.isfinite(x) for x in dummy_dense)

            results = index.search(
                dense_vector=dummy_dense,
                sparse_vector={},
                sparse_top_k=0,
                dense_top_k=5,
                include_vectors=False
            )

            assert results is not None
            assert isinstance(results, list)

            logger.info(f"Dense-only search results for {index_attr}: {results}")

            for r in results:
                assert isinstance(r, dict), f"Each result must be a dict, got {type(r)}"
                assert "dense_rank" in r and r["dense_rank"] is not None

        except Exception as e:
            logger.error(f"Error during dense-only search for {index_attr}: {e}")
            pytest.fail(str(e))
