import pytest
import os
import sys
import time
import random
import json
import numpy as np
from dotenv import load_dotenv
from endee.endee import Endee
from config.test_config import TestConfig
import logging
import builtins

# Load environment variables
load_dotenv()
ENDEE_API_TOKEN = getattr(builtins, "ENDEE_API_KEY", None)
if ENDEE_API_TOKEN == None:
    ENDEE_API_TOKEN = os.getenv("ENDEE_API_TOKEN")

timestamp = getattr(builtins, "TEST_RUN_TIMESTAMP", None)

print(ENDEE_API_TOKEN)

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TestUpsertVectors:
    @classmethod
    def setup_class(cls):
        """
        Setup Endee client,
        clean old test indexes, then create the 4 test indexes manually.
        """
        cls.nd = Endee(token=ENDEE_API_TOKEN)
        # cls.encryption_key = cls.nd.generate_key()
        
        # with open(f"config/pipeline_mode_bool_{timestamp}.txt", "r") as f:
        #     cls.pipeline_mode = f.read().strip().lower() == "true"
        
        # if cls.pipeline_mode:
        #     with open(f"config/tmp_encryption_key_{timestamp}.txt", "w") as f:
        #         f.write(cls.encryption_key)

        # Delete any leftover test indexes first
        index_lst = cls.nd.list_indexes()
        if len(index_lst['indixes'])>0:
            for index in index_lst['indixes']:
                cls.nd.delete_index(index['name'])

        # Updated index configs with new space_types
        cls.index_configs = [
            {"name": TestConfig.TEST_UPSERT_INDEX1, "dimension": 5, "space_type": "cosine"},
            {"name": TestConfig.TEST_UPSERT_INDEX2, "dimension": 768, "space_type": "l2"},
            # {"name": TestConfig.TEST_UPSERT_INDEX3, "dimension": 5, "space_type": "ip"},
            # {"name": TestConfig.TEST_UPSERT_INDEX4, "dimension": 768, "space_type": "cosine"},
        ]

        # Create each index according to config
        for config in cls.index_configs:
            create_kwargs = {
                "name": config["name"],
                "dimension": config["dimension"],
                "space_type": config["space_type"],
            }
            # if config["encryption"]:
            #     create_kwargs["key"] = cls.encryption_key

            logger.info(f"Creating index: {config['name']} with dimension {config['dimension']} and space_type {config['space_type']}")
            result = cls.nd.create_index(**create_kwargs)
            assert result in ("Index created successfully", "Index already exists")

        # Sleep shortly to ensure indexes are ready
        time.sleep(2)

    def setup_method(self):
        """
        Fetch the manually created indexes for each test method.
        """
        self.index_no_enc_5 = self.nd.get_index(name=TestConfig.TEST_UPSERT_INDEX1)
        self.index_no_enc_768 = self.nd.get_index(name=TestConfig.TEST_UPSERT_INDEX2)
        # self.index_enc_5 = self.nd.get_index(name=TestConfig.TEST_UPSERT_INDEX3, key=self.encryption_key)
        # self.index_enc_768 = self.nd.get_index(name=TestConfig.TEST_UPSERT_INDEX4, key=self.encryption_key)

    # @classmethod
    # def teardown_class(cls):
    #     """
    #     Clean up test indexes after all tests.
    #     """
    #     indexes = cls.nd.list_indexes()
    #     for idx in indexes.get("indices", []):
    #         if idx["name"].startswith(TestConfig.TEST_INDEX_PREFIX):
    #             cls.nd.delete_index(idx["name"])
    #     logger.info("Deleted all test indexes after test run.")

    def _generate_vector(self, id_suffix: str, dim: int, title: str = "Test Vector", visibility=None, space_type=None):
        """
        Helper to generate a random vector dictionary with metadata and filter.
        Normalize vectors if space_type is "ip".
        """
        vec = np.random.rand(dim)
        if space_type == "ip":
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return {
            "id": f"vec_{id_suffix}",
            "vector": vec.tolist(),
            "meta": {"title": title},
            "filter": {"visibility": visibility or random.choice(["public", "private"])}
        }

    def _safe_get_vector(self, index, vec_id):
        """
        Helper to safely retrieve a vector by id, parsing JSON if necessary.
        """
        result = index.get_vector(vec_id)
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                pytest.fail(f"Response for vector '{vec_id}' is not valid JSON: {result}")
        return result

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        ("index_no_enc_5", 5, "cosine"),
        ("index_no_enc_768", 768, "l2"),
    ])
    def test_upsert_10_vectors(self, index_attr, dimension, space_type):
        """
        Test upserting valid vectors and retrieving them.
        """
        idx = getattr(self, index_attr)
        vectors = [self._generate_vector(str(i), dimension, space_type=space_type) for i in range(10)]
        sample = vectors[5]

        result = idx.upsert(vectors)
        assert result in ("Vectors upserted successfully", "Vectors inserted successfully")

        retrieved = self._safe_get_vector(idx, sample["id"])
        assert retrieved["id"] == sample["id"]
        assert retrieved["meta"]["title"] == sample["meta"]["title"]
        assert len(retrieved["vector"]) == dimension

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        ("index_no_enc_5", 5, "cosine"),
        ("index_no_enc_768", 768, "l2"),
    ])
    def test_upsert_dimension_mismatch(self, index_attr, dimension, space_type):
        """
        Test upserting vector with wrong dimension raises exception.
        """
        idx = getattr(self, index_attr)
        bad_vector = self._generate_vector("bad_dim", dim=dimension - 1, space_type=space_type)
        with pytest.raises(Exception) as e_info:
            idx.upsert([bad_vector])
        assert "dimension" in str(e_info.value).lower()

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        ("index_no_enc_5", 5, "cosine"),
        ("index_no_enc_768", 768, "l2"),
    ])
    def test_upsert_invalid_empty_id(self, index_attr, dimension, space_type):
        """
        Test upserting vector with empty ID raises exception.
        """
        idx = getattr(self, index_attr)
        bad_vector = self._generate_vector("empty_id", dimension, space_type=space_type)
        bad_vector["id"] = ""
        with pytest.raises(Exception) as e_info:
            idx.upsert([bad_vector])
        assert "bad request" in str(e_info.value).lower() or "id" in str(e_info.value).lower()

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        ("index_no_enc_5", 5, "cosine"),
        ("index_no_enc_768", 768, "l2"),
    ])
    def test_upsert_empty_vector(self, index_attr, dimension, space_type):
        """
        Test upserting vector with empty vector list raises exception.
        """
        idx = getattr(self, index_attr)
        bad_vector = self._generate_vector("empty_vec", dim=dimension, space_type=space_type)
        bad_vector["vector"] = []

        with pytest.raises(Exception) as e_info:
            idx.upsert([bad_vector])
        assert "vector dimension mismatch" in str(e_info.value).lower()

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        ("index_no_enc_5", 5, "cosine"),
        ("index_no_enc_768", 768, "l2"),
    ])
    def test_upsert_and_retrieve_non_existent_id(self, index_attr, dimension, space_type):
        """
        Test that retrieving a non-existent vector raises exception.
        """
        idx = getattr(self, index_attr)
        valid_vector = self._generate_vector("valid", dimension, space_type=space_type)
        idx.upsert([valid_vector])

        with pytest.raises(Exception) as e_info:
            idx.get_vector("vec_nonexistent")
        assert "not found" in str(e_info.value).lower()

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        ("index_no_enc_5", 5, "cosine"),
        ("index_no_enc_768", 768, "l2"),
    ])
    def test_upsert_update_existing_vector(self, index_attr, dimension, space_type):
        """
        Test updating an existing vector overwrites metadata.
        """
        idx = getattr(self, index_attr)
        vec_id = "vec_update"
        original = self._generate_vector("update", dimension, title="Original Title", space_type=space_type)
        idx.upsert([original])

        updated = self._generate_vector("update", dimension, title="Updated Title", space_type=space_type)
        idx.upsert([updated])

        retrieved = self._safe_get_vector(idx, vec_id)
        assert retrieved["meta"]["title"] == "Updated Title"

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        ("index_no_enc_5", 5, "cosine"),
        ("index_no_enc_768", 768, "l2"),
    ])
    def test_upsert_list_meta_type(self, index_attr, dimension, space_type):
        """
        Test that upserting vector with meta as list is accepted.
        """
        idx = getattr(self, index_attr)
        bad_vector = self._generate_vector("bad_meta", dimension, space_type=space_type)
        bad_vector["meta"] = ["not", "a", "dict"]

        idx.upsert([bad_vector])
        retrieved = self._safe_get_vector(idx, bad_vector["id"])
        assert isinstance(retrieved["meta"], list)

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        ("index_no_enc_5", 5, "cosine"),
        ("index_no_enc_768", 768, "l2"),
    ])
    def test_upsert_long_metadata(self, index_attr, dimension, space_type):
        """
        Test upserting vector with very long metadata string.
        """
        idx = getattr(self, index_attr)
        long_title = "A" * 2000
        vector = self._generate_vector("long_meta", dimension, title=long_title, space_type=space_type)
        idx.upsert([vector])

        retrieved = self._safe_get_vector(idx, vector["id"])
        assert retrieved["meta"]["title"] == long_title

    # @pytest.mark.parametrize("index_attr, dimension, space_type", [
    #     ("index_no_enc_5", 5, "cosine"),
    #     ("index_no_enc_768", 768, "l2"),
    # ])
    # def test_large_batch_upsert(self, index_attr, dimension, space_type):
    #     """
    #     Test upserting large batches in multiple chunks.
    #     """
    #     idx = getattr(self, index_attr)
    #     num_vectors = 2000
    #     BATCH_SIZE = 100  # Reduced batch size
    #     DELAY = 0.5       # 300ms delay between batches
    #     vectors = [self._generate_vector(str(i), dimension, space_type=space_type) for i in range(num_vectors)]

    #     # Upsert in smaller batches with delay
    #     for i in range(0, num_vectors, BATCH_SIZE):
    #         batch = vectors[i:i + BATCH_SIZE]
    #         idx.upsert(batch)
    #         logger.info(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} vectors)")
    #         time.sleep(DELAY)

    #     # Validate a few specific vectors
    #     for i in [0, num_vectors // 2, num_vectors - 1]:
    #         vec_id = f"vec_{i}"
    #         retrieved = self._safe_get_vector(idx, vec_id)
    #         assert retrieved["id"] == vec_id

    @pytest.mark.parametrize("index_attr, dimension, space_type", [
        ("index_no_enc_5", 5, "cosine"),
        ("index_no_enc_768", 768, "l2"),
    ])
    def test_large_batch_upsert(self, index_attr, dimension, space_type):
        """
        Test upserting large batches in multiple chunks,
        with retry logic to avoid server-busy breaks.
        """
        idx = getattr(self, index_attr)
        num_vectors = 2000
        BATCH_SIZE = 1000
        BASE_DELAY = 1.0  # backoff starting point
        MAX_RETRIES = 10   # retry attempts per batch

        vectors = [
            self._generate_vector(str(i), dimension, space_type=space_type)
            for i in range(num_vectors)
        ]

        # Upsert in smaller batches
        for i in range(0, num_vectors, BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]

            # ---- RETRY MECHANISM (inline, no helper function) ----
            for attempt in range(MAX_RETRIES):
                try:
                    idx.upsert(batch)
                    break  # success → stop retry loop
                except Exception as e:
                    msg = str(e).lower()
                    if ("busy" in msg or 
                        "rate" in msg or 
                        "timeout" in msg or 
                        "unavailable" in msg or 
                        "temporarily" in msg):

                        wait_time = BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            f"[batch {i//BATCH_SIZE + 1}] "
                            f"Server busy, retry {attempt+1}/{MAX_RETRIES} in {wait_time:.2f}s: {e}"
                        )
                        time.sleep(wait_time)
                    else:
                        # Non-transient → raise immediately
                        raise
            else:
                # If all retries failed → log but continue (do NOT fail test)
                logger.error(
                    f"Batch {i//BATCH_SIZE + 1} failed after {MAX_RETRIES} retries. Continuing..."
                )

            # optional small delay between batches
            time.sleep(0.5)
            logger.info(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} vectors)")

        # Validate a few specific vectors
        for i in [0, num_vectors // 2, num_vectors - 1]:
            vec_id = f"vec_{i}"
            retrieved = self._safe_get_vector(idx, vec_id)
            assert retrieved["id"] == vec_id
