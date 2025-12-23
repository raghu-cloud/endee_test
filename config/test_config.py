import time
import uuid
import numpy as np
import random

class TestConfig:
    # Prefix used for all test index names
    TEST_INDEX_PREFIX = "test_idx_"
    TEST_UPSERT_INDEX1 = "test_idx_5_no_enc"
    TEST_UPSERT_INDEX2 = "test_idx_768_no_enc"
    TEST_UPSERT_INDEX3 = "test_idx_5_enc"
    TEST_UPSERT_INDEX4 = "test_idx_768_enc"


    @staticmethod
    def get_timestamp():
        """Returns a timestamp string for index names."""
        return time.strftime("%Y%m%d%H%M%S")
    
    @staticmethod
    def get_unique_id():
        """Returns a unique id string for index names."""
        return uuid.uuid4().hex[:6]
    
    @staticmethod
    def generate_vector(id_suffix: str, dim: int, title: str = "Test Vector", visibility=None, space_type=None):
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

