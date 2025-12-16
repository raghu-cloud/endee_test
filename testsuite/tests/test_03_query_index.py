import pytest
import os
import sys
import numpy as np
from endee import Endee
from endee.exceptions import APIException
from dotenv import load_dotenv
import logging
from config.test_config import TestConfig
import builtins

load_dotenv()
ENDEE_API_TOKEN = getattr(builtins, "ENDEE_API_KEY", None)
if ENDEE_API_TOKEN == None:
    ENDEE_API_TOKEN = os.getenv("ENDEE_API_TOKEN")

timestamp = getattr(builtins, "TEST_RUN_TIMESTAMP", None)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # or DEBUG

if not logger.hasHandlers():  # Prevent duplicate handlers
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TestQueryIndex:
    @classmethod
    def setup_class(cls):
        cls.nd = Endee(token=ENDEE_API_TOKEN)
        # cls.pipeline_mode = os.getenv("PIPELINE_MODE", "false").lower() == "true"
        # with open(f"config/pipeline_mode_bool_{timestamp}.txt", "r") as f:
        #     cls.pipeline_mode = f.read().strip().lower() == "true"
        # if not cls.pipeline_mode:
        #     cls.encryption_key = cls.vx.generate_key()
        #     # Delete any leftover test indexes first
        #     index_lst = cls.vx.list_indexes()
        #     if len(index_lst['indixes'])>0:
        #         for index in index_lst['indixes']:
        #             cls.vx.delete_index(index['name'])

        #     # Updated index configs with new space_types
        #     cls.index_configs = [
        #         {"name": TestConfig.TEST_UPSERT_INDEX1, "dimension": 5, "encryption": False, "space_type": "cosine"},
        #         {"name": TestConfig.TEST_UPSERT_INDEX2, "dimension": 768, "encryption": False, "space_type": "l2"},
        #         {"name": TestConfig.TEST_UPSERT_INDEX3, "dimension": 5, "encryption": True, "space_type": "ip"},
        #         {"name": TestConfig.TEST_UPSERT_INDEX4, "dimension": 768, "encryption": True, "space_type": "cosine"},
        #     ]

        #     # Create each index according to config
        #     for config in cls.index_configs:
        #         create_kwargs = {
        #             "name": config["name"],
        #             "dimension": config["dimension"],
        #             "space_type": config["space_type"],
        #         }
        #         if config["encryption"]:
        #             create_kwargs["key"] = cls.encryption_key

        #         logger.info(f"Creating index: {config['name']} with dimension {config['dimension']} and encryption {config['encryption']} and space_type {config['space_type']}")
        #         result = cls.vx.create_index(**create_kwargs)

        #     num_vectors = 2000
        #     for config in cls.index_configs:
        #         if not config['encryption']:
        #             idx = cls.vx.get_index(config['name'])
        #         else:
        #             idx = cls.vx.get_index(config['name'], key=cls.encryption_key)
        #         vectors = [TestConfig.generate_vector(str(i), config["dimension"], space_type=config["space_type"]) for i in range(num_vectors)]

        #         # Upsert in batches of 1000
        #         for i in range(0, num_vectors, 1000):
        #             batch = vectors[i:i + 1000]
        #             idx.upsert(batch)
        #             logger.info(f"Upserted batch {i // 1000 + 1}")
        #         logger.info(f"Upserted 2000 vectors for index {config['name']}")
        # else:
        #     with open(f"config/tmp_encryption_key_{timestamp}.txt", "r") as f:
        #         cls.encryption_key = f.read().strip()

    @classmethod
    def teardown_class(cls):
        # index_lst = cls.nd.list_indexes()
        # if len(index_lst['indixes'])>0:
        #     for index in index_lst['indixes']:
        #         cls.nd.delete_index(index['name'])
        logger.info("Testing Query index done")

        

    def setup_method(self):
        """Setup before each test"""
        self.index_no_enc_5 = self.nd.get_index(name=TestConfig.TEST_UPSERT_INDEX1)
        self.index_no_enc_768 = self.nd.get_index(name=TestConfig.TEST_UPSERT_INDEX2)
        # self.index_enc_5 = self.nd.get_index(name=TestConfig.TEST_UPSERT_INDEX3, key= self.encryption_key)
        # self.index_enc_768 = self.nd.get_index(name=TestConfig.TEST_UPSERT_INDEX4, key=self.encryption_key)

    def test_missing_query_vector(self):
        """Test that missing query vector raise TypeError with correct messages."""
        with pytest.raises(TypeError) as exc_info:
            results = self.index_no_enc_5.query(
                top_k=5
            )
        assert "missing 1 required positional argument: 'vector'" in str(exc_info.value)

        # with pytest.raises(TypeError) as exc_info:
        #     results = self.index_enc_5.query(
        #         top_k=5
        #     )
        # assert "missing 1 required positional argument: 'vector'" in str(exc_info.value)

    def test_vector_dimension_mismatch(self):
        """Test that vector dimension mismatch raises ValueError with correct messages."""
        with pytest.raises(ValueError) as exc_info:
            results = self.index_no_enc_5.query(
                vector=[0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7],
                top_k=1
            )
        assert "Vector dimension mismatch:" in str(exc_info.value)

        # with pytest.raises(ValueError) as exc_info:
        #     results = self.index_enc_5.query(
        #         vector=[0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7],
        #         top_k=1
        #     )
        # assert "Vector dimension mismatch:" in str(exc_info.value)

    def test_invalid_top_k(self):
        """Test that invalid top_k value raises APIException with correct messages."""
        invalid_top_k_values = [    
            -1,
            0,
            4097,   
            10000 
        ]
        for invalid_top_k in invalid_top_k_values:
            with pytest.raises((APIException, ValueError)) as exc_info:
                results = self.index_no_enc_5.query(
                    vector=[0.1, 0.2, 0.3, 0.5, 0.7],
                    top_k= invalid_top_k
                )
            err = str(exc_info.value)
            assert "top_k cannot be greater than 512 and top_k cannot be less than 1" in err


    @pytest.mark.parametrize("index_attr", ["index_no_enc_5", "index_no_enc_768"])
    @pytest.mark.parametrize("top_k", [5, 10, 15])
    def test_valid_top_k_vectors_count(self, index_attr, top_k):
        """Test query returns correct number of results for different top_k values"""
        index = getattr(self, index_attr)
        
        # Generate appropriate random query vectors based on index dimension
        if "5" in index_attr:  # For 5-dimensional indexes
            query_vectors = [
                np.random.rand(5).tolist(),  # Random 5D vector
                [0.5] * 5  # Uniform 5D vector
            ]
        else:  # For 768-dimensional indexes
            query_vectors = [
                np.random.rand(768).tolist(),  # Random 768D vector
                [0.1] * 768  # Uniform 768D vector
            ]

        for query_vector in query_vectors:
            results = index.query(vector=query_vector, top_k=top_k)

            assert len(results) <= top_k
            index_description = index.describe()
            assert len(results) == min(top_k, index_description['count'])

            for result in results:
                assert 'id' in result
                assert 'similarity' in result
                assert 'distance' in result
                assert 'norm' in result 


    @pytest.mark.parametrize("index_attr", ["index_no_enc_5", "index_no_enc_768"])
    @pytest.mark.parametrize("filter_sub_category", ["public", "private"])
    def test_filter_match_and_vector_parameters(self, index_attr, filter_sub_category):
        """Test query returns correct results for with proper filter applied"""
        index = getattr(self, index_attr)

        if "5" in index_attr:  # For 5-dimensional indexes
            query_vector = np.random.rand(5).tolist()  # Random 5D vector
        else:  # For 768-dimensional indexes
            query_vector = np.random.rand(768).tolist()  # Random 768D vector
        
        results = index.query(
            vector=query_vector,      # Query vector
            top_k=20,           # Number of results to return
            filter= [{"visibility":{"$eq":filter_sub_category}}],   # Filter for matching
            ef=128,            # Runtime parameter for search quality
            include_vectors=True # Include vector data in results
        )

        assert len(results) <= 20

        # print("RESULTS",results)
        for result in results:
            vector_id_in_result = result['id']
            vector_meta_in_result = result['meta']
            vector_filter_category_in_result = result['filter']
            vector_filter_sub_category_in_result = vector_filter_category_in_result['visibility']
            vector_in_result = result['vector']

            vector_details = index.get_vector(vector_id_in_result)
            vector_id = vector_details['id']
            vector_meta = vector_details['meta']
            # vector_filter_category = 
            vector = vector_details['vector']

            assert vector_id_in_result == vector_id
            assert vector_meta_in_result == vector_meta
            assert vector_filter_sub_category_in_result == filter_sub_category
            assert np.array_equal(vector_in_result, vector)
                