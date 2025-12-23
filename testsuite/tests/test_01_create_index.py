import pytest
import os
import sys
import time
from endee import Endee
from dotenv import load_dotenv
import logging
from config.test_config import TestConfig
import builtins

load_dotenv()
ENDEE_API_TOKEN = getattr(builtins, "ENDEE_API_KEY", None)
if ENDEE_API_TOKEN == None:
    ENDEE_API_TOKEN = os.getenv("ENDEE_API_TOKEN")


# logging.basicConfig(
#     level=logging.INFO,  # or DEBUG if you want more detail
#     format='%(levelname)s:%(name)s:%(message)s'
# )

# logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # or DEBUG

if not logger.hasHandlers():  # Prevent duplicate handlers
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TestCreateIndex:
    @classmethod
    def setup_class(cls):
        cls.nd = Endee(token=ENDEE_API_TOKEN)
        cls.encryption_key = cls.nd.generate_key()
        index_lst = cls.nd.list_indexes()
        # print("Length", len(index_lst['indexes']))
 
        if len(index_lst['indexes'])>0:
            for index in index_lst['indexes']:
                cls.nd.delete_index(index['name'])
        cls.cleanup_indexes = []


    @classmethod
    def teardown_class(cls):
        # Runs once after all tests in this class
        index_lst = cls.nd.list_indexes()
        if len(index_lst['indexes'])>0:
            for index in index_lst['indexes']:
                cls.nd.delete_index(index['name'])
        logger.info("Create index tests done")

    
    def setup_method(self):
        """Setup before each test"""
        self.test_index_name = f"{TestConfig.TEST_INDEX_PREFIX}create_{TestConfig.get_timestamp()}_{TestConfig.get_unique_id()}"
        time.sleep(2)
        # self.cleanup_indexes = []

    def teardown_method(self):
        """Cleanup after each test"""
        index_lst = self.nd.list_indexes()
        print("Length", len(index_lst['indexes']))
        # if len(index_lst['indexes'])==3:
        for index_name in self.cleanup_indexes:
            print("indexes to be deleted", self.cleanup_indexes)
            try:
                self.nd.delete_index(index_name)
                logger.info(f"Deleted index: {index_name}")
                print("DELETED")
            except Exception as e:
                logger.warning(f"Failed to delete index '{index_name}': {e}")
            self.cleanup_indexes.clear()
            

    def test_create_index_missing_parameters(self):
        """Test that missing required parameters raise TypeError with correct messages."""
        # Test missing 'dimension'
        with pytest.raises(TypeError) as exc_info:
            self.nd.create_index(
                name="test_index",
                space_type="cosine"
            )
        assert "missing 1 required positional argument: 'dimension'" in str(exc_info.value)

        # Test missing 'space_type'
        with pytest.raises(TypeError) as exc_info:
            self.nd.create_index(
                name="test_index",
                dimension=768
            )
        assert "missing 1 required positional argument: 'space_type'" in str(exc_info.value)

        # Test missing 'name'
        with pytest.raises(TypeError) as exc_info:
            self.nd.create_index(
                dimension=768,
                space_type="cosine"
            )
        assert "missing 1 required positional argument: 'name'" in str(exc_info.value)

    def test_create_index_invalid_name(self):
        """Test index name validation with different invalid cases"""
        # Cases that should raise ValueError (invalid strings)
        invalid_strings = [
            "invalid name",  # contains space
            "invalid@name",  # contains special character
            "x" * 49,  # too long (48 character limit)
            "",  # empty string
        ]
        
        for name in invalid_strings:
            with pytest.raises(ValueError) as exc_info:
                self.nd.create_index(
                    name=name,
                    dimension=768,
                    space_type="cosine"
                )
            assert "Invalid index name" in str(exc_info.value)
            assert "alphanumeric" in str(exc_info.value)
            assert "underscores" in str(exc_info.value)
            assert "less than 48 characters" in str(exc_info.value)

        # Cases that should raise TypeError (wrong types)
        wrong_type_names = [
            None,  # None value
            123,  # integer
            ["invalid"],  # list
            {"name": "invalid"},  # dict
        ]
        
        for name in wrong_type_names:
            with pytest.raises(TypeError) as exc_info:
                self.nd.create_index(
                    name=name,
                    dimension=768,
                    space_type="cosine"
                )
            assert "expected string or bytes-like object" in str(exc_info.value).lower()

    @pytest.mark.parametrize("dimension", [5, 768])
    @pytest.mark.parametrize("space_type", ['cosine', 'l2', 'ip'])
    @pytest.mark.parametrize("M", [16, 32, 64])
    @pytest.mark.parametrize("ef_con", [128, 256])
    @pytest.mark.parametrize("precision", ["medium", "high","ultra-high","fp16"])

    @pytest.mark.parametrize("encryption", [True, False])
    def test_create_index_combinations(self, dimension, space_type, M, ef_con, precision,encryption):
        """Test all parameter combinations for index creation"""
        index_name = self.test_index_name
        self.cleanup_indexes.append(index_name)

        try:
            if encryption==False:
                result = self.nd.create_index(
                    name=index_name,
                    dimension=dimension,
                    space_type=space_type,
                    M=M,
                    ef_con=ef_con,
                    precision=precision
                )
            else:
                result = self.nd.create_index(
                    name=index_name,
                    dimension=dimension,
                    key=self.encryption_key,
                    space_type=space_type,
                    M=M,
                    ef_con=ef_con,
                    precision=precision
                )


            assert result == "Index created successfully", f"Unexpected result: {result}"
            if encryption == False:
                index_info = self.nd.get_index(index_name)
            else:
                index_info = self.nd.get_index(name = index_name, key = self.encryption_key)

            # index_info = self.nd.get_index(index_name)
      
            info = index_info.describe()
            assert info["name"] == index_name
            assert info["dimension"] == dimension
            assert info["space_type"] == space_type
            assert info["M"] == M
            assert info["precision"] == precision

        except Exception as e:
            logger.error(
                f"❌ Test failed with parameters passed: "
                f"dimension={dimension}, space_type={space_type}, M={M}, ef_con={ef_con}, precision={precision}\n"
                f"Error: {e}"
            )
            raise  # Re-raise the exception to let pytest report the failure