import unittest
from llama_index.core import Document
from vecx.vectorx import VectorX
import time
import os

class VectorXTestSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # API credentials
        cls.vecx_api_token = "cmaa1rt3:rfqgqjrOUlrwvWcJ69sK602rNyobFKeL:india-west-1"
        cls.vx = VectorX(
            token=cls.vecx_api_token,
        )
        print(cls.vx)
        cls.encryption_key = cls.vx.generate_key()
        
        # Test index configuration
        timestamp = int(time.time())
        cls.test_index_name = f"test_index_{timestamp}"  # Added timestamp to make unique
        cls.dimension = 384 
        cls.space_type = "cosine"
        
        # Track all test indexes for cleanup
        cls.test_indexes = {cls.test_index_name}
        
        # Create test documents with diverse metadata for comprehensive testing
        cls.test_documents = [
            # Programming category with different languages and difficulties
            Document(
                text="Python is a high-level, interpreted programming language known for its readability and simplicity.",
                metadata={"category": "programming", "language": "python", "difficulty": "beginner"}
            ),
            Document(
                text="JavaScript is a versatile language for web development with advanced features like async/await.",
                metadata={"category": "programming", "language": "javascript", "difficulty": "intermediate"}
            ),
            Document(
                text="Rust provides memory safety without garbage collection using ownership system.",
                metadata={"category": "programming", "language": "rust", "difficulty": "advanced"}
            ),
            
            # AI/ML category with different fields and difficulties
            Document(
                text="Machine learning algorithms learn patterns from data to make predictions.",
                metadata={"category": "ai", "field": "machine_learning", "difficulty": "intermediate"}
            ),
            Document(
                text="Deep learning uses neural networks with multiple layers for complex pattern recognition.",
                metadata={"category": "ai", "field": "deep_learning", "difficulty": "advanced"}
            ),
            
            # Database category with different types and features
            Document(
                text="Vector databases optimize similarity search for high-dimensional data.",
                metadata={"category": "database", "type": "vector", "feature": "similarity_search"}
            ),
            Document(
                text="Time-series databases are optimized for sequential temporal data storage.",
                metadata={"category": "database", "type": "time_series", "feature": "temporal_storage"}
            ),
            
            # Documents with multiple metadata fields for complex filtering
            Document(
                text="Building a real-time ML pipeline with Python and Vector DB.",
                metadata={
                    "category": ["programming", "ai", "database"],
                    "languages": ["python"],
                    "technologies": ["ml", "vector_db"],
                    "difficulty": "advanced"
                }
            ),
            Document(
                text="Implementing secure encryption in distributed databases.",
                metadata={
                    "category": ["programming", "database", "security"],
                    "field": "cryptography",
                    "difficulty": "advanced",
                    "feature": "encryption"
                }
            )
        ]

    @classmethod
    def tearDownClass(cls):
        """Clean up by deleting test indexes"""
        for index_name in cls.test_indexes:
            try:
                cls.vx.delete_index(name=index_name)
                print(f"Successfully deleted test index: {index_name}")
            except Exception as e:
                if "not found" not in str(e).lower():  # Ignore if index doesn't exist
                    print(f"Error deleting test index {index_name}: {e}")

    def setUp(self):
        """Additional setup before each test if needed"""
        pass

    def tearDown(self):
        """Additional cleanup after each test if needed"""
        # Clean up any indexes created during the test
        try:
            indexes = self.vx.list_indexes()
            if isinstance(indexes, list):  # Handle list response
                for index in indexes:
                    if isinstance(index, dict) and 'name' in index:
                        index_name = index['name']
                        if index_name.startswith("test_index_"):
                            try:
                                self.vx.delete_index(name=index_name)
                                print(f"Cleaned up test index: {index_name}")
                            except Exception as e:
                                print(f"Error cleaning up test index {index_name}: {e}")
            elif isinstance(indexes, dict):  # Handle dict response
                for index_name in indexes.values():
                    if isinstance(index_name, str) and index_name.startswith("test_index_"):
                        try:
                            self.vx.delete_index(name=index_name)
                            print(f"Cleaned up test index: {index_name}")
                        except Exception as e:
                            print(f"Error cleaning up test index {index_name}: {e}")
        except Exception as e:
            print(f"Error listing indexes for cleanup: {e}")

