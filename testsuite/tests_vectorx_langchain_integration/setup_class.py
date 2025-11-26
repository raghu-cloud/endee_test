import unittest
import time
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import builtins


load_dotenv()
VECTORX_API_TOKEN = getattr(builtins, "VECTORX_API_KEY", None)
if VECTORX_API_TOKEN == None:
    VECTORX_API_TOKEN = os.getenv("VECTORX_API_TOKEN")


class VectorXLangChainTestSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # API credentials
        cls.vecx_api_token = VECTORX_API_TOKEN
        from vecx.vectorx import VectorX
        cls.vx = VectorX(token=cls.vecx_api_token)
        cls.encryption_key = cls.vx.generate_key()

        # Test index configuration
        timestamp = int(time.time())
        cls.test_index_name = f"test_langchain_index_{timestamp}"
        cls.dimension = 384
        cls.space_type = "cosine"

        # Track all test indexes for cleanup
        cls.test_indexes = {cls.test_index_name}

        # Create test documents with diverse metadata for comprehensive testing
        cls.test_texts = [
            # Programming
            "Python is a high-level, interpreted programming language known for its readability and simplicity.",
            "JavaScript is a scripting language that enables interactive web pages and is an essential part of web applications.",
            "Rust provides memory safety without garbage collection using ownership system.",
            # AI/ML
            "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            # Database
            "Vector databases are specialized database systems designed to store and query high-dimensional vectors for similarity search.",
            "Time-series databases are optimized for sequential temporal data storage.",
            # Multi-category/complex
            "Building a real-time ML pipeline with Python and Vector DB.",
            "Implementing secure encryption in distributed databases."
        ]
        cls.test_metadatas = [
            {"category": "programming", "language": "python", "difficulty": "beginner", "doc_id": "doc1"},
            {"category": "programming", "language": "javascript", "difficulty": "intermediate", "doc_id": "doc2"},
            {"category": "programming", "language": "rust", "difficulty": "advanced", "doc_id": "doc3"},
            {"category": "ai", "field": "machine_learning", "difficulty": "intermediate", "doc_id": "doc4"},
            {"category": "ai", "field": "deep_learning", "difficulty": "advanced", "doc_id": "doc5"},
            {"category": "database", "type": "vector", "feature": "similarity_search", "doc_id": "doc6"},
            {"category": "database", "type": "time_series", "feature": "temporal_storage", "doc_id": "doc7"},
            {"category": ["programming", "ai", "database"], "languages": ["python"], "technologies": ["ml", "vector_db"], "difficulty": "advanced", "doc_id": "doc8"},
            {"category": ["programming", "database", "security"], "field": "cryptography", "difficulty": "advanced", "feature": "encryption", "doc_id": "doc9"}
        ]

    @classmethod
    def tearDownClass(cls):
        """Clean up by deleting test indexes"""
        for index_name in cls.test_indexes:
            try:
                cls.vx.delete_index(name=index_name)
                print(f"Successfully deleted test index: {index_name}")
            except Exception as e:
                if "not found" not in str(e).lower():
                    print(f"Error deleting test index {index_name}: {e}")

    def setUp(self):
        pass

    def tearDown(self):
        # Clean up any indexes created during the test
        try:
            indexes = self.vx.list_indexes()
            if isinstance(indexes, list):
                for index in indexes:
                    if isinstance(index, dict) and 'name' in index:
                        index_name = index['name']
                        if index_name.startswith("test_langchain_index_"):
                            try:
                                self.vx.delete_index(name=index_name)
                                print(f"Cleaned up test index: {index_name}")
                            except Exception as e:
                                print(f"Error cleaning up test index {index_name}: {e}")
            elif isinstance(indexes, dict):
                for index_name in indexes.values():
                    if isinstance(index_name, str) and index_name.startswith("test_langchain_index_"):
                        try:
                            self.vx.delete_index(name=index_name)
                            print(f"Cleaned up test index: {index_name}")
                        except Exception as e:
                            print(f"Error cleaning up test index {index_name}: {e}")
        except Exception as e:
            print(f"Error listing indexes for cleanup: {e}") 