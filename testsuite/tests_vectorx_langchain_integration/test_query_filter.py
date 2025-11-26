import unittest
from setup_class import VectorXLangChainTestSetup
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from vecx_langchain import VectorXVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

class TestVectorXQueryAndFilter(VectorXLangChainTestSetup):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        cls.vector_store = VectorXVectorStore.from_params(
            embedding=cls.embed_model,
            api_token=cls.vecx_api_token,
            index_name=cls.test_index_name,
            encryption_key=cls.encryption_key,
            dimension=cls.dimension,
            space_type=cls.space_type
        )
        # Add test documents
        cls.vector_store.add_texts(texts=cls.test_texts, metadatas=cls.test_metadatas)

    def test_basic_query(self):
        query = "What is Python?"
        results = self.vector_store.similarity_search(query, k=2)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        python_found = any("python" in doc.page_content.lower() for doc in results)
        self.assertTrue(python_found, "No Python-related content found in results")

    def test_single_filter(self):
        query = "What programming languages are mentioned?"
        filter_dict = {"category": {"$eq": "programming"}}
        results = self.vector_store.similarity_search(query, k=3, filter=filter_dict)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        all_prog = all("programming" in str(doc.metadata.get("category", "")).lower() or
                       (isinstance(doc.metadata.get("category", None), list) and "programming" in [c.lower() for c in doc.metadata["category"]])
                       for doc in results)
        self.assertTrue(all_prog, "Found results not from programming category")

    def test_multiple_filters(self):
        query = "Tell me about AI"
        filter_dict = {"category": {"$eq": "ai"}, "difficulty": {"$eq": "advanced"}}
        results = self.vector_store.similarity_search(query, k=3, filter=filter_dict)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        for doc in results:
            self.assertIn("ai", str(doc.metadata.get("category", "")).lower() or str(doc.metadata.get("category", [])), "Category filter failed")
            self.assertIn("advanced", str(doc.metadata.get("difficulty", "")).lower(), "Difficulty filter failed")

    def test_in_operator(self):
        query = "Tell me about programming languages"
        filter_dict = {"difficulty": {"$in": ["intermediate", "advanced"]}}
        results = self.vector_store.similarity_search(query, k=5, filter=filter_dict)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        for doc in results:
            self.assertIn(doc.metadata.get("difficulty", "").lower(), ["intermediate", "advanced"])

    def test_invalid_filter(self):
        query = "What will I get?"
        filter_dict = {"non_existent": {"$eq": "something"}}
        results = self.vector_store.similarity_search(query, k=2, filter=filter_dict)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 0, "Expected no results for invalid filter")

    def test_no_results_filter(self):
        query = "Tell me about programming languages"
        filter_dict = {"category": {"$eq": "ai"}, "difficulty": {"$eq": "beginner"}}
        results = self.vector_store.similarity_search(query, k=3, filter=filter_dict)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 0, "Expected no results for this filter combination")

if __name__ == '__main__':
    unittest.main() 