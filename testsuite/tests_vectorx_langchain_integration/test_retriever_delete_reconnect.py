import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from setup_class import VectorXLangChainTestSetup
from vecx_langchain import VectorXVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

class TestRetrieverDeleteReconnect(VectorXLangChainTestSetup):
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

    def test_retriever(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        retrieved_docs = retriever.invoke("What is machine learning?")
        self.assertIsNotNone(retrieved_docs)
        self.assertGreater(len(retrieved_docs), 0)
        found = any("machine learning" in doc.page_content.lower() for doc in retrieved_docs)
        self.assertTrue(found, "No machine learning content found in retriever results")

    def test_delete_by_filter_and_verify(self):
        # Use $in to match both string and list values for 'category'
        filter_to_delete = {"category": {"$in": ["programming"]}}
        self.vector_store.delete(filter=filter_to_delete)
        # Verify deletion by searching for programming content
        programming_query = "JavaScript programming"
        results_after_filter_delete = self.vector_store.similarity_search(programming_query, k=2)
        # Should not find programming docs
        for doc in results_after_filter_delete:
            self.assertFalse(
                (isinstance(doc.metadata.get("category", None), str) and "programming" in doc.metadata["category"].lower()) or
                (isinstance(doc.metadata.get("category", None), list) and "programming" in [c.lower() for c in doc.metadata["category"]]),
                "Programming document was not deleted"
            )

    def test_reconnect_and_persistence(self):
        # Reconnect to the same index
        reconnected_store = VectorXVectorStore.from_params(
            embedding=self.embed_model,
            api_token=self.vecx_api_token,
            encryption_key=self.encryption_key,
            index_name=self.test_index_name,
            dimension=self.dimension,
            space_type=self.space_type
        )
        # Search for a known AI document
        results = reconnected_store.similarity_search("What is machine learning?", k=1)
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        found = any("machine learning" in doc.page_content.lower() for doc in results)
        self.assertTrue(found, "Reconnected store did not return expected document")

if __name__ == '__main__':
    unittest.main() 