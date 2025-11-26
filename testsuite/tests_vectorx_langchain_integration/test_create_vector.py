import unittest
import sys
from pathlib import Path
from setup_class import VectorXLangChainTestSetup

sys.path.append(str(Path(__file__).parent.parent))

from vecx_langchain import VectorXVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings 

class TestVectorXVectorStore(VectorXLangChainTestSetup):
    def setUp(self):
        self.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

    def test_create_vector_store_from_params(self):
        """Test creating VectorX vector store using from_params method"""
        try:
            vector_store = VectorXVectorStore.from_params(
                embedding=self.embed_model,
                api_token=self.vecx_api_token,
                index_name=self.test_index_name,
                encryption_key=self.encryption_key,
                dimension=self.dimension,
                space_type=self.space_type
            )
            self.assertIsNotNone(vector_store)
            self.assertEqual(vector_store._vectorx_index.name, self.test_index_name)
            self.assertEqual(vector_store._vectorx_index.dimension, self.dimension)
            self.assertEqual(vector_store._vectorx_index.space_type, self.space_type)
            # Verify index exists in VectorX
            index = self.vx.get_index(name=self.test_index_name, key=self.encryption_key)
            self.assertIsNotNone(index)
            index_info = index.describe()
            self.assertEqual(index_info["dimension"], self.dimension)
            self.assertEqual(index_info["space_type"], self.space_type)
        except Exception as e:
            self.fail(f"Vector store creation failed: {str(e)}")

    def test_create_vector_store_with_documents(self):
        """Test creating vector store and adding documents"""
        try:
            vector_store = VectorXVectorStore.from_params(
                embedding=self.embed_model,
                api_token=self.vecx_api_token,
                index_name=self.test_index_name,
                encryption_key=self.encryption_key,
                dimension=self.dimension,
                space_type=self.space_type
            )
            ids = vector_store.add_texts(texts=self.test_texts, metadatas=self.test_metadatas)
            self.assertEqual(len(ids), len(self.test_texts))
            # Verify documents were added by performing a search
            test_query = "Python programming"
            results = vector_store.similarity_search(test_query, k=len(self.test_texts))
            self.assertGreater(len(results), 0, "No vectors found in index after adding documents")
        except Exception as e:
            self.fail(f"Vector store creation with documents failed: {str(e)}")

    def test_create_vector_store_invalid_params(self):
        """Test vector store creation with invalid parameters"""
        # Invalid dimension
        unique_index_name = f"{self.test_index_name}_invalid_dim"
        self.test_indexes.add(unique_index_name)
        with self.assertRaises(Exception):
            VectorXVectorStore.from_params(
                embedding=self.embed_model,
                api_token=self.vecx_api_token,
                index_name=unique_index_name,
                encryption_key=self.encryption_key,
                dimension=-1,
                space_type=self.space_type
            )
        # Invalid space type
        unique_index_name = f"{self.test_index_name}_invalid_space"
        self.test_indexes.add(unique_index_name)
        with self.assertRaises(Exception):
            VectorXVectorStore.from_params(
                embedding=self.embed_model,
                api_token=self.vecx_api_token,
                index_name=unique_index_name,
                encryption_key=self.encryption_key,
                dimension=self.dimension,
                space_type="invalid_space"
            )
        # Invalid API token
        unique_index_name = f"{self.test_index_name}_invalid_token"
        self.test_indexes.add(unique_index_name)
        with self.assertRaises(Exception):
            VectorXVectorStore.from_params(
                embedding=self.embed_model,
                api_token="invalid:invalid_key:india-west-1",
                index_name=unique_index_name,
                encryption_key=self.encryption_key,
                dimension=self.dimension,
                space_type=self.space_type
            )
        # Missing required parameters
        with self.assertRaises(Exception):
            VectorXVectorStore.from_params(
                embedding=self.embed_model,
                encryption_key=self.encryption_key,
                dimension=self.dimension,
                space_type=self.space_type
            )
        with self.assertRaises(Exception):
            VectorXVectorStore.from_params(
                embedding=self.embed_model,
                api_token=self.vecx_api_token,
                encryption_key=self.encryption_key,
                dimension=self.dimension,
                space_type=self.space_type
            )

if __name__ == '__main__':
    unittest.main() 