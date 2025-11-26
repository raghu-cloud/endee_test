import unittest
import sys
import os
import pytest
from pathlib import Path

# Add parent directory to Python path to import local modules
sys.path.append(str(Path(__file__).parent.parent))

from setup_class import VectorXTestSetup
from vecx_llamaindex import VectorXVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



class TestVectorXVectorStore(VectorXTestSetup):
    def setUp(self):
        """Set up embedding model and other test-specific resources"""
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

    def test_create_vector_store_from_params(self):
        """Test creating VectorX vector store using from_params method"""
        try:
            # Create vector store
            vector_store = VectorXVectorStore.from_params(
                api_token=self.vecx_api_token,
                index_name=self.test_index_name,
                encryption_key=self.encryption_key,
                dimension=self.dimension,
                space_type=self.space_type
            )
            
            # Verify vector store attributes
            self.assertIsNotNone(vector_store)
            self.assertEqual(vector_store.api_token, self.vecx_api_token)
            self.assertEqual(vector_store.encryption_key, self.encryption_key)
            self.assertEqual(vector_store.index_name, self.test_index_name)
            self.assertEqual(vector_store.dimension, self.dimension)
            self.assertEqual(vector_store.space_type, self.space_type)
            
            # Verify index exists in VectorX
            index = self.vx.get_index(name=self.test_index_name, key=self.encryption_key)
            self.assertIsNotNone(index)
            
            # Verify index properties
            index_info = index.describe()
            self.assertEqual(index_info["dimension"], self.dimension)
            self.assertEqual(index_info["space_type"], self.space_type)
            
        except Exception as e:
            self.fail(f"Vector store creation failed: {str(e)}")

    def test_create_vector_store_with_documents(self):
        """Test creating vector store and adding documents"""
        try:
            # Create vector store
            vector_store = VectorXVectorStore.from_params(
                api_token=self.vecx_api_token,
                index_name=self.test_index_name,
                encryption_key=self.encryption_key,
                dimension=self.dimension,
                space_type=self.space_type
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index with documents
            index = VectorStoreIndex.from_documents(
                self.test_documents,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            # Verify index was created
            self.assertIsNotNone(index)
            
            # Verify index properties
            vectorx_index = self.vx.get_index(name=self.test_index_name, key=self.encryption_key)
            index_info = vectorx_index.describe()
            self.assertEqual(index_info["dimension"], self.dimension)
            self.assertEqual(index_info["space_type"], self.space_type)
            
            # Verify documents were added by performing a search
            # Get embedding for a test query
            test_query = "Python programming"
            query_embedding = self.embed_model.get_text_embedding(test_query)
            
            # Perform search
            results = vectorx_index.query(vector=query_embedding, top_k=len(self.test_documents))
            self.assertGreater(len(results), 0, "No vectors found in index after adding documents")
            
        except Exception as e:
            self.fail(f"Vector store creation with documents failed: {str(e)}")

    def test_create_vector_store_invalid_params(self):
        """Test vector store creation with invalid parameters"""
        
        # Test with invalid dimension (using unique index name to ensure creation)
        unique_index_name = f"{self.test_index_name}_invalid_dim"
        self.test_indexes.add(unique_index_name)  # Track for cleanup
        
        with pytest.raises(Exception) as exc_info:
            VectorXVectorStore.from_params(
                api_token=self.vecx_api_token,
                encryption_key=self.encryption_key,
                index_name=unique_index_name,
                dimension=-1,  # Invalid dimension
                space_type=self.space_type
            )
        error_msg = str(exc_info.value).lower()
        assert any(msg in error_msg for msg in ["dimension must be between", "invalid dimension"])
        
        # Test with invalid space type
        unique_index_name = f"{self.test_index_name}_invalid_space"
        self.test_indexes.add(unique_index_name)  # Track for cleanup
        
        with pytest.raises(ValueError) as exc_info:
            VectorXVectorStore.from_params(
                api_token=self.vecx_api_token,
                encryption_key=self.encryption_key,
                index_name=unique_index_name,
                dimension=self.dimension,
                space_type="invalid_space"  # Invalid space type
            )
        error_msg = str(exc_info.value).lower()
        assert "space" in error_msg and "invalid" in error_msg
        
        # Test with invalid API token
        unique_index_name = f"{self.test_index_name}_invalid_token"
        self.test_indexes.add(unique_index_name)  # Track for cleanup
        
        with pytest.raises(Exception) as exc_info:
            VectorXVectorStore.from_params(
                api_token="invalid:invalid_key:india-west-1",
                encryption_key=self.encryption_key,
                index_name=unique_index_name,
                dimension=self.dimension,
                space_type=self.space_type
            )
        error_msg = str(exc_info.value).lower()
        assert "invalid token" in error_msg or "unauthorized" in error_msg  # Accept either message
        
        # Test with missing required parameters
        with pytest.raises(Exception) as exc_info:
            VectorXVectorStore.from_params(
                encryption_key=self.encryption_key,
                dimension=self.dimension,
                space_type=self.space_type
            )
        error_msg = str(exc_info.value).lower()
        assert "nonetype" in error_msg  # Match the actual error for None api_token

        with pytest.raises(Exception) as exc_info:
            VectorXVectorStore.from_params(
                api_token=self.vecx_api_token,
                encryption_key=self.encryption_key,
                dimension=self.dimension,
                space_type=self.space_type
            )
        error_msg = str(exc_info.value).lower()
        assert "nonetype" in error_msg  # Match the actual error for None index_name

if __name__ == '__main__':
    unittest.main()
