import logging
from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Optional, cast
from crewai.memory.storage.rag_storage import RAGStorage
from datetime import datetime
import uuid
from fastembed import SparseTextEmbedding


def _import_vectorx() -> Any:
    """
    Try to import vectorx module. If it's not already installed, instruct user how to install.
    """
    try:
        import vecx
        from vecx.vectorx import VectorX
    except ImportError as e:
        raise ImportError(
            "Could not import vectorx python package. "
            "Please install it with `pip install vecx`."
        ) from e
    return vecx
vecx = _import_vectorx()
from vecx.vectorx import VectorX


_logger = logging.getLogger(__name__)

import_err_msg = (
    "`vectorx` package not found, please run `pip install vecx` to install it.`"
)
class VectorXVectorStore(RAGStorage):
    
    TEST_STRING = "test"  
    MAX_LENGTH_BYTES = 8192
    app: VectorX | None = None
    def __init__(
        self,
        type: str,
        allow_reset: bool = True,
        crew: Any = None,
        embedder_config: Optional[Any] = None,
        vectorx_index: Optional[Any] = None,
        text_key: str = "value",
        api_token: Optional[str] = None,
        encryption_key: Optional[str] = None,
        space_type: str = "cosine",
    ):
        if vectorx_index is None:
            if api_token is None:
                raise ValueError("API token must be provided if vectorx_index is not provided")
            # if encryption_key is None:
            #     raise ValueError("Encryption key must be provided if vectorx_index is not provided")
        self._api_token=api_token
        self._encryption_key=encryption_key
        self._space_type=space_type
        self._text_key=text_key
        self._vocab_size=30522
        self._sparse_embedder=SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
        super().__init__(type, allow_reset, embedder_config, crew)

    #here in crewai rag storage only internally calls this _initialize_app to intialize the index so we don't need to intailize the index in __int__
    def _initialize_app(self):
        """Initialize VectorX index using the current API."""
        # Initialize VectorX client
        self._set_embedder_config()
        #To get the dimesion of the embedding using the provided embedder_config we are using a test string to get the dimension
        self._dimension=len(self.embedder_config([self.TEST_STRING])[0])
        vx = VectorX(token=self._api_token)
        self.app=vx
        try:
            # Try to get existing index
            index = vx.get_hybrid_index(name=self.type,key=self._encryption_key)
            self._vectorx_index=index
            _logger.info(f"Retrieved existing index: {self.type}")
            return index
        except Exception as e:
            if self._dimension is None:
                raise ValueError(
                    "Must provide dimension when creating a new index"
                ) from e
            
            # Create a new index if it doesn't exist
            _logger.info(f"Creating new index: {self.type}")
            vx.create_hybrid_index(
                name=self.type,
                dimension=self._dimension,
                space_type=self._space_type,  
                vocab_size=self._vocab_size,  
                key=self._encryption_key
            )
            self._vectorx_index=vx.get_hybrid_index(name=self.type,key=self._encryption_key)
    
    def save(self, value: str, metadata: Dict[str, Any]) -> None:
        # Limit the document length to avoid it being too large for the model
        value = self._normalize_text(value)
        sparse_vector_obj=self._sparse_embedder.embed([value])
        sparse_vector = list(sparse_vector_obj)[0]
        indices=list(sparse_vector.indices)
        values=list(sparse_vector.values)
        indices=[int(index) for index in indices]
        values=[int(value) for value in values ]
        sparse_vector={
            "indices":indices,
            "values":values,
        }
        embedding = self.embedder_config([value])[0]
        metadata[self._text_key]=value
        try:
            event={
                "id":uuid.uuid4().hex,
                "dense_vector":embedding,
                "sparse_vector":sparse_vector,
                "meta":metadata,
            }
            res=self._vectorx_index.upsert([event])
        except Exception as e:
            _logger.error("There is some error in saving the data:",str(e))
    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0,
    ) -> list[dict]:
        # Limit the text length to avoid the document being too large for the model
        # Embed the text and search for similar points
        query = self._normalize_text(query)
        embedding = self.embedder_config([query])[0]
        sparse_vector_obj=self._sparse_embedder.embed([query])
        sparse_vector = list(sparse_vector_obj)[0]
        indices=list(sparse_vector.indices)
        values=list(sparse_vector.values)
        indices=[int(index) for index in indices]
        values=[int(value) for value in values ]
        sparse_vector={
            "indices":indices,
            "values":values,
        }
        try:
            results = self._vectorx_index.search(
        dense_vector=embedding,
        sparse_vector=sparse_vector,
        sparse_top_k=limit, 
        dense_top_k=limit,   
        include_vectors=False,  
        rrf_k=60,
        filter_query="{}"
    )
        except Exception as e:
            _logger.error(f"Error querying VectorX: {e}")
            return []
        
        final_results=[]
        for result in results:
            final_results.append(
                {
                   "id":result["id"],
                   "metadata":result["meta"],
                   "context":result["meta"].get(self._text_key, None),
                   "score":result['rrf_score']
                }
            )
        return final_results
    
    def reset(self) -> None:
        if self.app:
            try:
                self.app.delete_hybrid_index(self.type)
                _logger.info(f"Reset/deleted index: {self.type}")
            except Exception as e:
                _logger.error(f"Error resetting index: {e}")
        self._vectorx_index = None
    def _normalize_text(self, text: str) -> str:
        """
        Normalize the text to be within the maximum length.
        :param text:
        :return:
        """
        text = text.encode("utf-8")[: self.MAX_LENGTH_BYTES]
        text = text.decode("utf-8")
        return text




        



