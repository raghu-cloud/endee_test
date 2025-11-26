import logging
from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Optional, cast
from crewai.memory.storage.rag_storage import RAGStorage
from datetime import datetime
import uuid


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
            if encryption_key is None:
                raise ValueError("Encryption key must be provided if vectorx_index is not provided")
        self._api_token=api_token
        self._encryption_key=encryption_key
        self._space_type=space_type
        self._text_key=text_key
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
            index = vx.get_index(name=self.type, key=self._encryption_key)
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
            vx.create_index(
                name=self.type,
                dimension=self._dimension,
                key=self._encryption_key,
                space_type=self._space_type,
            )
            self._vectorx_index=vx.get_index(name=self.type, key=self._encryption_key)
    
    def save(self, value: str, metadata: Dict[str, Any]) -> None:
        # Limit the document length to avoid it being too large for the model
        value = self._normalize_text(value)
        embedding = self.embedder_config([value])[0]
        metadata[self._text_key]=value
        try:
            event={
                "id":uuid.uuid4().hex,
                "vector":embedding,
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
        try:
            results = self._vectorx_index.query(
                vector=embedding,
                top_k=limit,
                include_vectors=True
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
                   "score":result["similarity"]
                }
            )
        return final_results
    
    def reset(self) -> None:
        if self.app:
            try:
                self.app.delete_index(self.type)
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




        



