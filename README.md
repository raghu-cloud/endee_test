# VectorX Testing Documentation

## Overview

These test suites are designed to validate the basic functionalities of the VectorX API, including operations related to index creation, vector upsert, querying vectors, and deleting vectors and indexes. The tests ensure that operations behave correctly under different scenarios, including positive, negative, and edge cases. Each test case is grouped into specific categories based on the type of operation being tested.

## Required Libraries and Modules for Test Execution

### 1. pytest
A testing framework used for writing and running tests. Used for defining test functions and running parameterized tests (e.g., `pytest.mark.parametrize`).

### 2. os
Interacts with the operating system, such as handling environment variables. Used to fetch environment variables like `VECTORX_API_TOKEN` (e.g., `os.getenv("VECTORX_API_TOKEN")`).

### 3. sys
Provides access to system-specific parameters and functions. Used for managing standard input/output (e.g., `sys.stdout` for logging).

### 4. time
Provides time-related functions like delays. Usage: `time.sleep(2)` is used to introduce a 2-second delay before tests.

### 5. VectorX
```python
from vecx.vectorx import VectorX
```
Imports the VectorX client class for interacting with a vector database. Used to create and manage vector indexes (e.g., `self.vx.create_index()`).

### 6. dotenv
```python
from dotenv import load_dotenv
```
Loads environment variables from a `.env` file into the environment. `load_dotenv()` loads variables like API keys into the environment.

### 7. logging
Handles logging messages for tracking the execution of the program. Logs messages at different levels (e.g., `logger.info()`, `logger.warning()`).

### 8. TestConfig
```python
from config.test_config import TestConfig
```
Imports configuration settings or helper methods from the `test_config` module. Provides constants and methods for generating index names and timestamps.

### 9. builtins
Accesses Python's built-in functions and variables. Retrieves environment variables (e.g., `getattr(builtins, "VECTORX_API_KEY", None)`).

### 10. random
Generates random numbers and selects random items. For creating random test data (e.g., `random.randint(1, 100)`).

### 11. json
Handles JSON parsing and serialization. Converts Python objects to JSON and vice versa (e.g., `json.loads(response.text)`).

### 12. numpy
```python
import numpy as np
```
Provides support for arrays and numerical operations. For creating and manipulating arrays and vectors in tests (e.g., `np.random.rand(10)`).

## Environment Setup and Logging Configuration

This section explains how the test setup loads important settings like API keys from files or system variables, gets test run details, and sets up logging to show useful messages while tests run.

### 1. Loading Environment Variables with load_dotenv()

This function loads environment variables from a `.env` file into the Python environment. It ensures sensitive data like API tokens can be configured externally without hardcoding them.

```python
load_dotenv()  # Load .env vars
```

### 2. Retrieving API Token from builtins

The code first attempts to get the API token from a dynamic attribute in builtins. This allows the test framework or CI pipeline to inject the token programmatically at runtime.

```python
VECTORX_API_TOKEN = getattr(builtins, "VECTORX_API_KEY", None)
```

### 3. Falling Back to System Environment Variables

If the token isn't found in builtins, it falls back to reading the `VECTORX_API_TOKEN` from the system's environment variables, typically loaded from the `.env` file or OS environment.

```python
if VECTORX_API_TOKEN is None:
    VECTORX_API_TOKEN = os.getenv("VECTORX_API_TOKEN")
```

### 4. Fetching Test Run Timestamp

The timestamp variable captures a unique identifier for the current test execution, usually injected during automated test runs to correlate resources and logs.

```python
timestamp = getattr(builtins, "TEST_RUN_TIMESTAMP", None)
```

### 5. Creating and Configuring the Logger

A logger specific to the current module is created and set to the INFO logging level, enabling informative messages to be captured during test execution.

```python
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
```

### 6. Adding a Stream Handler to Logger

To avoid duplicate handlers, it checks if handlers exist. If none are found, it attaches a StreamHandler to output logs to the console with a clear format showing level, logger name, and message.

```python
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
```

## Index Creation

### Overview

This testing suite is designed to ensure the correct functionality of the index creation in the VectorX system. The tests validate various edge cases, parameter combinations, and the handling of invalid inputs. The tests also include encryption functionality and proper index cleanup. The testing suite uses pytest as the testing framework and is structured into several classes and methods to ensure organized and modular tests.

### Test Class: TestCreateIndex

The main test class is `TestCreateIndex`, which includes methods for setting up and tearing down tests, along with the actual test cases.

#### 1. Class-level Setup and Teardown

**setup_class**: This class-level setup is executed once before any tests in the class are run. It initializes the VectorX instance, generates an encryption key, and cleans up any existing indexes.

```python
@classmethod
def setup_class(cls):
    cls.vx = VectorX(token=VECTORX_API_TOKEN)
    cls.encryption_key = cls.vx.generate_key()
    for index in cls.vx.list_indexes()['indixes']:
        cls.vx.delete_index(index['name'])
```

**teardown_class**: This class-level teardown is executed once after all tests in the class have run. It ensures that all indexes are deleted after tests are complete.

```python
@classmethod
def teardown_class(cls):
    for index in cls.vx.list_indexes()['indixes']:
        cls.vx.delete_index(index['name'])
    logger.info("Tests completed")
```

#### 2. Test Method Setup and Teardown

**setup_method**: This method is executed before each individual test, ensuring that each test runs with a fresh state. A unique index name is generated by appending a timestamp and unique ID.

```python
def setup_method(self):
    """Setup before each test"""
    self.test_index_name = f"{TestConfig.TEST_INDEX_PREFIX}create_{TestConfig.get_timestamp()}_{TestConfig.get_unique_id()}"
    time.sleep(2)
```

**teardown_method**: After each test is completed, this method is executed. It ensures any created indexes are cleaned up.

```python
def teardown_method(self):
    """Cleanup after each test"""
    index_lst = self.vx.list_indexes()
    if len(index_lst['indixes']) == 5:
        for index_name in self.cleanup_indexes:
            try:
                self.vx.delete_index(index_name)
                logger.info(f"Deleted index: {index_name}")
            except Exception as e:
                logger.warning(f"Failed to delete index '{index_name}': {e}")
        self.cleanup_indexes.clear()
```

### Test Cases

#### 1. Test Create Index with Missing Parameters

**Description**: This test case ensures that missing required parameters for index creation raise a TypeError with the appropriate error message. The test checks the following:
- Missing dimension
- Missing space_type  
- Missing name

```python
def test_create_index_missing_parameters(self):
    """Test that missing required parameters raise TypeError with correct messages."""
    # Test missing 'dimension'
    with pytest.raises(TypeError) as exc_info:
        self.vx.create_index(name="test_index", space_type="cosine")
    assert "missing 1 required positional argument: 'dimension'" in str(exc_info.value)

    # Test missing 'space_type'
    with pytest.raises(TypeError) as exc_info:
        self.vx.create_index(name="test_index", dimension=768)
    assert "missing 1 required positional argument: 'space_type'" in str(exc_info.value)

    # Test missing 'name'
    with pytest.raises(TypeError) as exc_info:
        self.vx.create_index(dimension=768, space_type="cosine")
    assert "missing 1 required positional argument: 'name'" in str(exc_info.value)
```

#### 2. Test Create Index with Invalid Name

**Description**: This test case validates the index name, ensuring that only valid names are accepted. It checks both invalid characters and the length of the name.

```python
def test_create_index_invalid_name(self):
    invalid_names = ["invalid name", "invalid@name", "x" * 49, ""]
    for name in invalid_names:
        with pytest.raises(ValueError):
            self.vx.create_index(name=name, dimension=768, space_type="cosine")

    wrong_types = [None, 123, ["invalid"], {"name": "invalid"}]
    for name in wrong_types:
        with pytest.raises(TypeError):
            self.vx.create_index(name=name, dimension=768, space_type="cosine")
```

#### 3. Test Create Index with Parameter Combinations

**Description**: This parameterized test case checks the index creation with all combinations of several parameters, such as:
- dimension (e.g., 5 or 768)
- space_type (e.g., 'cosine', 'l2', 'ip')
- M (e.g., 16, 32, 64)
- ef_con (e.g., 128, 256)
- use_fp16 (True or False)
- encryption (True or False)

This ensures that all valid combinations of parameters work as expected.

```python
@pytest.mark.parametrize("dimension", [5, 768])
@pytest.mark.parametrize("space_type", ['cosine', 'l2'])
@pytest.mark.parametrize("M", [16, 32])
@pytest.mark.parametrize("use_fp16", [True, False])
def test_create_index_combinations(self, dimension, space_type, M, use_fp16):
    index_name = f"test_index_{dimension}_{space_type}"
    result = self.vx.create_index(name=index_name, dimension=dimension, 
                                  space_type=space_type, M=M, use_fp16=use_fp16)
    assert result == "Index created successfully"
    info = self.vx.get_index(index_name).describe()
    assert info["dimension"] == dimension
```

## Upsertion

### Overview

This testing suite is designed to ensure the correct functionality of the upsertion process in the VectorX system. The tests validate vector upsertion for various edge cases, including parameter combinations, metadata validation, encryption functionality, and batch upserts. The tests also cover invalid inputs such as dimension mismatches, empty IDs, and empty vectors. This suite uses pytest as the testing framework, and it's organized into several classes and methods for modular tests.

### Test Class: TestUpsertVectors

The main test class is `TestUpsertVectors`, which includes methods for setting up and tearing down tests along with the actual test cases.

#### 1. Class-level Setup and Teardown

**setup_class**: This class-level setup is executed once before any tests in the class are run. It initializes the VectorX instance, generates an encryption key, cleans up any existing indexes, and creates the necessary test indexes.

```python
@classmethod
def setup_class(cls):
    cls.vx = VectorX(token=VECTORX_API_TOKEN)
    cls.encryption_key = cls.vx.generate_key()

    with open(f"config/tmp_encryption_key_{timestamp}.txt", "w") as f:
        f.write(cls.encryption_key)

    # Delete leftover test indexes
    index_lst = cls.vx.list_indexes()
    for index in index_lst['indixes']:
        cls.vx.delete_index(index['name'])

    # Create new test indexes according to configuration
    cls.index_configs = [...]
    for config in cls.index_configs:
        create_kwargs = {...}
        result = cls.vx.create_index(**create_kwargs)
        assert result in ("Index created successfully", "Index already exists")

    time.sleep(2)
```

**teardown_class**: This method would clean up the test indexes after all tests are complete. Optional if the same vector has to be used for the next querying stage.

```python
@classmethod
def teardown_class(cls):
    indexes = cls.vx.list_indexes()
    for idx in indexes.get("indixes", []):
        if idx["name"].startswith(TestConfig.TEST_INDEX_PREFIX):
            cls.vx.delete_index(idx["name"])
    logger.info("Deleted all test indexes after test run.")
```

#### 2. Test Method Setup

**setup_method**: This method is executed before each individual test, ensuring that each test runs with a fresh state. It fetches the necessary indexes for testing.

```python
def setup_method(self):
    self.index_no_enc_5 = self.vx.get_index(name=TestConfig.TEST_UPSERT_INDEX1)
    self.index_no_enc_768 = self.vx.get_index(name=TestConfig.TEST_UPSERT_INDEX2)
    self.index_enc_5 = self.vx.get_index(name=TestConfig.TEST_UPSERT_INDEX3, key=self.encryption_key)
    self.index_enc_768 = self.vx.get_index(name=TestConfig.TEST_UPSERT_INDEX4, key=self.encryption_key)
```

#### 3. Helper Methods

**_generate_vector**: This helper function generates a random vector with metadata and a filter. Also it ensures vectors are normalized when the space_type is "ip".

```python
def _generate_vector(self, id_suffix: str, dim: int, title: str = "Test Vector", visibility=None, space_type=None):
    vec = np.random.rand(dim)
    if space_type == "ip":
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
    return {
        "id": f"vec_{id_suffix}",
        "vector": vec.tolist(),
        "meta": {"title": title},
        "filter": {"visibility": visibility or random.choice(["public", "private"])}
    }
```

**_safe_get_vector**: This helper function safely retrieves a vector by ID, handling the case where the response might need parsing from a JSON string.

```python
def _safe_get_vector(self, index, vec_id):
    result = index.get_vector(vec_id)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            pytest.fail(f"Response for vector '{vec_id}' is not valid JSON: {result}")
    return result
```

### Test Cases

#### 1. Test Upsert 10 Vectors

**Description**: This parameterized test verifies that a batch of 10 vectors is upserted successfully into the system. It also checks that the upserted vectors can be retrieved correctly by their ID, ensuring proper insertion.

```python
@pytest.mark.parametrize("index_attr, dimension, encryption, space_type", [
    ("index_no_enc_5", 5, False, "cosine"),
    ("index_enc_5", 5, True, "ip"),
    ("index_no_enc_768", 768, False, "l2"),
    ("index_enc_768", 768, True, "cosine"),
])
def test_upsert_10_vectors(self, index_attr, dimension, encryption, space_type):
    idx = getattr(self, index_attr)
    vectors = [self._generate_vector(str(i), dimension, space_type=space_type) for i in range(10)]
    sample = vectors[5]

    result = idx.upsert(vectors)
    assert result in ("Vectors upserted successfully", "Vectors inserted successfully")

    retrieved = self._safe_get_vector(idx, sample["id"])
    assert retrieved["id"] == sample["id"]
    assert retrieved["meta"]["title"] == sample["meta"]["title"]
    assert len(retrieved["vector"]) == dimension
```

#### 2. Test Upsert with Dimension Mismatch

**Description**: This test ensures that upserting a vector with an incorrect dimension (not matching the index's dimension) raises an appropriate error. It validates that the system enforces dimension consistency during vector upsertion.

```python
def test_upsert_dimension_mismatch(self, index_attr, dimension, encryption, space_type):
    idx = getattr(self, index_attr)
    bad_vector = self._generate_vector("bad_dim", dim=dimension - 1, space_type=space_type)
    with pytest.raises(Exception) as e_info:
        idx.upsert([bad_vector])
    assert "dimension" in str(e_info.value).lower()
```

#### 3. Test Upsert with Invalid Empty ID

**Description**: This test checks that vectors with an empty ID raise an error when upserted. It ensures that the system rejects vectors without a valid identifier.

```python
def test_upsert_invalid_empty_id(self, index_attr, dimension, encryption, space_type):
    idx = getattr(self, index_attr)
    bad_vector = self._generate_vector("empty_id", dimension, space_type=space_type)
    bad_vector["id"] = ""
    with pytest.raises(Exception) as e_info:
        idx.upsert([bad_vector])
    assert "bad request" in str(e_info.value).lower() or "id" in str(e_info.value).lower()
```

#### 4. Test Upsert with Empty Vector

**Description**: This test validates that upserting a vector with an empty vector list (no vector data) triggers an error. The test ensures that the system rejects invalid vectors.

```python
def test_upsert_empty_vector(self, index_attr, dimension, encryption, space_type):
    idx = getattr(self, index_attr)
    bad_vector = self._generate_vector("empty_vec", dim=dimension, space_type=space_type)
    bad_vector["vector"] = []

    with pytest.raises(Exception) as e_info:
        idx.upsert([bad_vector])
    assert "vector dimension mismatch" in str(e_info.value).lower()
```

#### 5. Test Upsert and Retrieve Non-Existent ID

**Description**: This test checks that an attempt to retrieve a vector using a non-existent ID raises an exception. It ensures proper error handling for invalid vector retrievals.

```python
def test_upsert_and_retrieve_non_existent_id(self, index_attr, dimension, encryption, space_type):
    idx = getattr(self, index_attr)
    valid_vector = self._generate_vector("valid", dimension, space_type=space_type)
    idx.upsert([valid_vector])

    with pytest.raises(Exception) as e_info:
        idx.get_vector("vec_nonexistent")
    assert "not found" in str(e_info.value).lower()
```

#### 6. Test Large Batch Upsert

**Description**: This test validates the upsertion of large batches of vectors (e.g., 2000 vectors) in chunks. It ensures the system can handle large-scale vector upserts without failures.

```python
def test_large_batch_upsert(self, index_attr, dimension, encryption, space_type):
    idx = getattr(self, index_attr)
    num_vectors = 2000
    vectors = [self._generate_vector(str(i), dimension, space_type=space_type) for i in range(num_vectors)]

    for i in range(0, num_vectors, 1000):
        batch = vectors[i:i + 1000]
        idx.upsert(batch)
        logger.info(f"Upserted batch {i // 1000 + 1}")

    for i in [0, num_vectors // 2, num_vectors - 1]:
        vec_id = f"vec_{i}"
        retrieved = self._safe_get_vector(idx, vec_id)
        assert retrieved["id"] == vec_id
```

#### 7. Test Upsert Update Existing Vector

**Description**: This test ensures that when an existing vector is upserted with new data, the original metadata is replaced. It validates the system's ability to overwrite previously stored vector data with updated values.

```python
def test_upsert_update_existing_vector(self, index_attr, dimension, encryption, space_type):
    idx = getattr(self, index_attr)
    vec_id = "vec_update"
    original = self._generate_vector("update", dimension, title="Original Title", space_type=space_type)
    idx.upsert([original])
    
    updated = self._generate_vector("update", dimension, title="Updated Title", space_type=space_type)
    idx.upsert([updated])

    retrieved = self._safe_get_vector(idx, vec_id)
    assert retrieved["meta"]["title"] == "Updated Title"
```

#### 8. Test Upsert with List as Meta Type

**Description**: This test ensures that upserting a vector with metadata as a list is accepted. It validates that the system handles non-dictionary metadata types without raising errors.

```python
def test_upsert_list_meta_type(self, index_attr, dimension, encryption, space_type):
    idx = getattr(self, index_attr)
    bad_vector = self._generate_vector("bad_meta", dimension, space_type=space_type)
    bad_vector["meta"] = ["not", "a", "dict"]

    idx.upsert([bad_vector])
    retrieved = self._safe_get_vector(idx, bad_vector["id"])
    assert isinstance(retrieved["meta"], list)
```

#### 9. Test Upsert with Long Metadata

**Description**: This test checks if vectors with long metadata strings (such as lengthy titles) are accepted and stored correctly. The test ensures that the system can handle large strings in metadata fields without issues.

```python
def test_upsert_long_metadata(self, index_attr, dimension, encryption, space_type):
    idx = getattr(self, index_attr)
    long_title = "A" * 1000
    vector = self._generate_vector("long_meta", dimension, title=long_title, space_type=space_type)
    idx.upsert([vector])

    retrieved = self._safe_get_vector(idx, vector["id"])
    assert retrieved["meta"]["title"] == long_title
```

# VectorX Query Test Suite

## Overview

This test suite verifies the querying functionality of the VectorX system. It ensures correct handling of valid/invalid parameters, vector dimensions, encryption support, filtering, and top-k accuracy. Tests are organized using pytest in a modular format under the `TestQueryIndex` class.

## Pipeline Mode Context

The tests accommodate two operation modes:

### Standalone Mode (default)
- Indexes are created, populated, and queried directly within the test suite
- Encryption keys are generated dynamically
- Indexes are cleaned up after tests

### Pipeline Mode
- Indexing and encryption keys are pre-provisioned externally
- The test suite reads these keys and skips index creation
- Focuses only on querying behavior
- Supports efficient end-to-end testing in complex deployment pipelines

The mode is detected via a configuration file:
```python
with open(f"config/pipeline_mode_bool_{timestamp}.txt", "r") as f:
    cls.pipeline_mode = f.read().strip().lower() == "true"
```

## Test Class: TestQueryIndex

### Class-level Setup and Teardown

#### setup_class
- **Standalone Mode**: Sets up multiple test indexes with various configurations and populates them with vectors
- **Pipeline Mode**: Reads existing encryption keys and skips index creation to focus on querying

```python
@classmethod
def setup_class(cls):
    cls.vx = VectorX(token=VECTORX_API_TOKEN)
    if not cls.pipeline_mode:
        # Create and populate indexes
        cls.vx.create_index(name="index1", dimension=5, space_type="cosine")
        idx.upsert([...])
    else:
        # Load encryption key for queries only
        with open(f"config/tmp_encryption_key_{timestamp}.txt") as f:
            cls.encryption_key = f.read().strip()
```

#### teardown_class
- **Pipeline Mode**: Deletes indexes after tests to maintain the environment
- **Standalone Mode**: Indexes are already cleaned up post-testing

```python
@classmethod
def teardown_class(cls):
    if cls.pipeline_mode:
        index_lst = cls.vx.list_indexes()
        if len(index_lst['indixes'])>0:
            for index in index_lst['indixes']:
                cls.vx.delete_index(index['name'])
    logger.info("Testing Query index done")
```

### Test Method Setup

#### setup_method
Prepares commonly used index handles before each test method. Fetches all four test indexes (encrypted and unencrypted) with correct keys for use in query-based test cases.

```python
def setup_method(self):
    self.index_no_enc_5 = self.vx.get_index(name=TestConfig.TEST_UPSERT_INDEX1)
    self.index_no_enc_768 = self.vx.get_index(name=TestConfig.TEST_UPSERT_INDEX2)
    self.index_enc_5 = self.vx.get_index(name=TestConfig.TEST_UPSERT_INDEX3, key= self.encryption_key)
    self.index_enc_768 = self.vx.get_index(name=TestConfig.TEST_UPSERT_INDEX4, key=self.encryption_key)
```

## Test Cases

### 1. Test Missing Query Vector

**Description**: Confirms that omitting the mandatory vector parameter results in a TypeError. This enforces strict API usage regardless of pipeline mode.

```python
def test_missing_query_vector(self):
    with pytest.raises(TypeError) as exc_info:
        results = self.index_no_enc_5.query(
            top_k=5
        )
    assert "missing 1 required positional argument: 'vector'" in str(exc_info.value)
```

### 2. Test Vector Dimension Mismatch

**Description**: Checks that queries with vectors whose dimensions do not match the index's configured dimension raise a ValueError. Prevents invalid or nonsensical queries.

```python
def test_vector_dimension_mismatch(self):
    with pytest.raises(ValueError) as exc_info:
        results = self.index_no_enc_5.query(
            vector=[0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7],
            top_k=1
        )
    assert "Vector dimension mismatch:" in str(exc_info.value)
```

### 3. Test Invalid top_k Values

**Description**: Ensures that invalid top_k inputs (e.g., zero, negative, or exceeding max allowed) trigger an APIException with descriptive errors, protecting query stability.

```python
def test_invalid_top_k(self):
    invalid_top_k_values = [-1, 0, 4097, 10000]
    for invalid_top_k in invalid_top_k_values:
        with pytest.raises(APIException) as exc_info:
            results = self.index_enc_5.query(
                vector=[0.1, 0.2, 0.3, 0.5, 0.7],
                top_k= invalid_top_k
            )
        assert "k must be between 1 and 4096" in str(exc_info.value)
```

### 4. Test Valid top_k Vectors Count

**Description**: Validates that queries return the expected number of nearest neighbors for various values of top_k across different index configurations (encrypted and unencrypted, 5D and 768D). Also confirms the presence of required metadata fields like id, similarity, distance, and norm.

**Parameters**:
- `index_attr`: One of "index_no_enc_5", "index_no_enc_768", "index_enc_5", "index_enc_768"
- `top_k`: 5, 10, or 15

```python
@pytest.mark.parametrize("index_attr", [...])
@pytest.mark.parametrize("top_k", [5, 10, 15])
def test_valid_top_k_vectors_count(self, index_attr, top_k):
    index = getattr(self, index_attr)
    results = index.query(vector=query_vector, top_k=top_k)
    assert len(results) == min(top_k, index.describe()["count"])
```

### 5. Test Filter Match and Vector Parameters

**Description**: Ensures queries respect metadata filters (visibility = public/private) and return vectors and metadata when include_vectors=True. Runs across multiple index types and filter values to verify consistent filtering and data integrity.

**Parameters**:
- `index_attr`: One of "index_no_enc_5", "index_no_enc_768", "index_enc_5", "index_enc_768"
- `filter_sub_category`: "public" or "private"

```python
@pytest.mark.parametrize("index_attr", [...])
@pytest.mark.parametrize("filter_sub_category", ["public", "private"])
def test_filter_match_and_vector_parameters(self, index_attr, filter_sub_category):
    index = getattr(self, index_attr)
    results = index.query(..., filter={"visibility": {"eq": filter_sub_category}})
    for result in results:
        assert result["filter"]["visibility"] == filter_sub_category
```

## Test Configuration

The test suite uses four main test indexes:
- `index_no_enc_5`: Unencrypted 5-dimensional index
- `index_no_enc_768`: Unencrypted 768-dimensional index  
- `index_enc_5`: Encrypted 5-dimensional index
- `index_enc_768`: Encrypted 768-dimensional index

## Running the Tests

To run the query test suite:

```bash
pytest test_query_index.py -v
```

For pipeline mode testing, ensure the configuration file is properly set up before running the tests.


# VectorX Deletion Test Suite

## Overview

This test suite verifies the functionality for deleting vectors and indexes in VectorX, ensuring correct behavior for single and batch deletions, deletion with filters, and cleanup of indexes. It also covers negative cases like deletion attempts on nonexistent indexes or using invalid encryption keys. The tests support both standalone and pipeline modes, where pipeline mode skips setup and focuses on cleanup and deletion verification.

## Pipeline Mode Context

### Standalone Mode
- The test suite creates indexes, inserts vectors, and tests delete operations fully, including cleanup

### Pipeline Mode
- Assumes indexes and encryption keys are pre-provisioned
- Skips vector insertions and some deletion tests
- Focuses mainly on deleting indexes and environment cleanup
- Encryption keys are loaded from configuration files

Pipeline mode is detected as:
```python
with open(f"config/pipeline_mode_bool_{timestamp}.txt", "r") as f:
    cls.pipeline_mode = f.read().strip().lower() == "true"
```

## Test Class: TestDeleteVectorsAndIndexes

This class focuses on validating the deletion of vectors and indexes in the VectorX system. It handles setup and teardown at both the class and method level and contains positive and negative test scenarios.

### Class-Level Setup and Teardown

#### setup_class
Executed once before any tests run. This method initializes the VectorX client, determines the pipeline mode, and either loads or generates the encryption key. It also defines the test index configurations and sets up the necessary indexes and vectors if not running in pipeline mode.

```python
@classmethod
def setup_class(cls):
    cls.vx = VectorX(token=VECTORX_API_TOKEN)
    # Load pipeline mode and encryption key
    cls.encryption_key = cls.vx.generate_key() if not cls.pipeline_mode else ...
    # Define and create test indexes
    cls.index_configs = [...]
    if not cls.pipeline_mode:
        # Clean existing indexes, create new ones, and insert vectors
```

#### teardown_class
Executed once after all tests are completed. It ensures cleanup by deleting the test indexes and removing the temporary encryption key file if in pipeline mode. This method helps to maintain a clean environment for subsequent test runs.

```python
@classmethod
def teardown_class(cls):
    if cls.pipeline_mode:
        for idx in cls.test_indexes:
            cls.vx.delete_index(idx["name"])
        os.remove(f"config/tmp_encryption_key_{timestamp}.txt")
    logger.info("Teardown complete.")
```

### Helper Methods

#### get_test_index
**Description**: Fetches a test index from VectorX, applying the encryption key if required. Logs a warning on failure and re-raises the exception to ensure the test fails visibly.

```python
def get_test_index(self, name, encrypted):
    try:
        return self.vx.get_index(name=name, key=self.encryption_key if encrypted else None)
    except Exception as e:
        logger.warning(f"Failed to get index '{name}': {e}")
        raise
```

## Test Cases

### 1. Test Delete Vectors with Invalid Filter

**Description**: Checks that deleting vectors with an invalid filter does not raise exceptions, although it effectively does nothing. All tests are parameterized for all types of indexes.

**Parameters**:
- `index_attr`: Test index name
- `dimension`: 5 or 768
- `encryption`: True/False
- `space_type`: "cosine", "ip", or "l2"

```python
@pytest.mark.parametrize("index_attr, dimension, encryption, space_type", [
    (TestConfig.TEST_UPSERT_INDEX1, 5, False, "cosine"),
    (TestConfig.TEST_UPSERT_INDEX3, 5, True, "ip"),
    (TestConfig.TEST_UPSERT_INDEX2, 768, False, "l2"),
    (TestConfig.TEST_UPSERT_INDEX4, 768, True, "cosine"),
])
def test_delete_vectors_with_invalid_filter(self, index_attr, dimension, encryption, space_type):
    index = self.get_test_index(index_attr, encryption)
    index.delete_with_filter({"invalid_field": {"eq": "invalid_value"}})
    logger.info(f"Invalid filter delete on '{index_attr}'")
```

### 2. Test Delete Non-existent Vector

**Description**: Ensures that when deleting a vector with an ID that does not exist in the index, the system raises an exception indicating the vector was not found.

```python
def test_delete_nonexistent_vector(self, index_attr, dimension, encryption, space_type):
    index = self.get_test_index(index_attr, encryption)
    with pytest.raises(Exception) as exc_info:
        index.delete_vector("nonexistent_vector_id_12345")
    assert "not found" in str(exc_info.value).lower(), "Expected 'not found' error message"
```

### 3. Test Delete Single Vector

**Description**: Inserts a unique vector into each index, verifies its presence, deletes it, and confirms that fetching it afterward raises an exception. Skipped during pipeline mode to avoid interference.

```python
def test_delete_vector(self, index_attr, dimension, encryption, space_type):
    index = self.get_test_index(index_attr, encryption)
    vector = TestConfig.generate_vector(id_suffix="del_test", dim=dimension, space_type=space_type)
    index.upsert([vector])
    assert index.get_vector(vector["id"])["id"] == vector["id"]
    index.delete_vector(vector["id"])
    with pytest.raises(Exception):
        index.get_vector(vector["id"])
```

### 4. Test Delete Multiple Vectors

**Description**: Multiple vectors are upserted per index, then deleted one by one with checks to ensure they no longer exist. The test skips execution if pipeline mode is active.

```python
def test_delete_multiple_vectors(self, index_attr, dimension, encryption, space_type):
    index = self.get_test_index(index_attr, encryption)
    vectors = [TestConfig.generate_vector(id_suffix=TestConfig.get_unique_id(), dim=dimension, space_type=space_type) for _ in range(5)]
    index.upsert(vectors)
    for vec in vectors:
        index.delete_vector(vec["id"])
        with pytest.raises(Exception):
            index.get_vector(vec["id"])
```

### 5. Test Delete Vectors in Non-Existent Index

**Description**: Verifies that attempting to delete vectors from a non-existent index raises an exception.

```python
def test_delete_vectors_in_nonexistent_index(self):
    with pytest.raises(Exception) as exc_info:
        idx = self.vx.get_index("nonexistent_index_123456")
        idx.delete_with_filter({})
    assert "not found" in str(exc_info.value).lower()
```

### 6. Delete Test Indexes

**Description**: Removes all the test indexes themselves from the system, verifying their deletion by ensuring fetching afterward fails. Skipped in pipeline mode.

```python
def test_delete_test_indexes(self, index_attr, dimension, encryption, space_type):
    '''Delete all test indexes and verify deletion'''
    try:
        self.vx.delete_index(index_attr)
        logger.info(f"Deleted index: {index_attr}")
        with pytest.raises(Exception) as exc_info:
            self.vx.get_index(index_attr)
        assert "not found" in str(exc_info.value).lower()
    except Exception as e:
        if "not found" in str(e).lower():
            logger.warning(f"Index already missing: {index_attr}")
        else:
            raise
```

## Optional Test Cases

### 7. Test Fetch Existing Indexes

**Description**: Verifies that all the predefined test indexes are accessible and exist in the system. It attempts to fetch each index by name, logging warnings if any are missing.

```python
def test_fetch_existing_indexes(self):
    for idx in self.test_indexes:
        try:
            index = self.get_test_index(idx["name"], idx["encrypted"])
            assert index is not None, f"Index not found: {idx['name']}"
        except Exception:
            logger.warning(f"Index missing or failed to fetch: {idx['name']}")
```

### 8. Test Describe Index Vectors

**Description**: Fetches metadata for each index and logs the number of vectors it contains. It confirms the ability to successfully describe indexes and retrieve vector counts.

```python
def test_describe_index_vectors(self):
    for idx in self.test_indexes:
        index = self.get_test_index(idx["name"], idx["encrypted"])
        count = index.describe().get("count", 0)
        logger.info(f"Index: {idx['name']} | Vector Count: {count}")
```

### 9. Test Fetch Nonexistent Index

**Description**: Confirms that fetching an index which does not exist raises an appropriate exception.

```python
def test_fetch_nonexistent_index(self):
    """Ensure fetching nonexistent index raises error."""
    with pytest.raises(Exception) as exc_info:
        self.vx.get_index("nonexistent_index_123456")
    assert "not found" in str(exc_info.value).lower(), "Expected 'not found' error"
```

### 10. Test Describe Non-Existent Index

**Description**: Ensures that describing a non-existent index raises an error, as the index cannot be found.

```python
def test_describe_nonexistent_index(self):
    """Describing nonexistent index raises error."""
    with pytest.raises(Exception) as exc_info:
        idx = self.vx.get_index("nonexistent_index_123456")
        idx.describe()
    assert "not found" in str(exc_info.value).lower(), "Expected 'not found' error on describe"
```

### 11. Test Fetch Index with Wrong Encryption Key

**Description**: Tests that attempt to fetch an encrypted index using an invalid encryption key fails, raising an exception.

```python
def test_fetch_index_with_wrong_encryption_key(self):
    for idx in self.test_indexes:
        if idx["encrypted"]:
            with pytest.raises(Exception):
                self.vx.get_index(name=idx["name"], key="wrong_key_123")
```

### 12. Test Describe Index with Wrong Encryption Key

**Description**: Confirms that describing an encrypted index with a wrong encryption key raises an exception, denying access.

```python
def test_describe_index_with_wrong_encryption_key(self):
    for idx in self.test_indexes:
        if idx["encrypted"]:
            with pytest.raises(Exception):
                index = self.vx.get_index(name=idx["name"], key="wrong_key_123")
                index.describe()
```

## Test Configuration

The test suite uses four main test indexes with various configurations:
- Different dimensions (5 and 768)
- Different encryption states (encrypted and unencrypted)
- Different space types (cosine, ip, l2)

## Running the Tests

To run the deletion test suite:

```bash
pytest test_delete_vectors_and_indexes.py -v
```

For pipeline mode testing, ensure the configuration file is properly set up before running the tests.

## Key Features Tested

- Single vector deletion
- Multiple vector deletion
- Deletion with filters
- Index deletion
- Error handling for non-existent vectors/indexes
- Encryption key validation
- Pipeline mode compatibility