import pytest
import logging
import sys
import os
from dotenv import load_dotenv
from vecx.vectorx import VectorX
from testsuite.vecx_crewai.base1 import VectorXVectorStore
from testsuite.vecx_crewai.hugging_face import HuggingFaceEmbedder
from crewai import Crew, Agent, Task, Process, LLM
from crewai.memory import ShortTermMemory,EntityMemory
from crewai_tools import FileReadTool
import builtins


load_dotenv()
VECTORX_API_TOKEN = getattr(builtins, "VECTORX_API_KEY", None)
if VECTORX_API_TOKEN == None:
    VECTORX_API_TOKEN = os.getenv("VECTORX_API_TOKEN")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # or DEBUG

if not logger.hasHandlers():  # Prevent duplicate handlers
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TestCrewPipeline:
    @classmethod
    def setup_class(cls):
        cls.vx = VectorX(token=VECTORX_API_TOKEN)
        cls.encryption_key = cls.vx.generate_key()
        cls.embedder_config = {
            "provider": "custom",  # Tell CrewAI to use your embedder
            "config": {
                "embedder": HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
            }
        }
        cls.dimension = 384
        index_lst = cls.vx.list_indexes()
        if len(index_lst['indixes'])>0:
            for index in index_lst['indixes']:
                cls.vx.delete_index(index['name'])
        cls.cleanup_indexes = []

        cls.memory_storage = VectorXVectorStore(
            type="vectordb_knowledge_hybrid_index",
            api_token=VECTORX_API_TOKEN,
            embedder_config=cls.embedder_config,
            encryption_key=cls.encryption_key
        )

        cls.llm = LLM(
            model="gemini/gemini-1.5-flash",
            api_key=GOOGLE_API_KEY
        )

        cls.file_reader = FileReadTool()

        cls.memory = ShortTermMemory(storage=cls.memory_storage)
        cls.entity_memory=EntityMemory(storage=cls.memory_storage)

        # ----- AGENTS -----
        cls.vector_db_expert = Agent(
            role="Vector Database Expert",
            goal="Analyze and extract comprehensive information about vector databases and embeddings",
            backstory="""You are a specialized AI/ML engineer with deep expertise in vector databases, 
            embeddings, similarity search, and retrieval systems. You understand the technical concepts, 
            architectures, use cases, and implementation details of vector storage and retrieval systems.""",
            llm=cls.llm,
            tools=[cls.file_reader],
            verbose=True
        )


        cls.ml_concepts_organizer = Agent(
            role="ML Concepts Organizer",
            goal="Organize and categorize vector database and ML concepts systematically",
            backstory="""You specialize in organizing machine learning and database concepts into 
            coherent knowledge structures. You understand how different vector database technologies 
            relate to each other and can categorize concepts by complexity, use case, and technical domain.""",
            llm=cls.llm,
            verbose=True,
        )

        cls.technical_validator = Agent(
            role="Technical Accuracy Validator",
            goal="Verify and validate technical information about vector databases and ML concepts",
            backstory="""You are a technical validator who ensures the accuracy and consistency 
            of vector database and machine learning information. You cross-check technical specifications, 
            validate code examples, and verify architectural descriptions.""",
            llm=cls.llm,
            verbose=True
        )


        # ----- TASKS -----
        cls.research_task = Task(
            description="""
            Read and analyze the vector database concepts file (vector_db_concepts.txt).
            Extract detailed information about:
            - Vector database fundamentals and core concepts
            - Embedding models and vector representations
            - Similarity search algorithms (cosine, euclidean, dot product)
            - Popular vector database systems (Pinecone, Weaviate, Chroma, Qdrant, etc.)
            - Vector indexing techniques (HNSW, IVF, LSH, etc.)
            - Use cases and applications (RAG, semantic search, recommendation systems)
            - Performance considerations and optimization techniques
            - Integration patterns with LLMs and ML pipelines
            
            Provide comprehensive technical analysis of each concept.
            """,
            agent=cls.vector_db_expert,
            expected_output="Detailed technical analysis of vector database concepts and technologies"
        )

        cls.organization_task = Task(
            description="""
            Based on the research findings, organize the vector database information into 
            systematic categories:
            - Fundamental concepts (vectors, embeddings, similarity metrics)
            - Database systems (cloud vs self-hosted, open source vs commercial)
            - Algorithms and techniques (indexing, search, clustering)
            - Use cases and applications (by domain and complexity)
            - Performance and scalability considerations
            - Integration and deployment patterns
            
            Create knowledge hierarchies and store relationships between concepts 
            for efficient retrieval and learning.
            """,
            agent=cls.ml_concepts_organizer,
            expected_output="Systematically organized vector database knowledge with concept relationships",
            context=[cls.research_task]
        )

        cls.validation_task = Task(
            description="""
            Validate the accuracy of the stored vector database information:
            - Check technical specifications and parameters
            - Verify algorithm descriptions and mathematical formulations
            - Validate use case examples and implementation details
            - Cross-check performance claims and benchmarks
            - Identify any inconsistencies or outdated information
            
            Provide a comprehensive validation report with technical accuracy assessment.
            """,
            agent=cls.technical_validator,
            expected_output="Technical validation report with accuracy assessment and quality metrics",
            context=[cls.research_task, cls.organization_task]
        )

        # ----- SAMPLE VECTOR DATABASE CONCEPTS DATA -----
        cls.sample_vector_concepts_data = """
        # Vector Database Concepts and Knowledge

        ## What are Vector Databases?
        Vector databases are specialized storage systems designed to efficiently store, index, and query high-dimensional vectors (embeddings). Unlike traditional databases that store structured data in rows and columns, vector databases are optimized for similarity search operations on dense numerical vectors.

        **Key Characteristics:**
        - Store high-dimensional vectors (typically 100-4096 dimensions)
        - Support similarity search queries (find similar vectors)
        - Optimized for approximate nearest neighbor (ANN) search
        - Scale to billions of vectors
        - Enable semantic search and AI applications

        ## Embeddings and Vector Representations
        Embeddings are dense numerical representations of data (text, images, audio) in a high-dimensional space where semantically similar items are positioned close together.

        **Types of Embeddings:**
        - **Text Embeddings**: Word2Vec, GloVe, BERT, OpenAI, Sentence Transformers
        - **Image Embeddings**: ResNet, VGG, CLIP, Vision Transformers
        - **Multimodal Embeddings**: CLIP, DALL-E, Flamingo
        - **Code Embeddings**: CodeBERT, GraphCodeBERT

        **Embedding Dimensions:**
        - Small: 100-300 dimensions (Word2Vec, GloVe)
        - Medium: 512-768 dimensions (BERT, Sentence Transformers)
        - Large: 1024-4096 dimensions (OpenAI Ada-002, Large Language Models)

        ## Similarity Search Algorithms
        Vector databases use various distance metrics to measure similarity between vectors:

        **Distance Metrics:**
        - **Cosine Similarity**: Measures angle between vectors, range [-1, 1]
        - **Euclidean Distance**: L2 distance, measures straight-line distance
        - **Dot Product**: Measures vector magnitude and direction alignment
        - **Manhattan Distance**: L1 distance, sum of absolute differences
        - **Hamming Distance**: For binary vectors, counts differing bits

        **Search Algorithms:**
        - **Brute Force**: Exact search, O(n) complexity
        - **Approximate Nearest Neighbor (ANN)**: Trade accuracy for speed
        - **Hierarchical Navigable Small World (HNSW)**: Graph-based ANN
        - **Inverted File (IVF)**: Clustering-based search
        - **Locality Sensitive Hashing (LSH)**: Hash-based approximation

        ## Popular Vector Database Systems

        ### Cloud-Based Solutions:
        **VectorX:**
        - Security-first, enterprise-grade **vector database**
        - Performs ANN searches directly over **encrypted data** (queryable encryption)
        - **End-to-end encryption**: data is encrypted client-side, in-transit, in-memory, and at-rest (HIPAA/SOC2 compliant)
        - Fast similarity search on encrypted vectors: **~14 ms P99 latency** on 500K vectors, 563 QPS, 96.3% recall (4 CPU + 30GB RAM)
        - **Memory-efficient hybrid graph structure**: ~90% lower memory usage compared to other vector DBs
        - Supports **metadata filtering** with `$eq`, `$in`, and other flexible filter expressions during encrypted search
        - Multi-region, **fully managed cloud** or **on-prem deployment** options with audit logs and SLA support
        - Client SDKs available for **Python**, **JavaScript** (Go coming soon)

        **Pinecone:**
        - Fully managed vector database service
        - Supports metadata filtering and hybrid search
        - Auto-scaling and high availability
        - Pricing based on vector dimensions and queries

        **Weaviate:**
        - Open-source with cloud offering
        - Built-in vectorization modules
        - GraphQL API and REST endpoints
        - Supports multimodal search

        ### Self-Hosted Solutions:
        **Chroma:**
        - Open-source embedding database
        - Python-native with simple API
        - Built-in embedding functions
        - Excellent for development and prototyping

        **Qdrant:**
        - Rust-based vector search engine
        - High performance and low latency
        - Supports payload filtering
        - Docker-ready deployment

        **Milvus:**
        - Open-source vector database
        - Supports multiple index types
        - Kubernetes-native deployment
        - Enterprise features available

        **FAISS (Facebook AI Similarity Search):**
        - Library for efficient similarity search
        - Optimized for large-scale datasets
        - Multiple index algorithms
        - CPU and GPU support

        ## Vector Indexing Techniques

        ### HNSW (Hierarchical Navigable Small World):
        - Graph-based indexing structure
        - Multiple layers with skip connections
        - Excellent query performance
        - Trade-off between build time and search speed

        ### IVF (Inverted File):
        - Clustering-based approach
        - Partitions vectors into clusters
        - Faster than brute force for large datasets
        - Can be combined with quantization

        ### LSH (Locality Sensitive Hashing):
        - Hash similar vectors to same buckets
        - Probabilistic approach
        - Good for high-dimensional data
        - Multiple hash functions for better recall

        ### Quantization Techniques:
        - **Product Quantization (PQ)**: Reduces memory usage
        - **Scalar Quantization**: Reduces precision
        - **Binary Quantization**: Converts to binary vectors

        ## Use Cases and Applications

        ### Retrieval-Augmented Generation (RAG):
        - Store document embeddings in vector database
        - Retrieve relevant context for LLM queries
        - Combine parametric and non-parametric knowledge
        - Enable up-to-date information retrieval

        ### Semantic Search:
        - Search by meaning rather than exact keywords
        - Natural language queries
        - Cross-lingual search capabilities
        - Context-aware results

        ### Recommendation Systems:
        - User and item embeddings
        - Collaborative filtering with vectors
        - Content-based recommendations
        - Real-time personalization

        ### Image and Video Search:
        - Visual similarity search
        - Reverse image search
        - Video content analysis
        - Multimodal search (text-to-image)

        ### Anomaly Detection:
        - Identify outliers in high-dimensional space
        - Fraud detection
        - Network security monitoring
        - Quality control in manufacturing

        ## Performance Considerations

        ### Indexing Performance:
        - Build time vs query speed trade-offs
        - Memory usage for different index types
        - Batch vs real-time indexing
        - Index update strategies

        ### Query Performance:
        - Latency requirements (sub-millisecond to seconds)
        - Throughput (queries per second)
        - Accuracy vs speed trade-offs
        - Caching strategies

        ### Scalability Factors:
        - Number of vectors (millions to billions)
        - Vector dimensions (100 to 4096+)
        - Concurrent users and queries
        - Horizontal scaling capabilities

        ## Integration Patterns

        ### LLM Integration:
        - Embedding generation pipelines
        - Vector storage and retrieval
        - Prompt augmentation with retrieved context
        - Feedback loops for continuous improvement

        ### ML Pipeline Integration:
        - Feature stores with vector support
        - Model serving with embedding endpoints
        - Batch and streaming processing
        - MLOps and monitoring

        ### Application Architecture:
        - Microservices with vector databases
        - API design patterns
        - Authentication and authorization
        - Monitoring and observability

        ## Best Practices

        ### Data Preparation:
        - Normalize embeddings for cosine similarity
        - Handle missing or corrupted vectors
        - Metadata design for filtering
        - Version control for embeddings

        ### Performance Optimization:
        - Choose appropriate distance metrics
        - Tune index parameters
        - Implement proper caching
        - Monitor query patterns

        ### Security and Privacy:
        - Encrypt vectors at rest and in transit
        - Access control and authentication
        - Data retention policies
        - Compliance with regulations (GDPR, CCPA)

        ### Monitoring and Maintenance:
        - Track query performance metrics
        - Monitor index health and accuracy
        - Plan for capacity scaling
        - Regular backup and recovery procedures
        """

        # Create vector database concepts data file
        with open("vector_db_concepts.txt", "w", encoding="utf-8") as file:
            file.write(cls.sample_vector_concepts_data)
        print("✅ Created vector_db_concepts.txt with comprehensive vector database knowledge")

    @classmethod
    def teardown_class(cls):
        # Runs once after all tests in this class
        index_lst = cls.vx.list_indexes()
        if len(index_lst['indixes'])>0:
            for index in index_lst['indixes']:
                cls.vx.delete_index(index['name'])
        logger.info("Test crew pipeline hybrid index tests done")


    def test_vector_db_knowledge_ingestion(self):
        try:
            # ----- CREW -----
            crew = Crew(
                agents=[self.vector_db_expert, self.ml_concepts_organizer, self.technical_validator],
                tasks=[self.research_task, self.organization_task, self.validation_task],
                process=Process.sequential,
                memory=True,
                short_term_memory=self.memory,
                entity_memory=self.entity_memory,
                verbose=True
            )
            assert crew is not None

            # Test kickoff to store knowledge
            result = crew.kickoff()
            assert result is not None, "Crew failed to store knowledge"
        except Exception as e:
            pytest.fail(f"Initialization failed with error: {e}")


    @pytest.mark.parametrize("query", [
        "vector database fundamentals",
        "embedding models",
        "similarity search algorithms",
        "HNSW indexing",
        "Pinecone database",
        "cosine similarity",
        "RAG applications",
        "semantic search",
        "vector quantization",
        "performance optimization"
    ])
    def test_vector_db_query_retrieval(self, query):
        """
        Test retrieval of vector database concept vectors from VectorX for each query.
        """
        results = self.memory_storage.search(query=query, limit=3)
        assert results is not None, f"No results returned for query: '{query}'"
        assert len(results) > 0, f"No vectors found for query: '{query}'"
        assert len(results) <= 3, f"error, not more than 3 vectors retrieved for query: '{query}'"


    @pytest.mark.parametrize("question, expected_keywords", [
        ("What is the difference between cosine similarity and euclidean distance?",
        ["cosine", "euclidean", "distance"]),
        ("How does HNSW indexing work in vector databases?",
        ["hnsw", "index", "graph"]),
        ("What are the main use cases for vector databases?",
        ["use case", "vector database"]),
        ("Compare Pinecone and Weaviate vector databases",
        ["pinecone", "weaviate", "compare"]),
        ("How does RAG (Retrieval-Augmented Generation) use vector databases?",
        ["rag", "retrieval", "vector"]),
        ("What are the performance considerations for vector databases?",
        ["performance", "latency", "throughput"]),
        ("Explain different types of embeddings and their dimensions",
        ["embedding", "dimension", "vector"])
    ])
    def test_vector_db_knowledge_qna(self,question, expected_keywords):
        """
        Test if the stored knowledge can answer vector database related questions
        with relevant keywords present in the response.
        """
        # Create an agent for testing retrieval
        knowledge_tester = Agent(
            role="Vector DB Knowledge Tester",
            goal="Test retrieval of stored vector database information",
            backstory="You test the knowledge system by asking specific questions about vector databases and ML concepts.",
            llm=self.llm,
            verbose=False
        )

        # Create the task
        test_task = Task(
            description=f"Answer this question based on stored vector database knowledge: {question}",
            agent=knowledge_tester,
            expected_output="Specific technical answer based on stored vector database concepts"
        )

        # Create a temporary crew to run the task
        test_crew = Crew(
            agents=[knowledge_tester],
            tasks=[test_task],
            process=Process.sequential,
            memory=True,
            short_term_memory=self.memory,
            entity_memory=self.entity_memory,
            verbose=False
        )

        # Kickoff the crew and get the result
        answer = test_crew.kickoff()
        str_answer = answer.raw

        # Assertions
        assert answer is not None, f"No answer returned for question: {question}"
        assert isinstance(str_answer, str), "Answer should be a string"
        assert len(str_answer.strip()) > 20, "Answer is too short to be meaningful"

        # Check for expected keywords
        matched = all(keyword.lower() in str_answer.lower() for keyword in expected_keywords)
        assert matched, f"Answer missing expected keywords for '{question}': {str_answer}"



    @pytest.mark.parametrize(
        "query,expected_keywords",
        [
            ("What do you remember about HNSW algorithm?", ["HNSW", "node", "vector"]),
            ("Tell me about Pinecone vector database", ["Pinecone", "vector database"]),
            ("What are the characteristics of cosine similarity?", ["cosine", "similarity"]),
            ("How does semantic search work with embeddings?", ["semantic search", "embedding"]),
        ]
    )
    def test_entity_memory_qna(self,query, expected_keywords):
        """Test entity memory for vector database concepts and technologies"""
        tester = Agent(
            role="Entity Memory Tester",
            goal="Verify that entity memory recalls specific vector database concepts",
            backstory="You query whether key vector database technologies and concepts were learned earlier.",
            llm=self.llm,
            verbose=False
        )

        task = Task(
            description=f"Answer using entity memory: {query}",
            agent=tester,
            expected_output="Recall the technical details previously stored"
        )

        crew = Crew(
            agents=[tester],
            tasks=[task],
            process=Process.sequential,
            memory=True,
            short_term_memory=self.memory,
            entity_memory=self.entity_memory,
            verbose=False
        )

        answer = crew.kickoff()
        str_answer = answer.raw
        print(f"\n❓ {query}")
        print(f"🧠 Entity memory answer: {str_answer}")


        # Check if ALL expected keywords are present
        matched = all(keyword.lower() in str_answer.lower() for keyword in expected_keywords)
        assert matched==True , f"Answer missing expected keywords for '{query}': {str_answer}"


    @pytest.mark.parametrize(
        "query,expected_keywords",
        [
            ("Compare HNSW vs IVF indexing algorithms", ["HNSW", "IVF"]),
            ("What are the trade-offs between Pinecone and open-source solutions?", ["Pinecone", "open-source"]),
            ("Compare different distance metrics for similarity search", ["cosine", "euclidean", "manhattan"]),  # common metrics
            ("Analyze the pros and cons of different embedding models", ["pros", "cons"]),
            ("Compare cloud-based vs self-hosted vector databases", ["cloud", "self-hosted"]),
        ]
    )
    def test_technical_comparisons(self, query, expected_keywords):
        """Test system's ability to compare vector database technologies"""
        comparison_expert = Agent(
            role="Vector DB Comparison Expert",
            goal="Compare and contrast different vector database technologies and concepts",
            backstory="You analyze stored technical data to compare vector database systems and algorithms.",
            llm=self.llm,
            verbose=False
        )

        task = Task(
            description=f"Provide technical comparison: {query}",
            agent=comparison_expert,
            expected_output="Detailed technical comparison with pros, cons, and use cases"
        )

        crew = Crew(
            agents=[comparison_expert],
            tasks=[task],
            process=Process.sequential,
            memory=True,
            short_term_memory=self.memory,
            entity_memory=self.entity_memory,
            verbose=False
        )

        answer = crew.kickoff()
        str_answer = answer.raw
        print(f"\n⚖️ Query: {query}\n🧠 Answer: {str_answer}\n")

        # Check if all expected keywords are present in the answer
        matched = all(keyword.lower() in str_answer.lower() for keyword in expected_keywords)
        assert matched, f"Answer for '{query}' missing expected keywords: {expected_keywords}"


    @pytest.mark.parametrize(
        "question ,expected_keywords",
        [
            (
                "How should I choose the right vector database for my RAG application?",
                ["RAG", "vector database", "choose"]
            ),
            (
                "What are the best practices for optimizing vector database performance?",
                ["best practices", "optimize", "performance"]
            ),
            (
                "How do I properly normalize embeddings for cosine similarity?",
                ["normalize", "embeddings", "cosine similarity"]
            ),
            (
                "What indexing strategy should I use for 1 million vectors?",
                ["indexing", "strategy", "million vectors"]
            ),
            (
                "How do I implement monitoring for a vector database system?",
                ["monitoring", "vector database", "system"]
            ),
        ]
    )
    def test_practical_guidance(self, question, expected_keywords):
        """Test practical implementation guidance for vector databases"""
        guidance_expert = Agent(
            role="Vector DB Implementation Guide",
            goal="Provide practical guidance for implementing vector database solutions",
            backstory="You help developers and engineers implement vector database solutions by providing practical advice.",
            llm=self.llm,
            verbose=False
        )

        task = Task(
            description=f"Provide practical implementation guidance: {question}",
            agent=guidance_expert,
            expected_output="Actionable advice and best practices for implementation"
        )

        crew = Crew(
            agents=[guidance_expert],
            tasks=[task],
            process=Process.sequential,
            memory=True,
            short_term_memory=self.memory,
            entity_memory=self.entity_memory,
            verbose=False
        )

        answer = crew.kickoff()
        str_answer = answer.raw
        print(f"\n💡 Query: {question}\n🧠 Guidance Answer: {str_answer}\n")

        # Check if all expected keywords are present in the answer
        matched = all(keyword.lower() in str_answer.lower() for keyword in expected_keywords)
        assert matched, f"Answer for '{question}' is missing expected keywords: {expected_keywords}"


