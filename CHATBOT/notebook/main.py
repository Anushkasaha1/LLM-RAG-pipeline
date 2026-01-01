from langchain_core.documents import Document
import os

try:
    os.makedirs("../data/txt_files", exist_ok=True)
    print("✓ Directory created/verified")
except Exception as e:
    print(f"Error creating directory: {e}")

sample_txt = {
    "../data/txt_files/file1.txt": """AI OVERVIEW
Artificial Intelligence (AI) refers to systems designed to perform tasks that normally require human intelligence such as reasoning, learning, perception, decision-making, and language understanding.

TYPES OF ARTIFICIAL INTELLIGENCE
1. Narrow AI: Designed for a specific task such as chatbots or recommendation systems.
2. General AI: Capable of performing any intellectual task a human can do (theoretical).
3. Super AI: Intelligence surpassing human capabilities (hypothetical).

MACHINE LEARNING
Machine Learning (ML) is a subset of AI where models learn patterns from data instead of being explicitly programmed.

TYPES OF MACHINE LEARNING
1. Supervised Learning – Uses labeled data for training (classification, regression).
2. Unsupervised Learning – Discovers patterns in unlabeled data (clustering, dimensionality reduction).
3. Reinforcement Learning – Learns actions through reward and punishment.

DEEP LEARNING
Deep Learning uses multi-layer neural networks to learn hierarchical representations.

COMMON DEEP LEARNING MODELS
- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- Transformers

TRANSFORMER ARCHITECTURE
Transformers use attention mechanisms to process input tokens in parallel.
They are the foundation of modern language models.

LARGE LANGUAGE MODELS (LLMs)
LLMs are trained on massive text corpora to predict the next token.
They can generate text, answer questions, summarize documents, and write code.

EXAMPLES OF LLMS
- GPT
- Claude
- Gemini
- LLaMA
- Mistral

EMBEDDINGS
Embeddings are numerical vector representations of text or data.
Similar meanings produce similar vectors.

USES OF EMBEDDINGS
- Semantic search
- Document similarity
- Clustering
- Recommendation systems
- Retrieval-Augmented Generation

VECTOR DATABASES
Vector databases store embeddings and support similarity search.

POPULAR VECTOR DATABASES
- FAISS
- Chroma
- Pinecone
- Weaviate
- Milvus

SIMILARITY METRICS
- Cosine similarity
- Euclidean distance
- Dot product

RETRIEVAL-AUGMENTED GENERATION (RAG)
RAG combines retrieval with text generation to reduce hallucinations.

RAG WORKFLOW
1. Collect documents
2. Split text into chunks
3. Generate embeddings
4. Store embeddings in a vector database
5. Embed user query
6. Retrieve relevant chunks
7. Pass retrieved context to LLM
8. Generate grounded answer

CHUNKING STRATEGY
Chunking divides long text into smaller parts.
Chunk overlap helps preserve context.

METADATA IN RAG
Metadata helps track document source and context.
Examples include filename, chunk index, topic, and author.

LIMITATIONS OF RAG
- Dependent on data quality
- Sensitive to chunk size
- Retrieval errors affect output
- Context window limits
- Multi-hop reasoning challenges

ADVANCED RAG APPROACHES
- Agentic RAG
- Graph-based RAG
- Hybrid keyword + vector search
- Memory-augmented systems
- Tool-using agents

EVALUATION OF RAG SYSTEMS
- Precision
- Recall
- Faithfulness
- Answer relevance
- Latency
- Human evaluation

APPLICATIONS OF RAG
- Enterprise knowledge base
- Chat with documents
- Legal and medical QA
- Educational tutors
- Research assistants
- Customer support automation

FUTURE OF AI SYSTEMS
AI systems are moving toward multi-agent reasoning, tool usage, and deeper integration with external knowledge sources.
""",
    "../data/txt_files/file2.txt": """SOFTWARE ENGINEERING OVERVIEW
Software engineering focuses on designing, developing, testing, deploying, and maintaining software systems in a structured and reliable manner.

CORE SOFTWARE ENGINEERING PRINCIPLES
- Modularity
- Reusability
- Scalability
- Maintainability
- Reliability
- Performance optimization

SOFTWARE DEVELOPMENT LIFE CYCLE (SDLC)
1. Requirement analysis
2. System design
3. Implementation
4. Testing
5. Deployment
6. Maintenance

BACKEND DEVELOPMENT
Backend development handles server-side logic, APIs, databases, and authentication.

POPULAR BACKEND TECHNOLOGIES
- Node.js
- Python (Django, Flask, FastAPI)
- Java (Spring Boot)
- Go

APPLICATION PROGRAMMING INTERFACES (APIs)
APIs allow communication between different systems.

COMMON HTTP METHODS
- GET: Retrieve data
- POST: Create data
- PUT: Update data
- DELETE: Remove data

REST API PRINCIPLES
- Stateless communication
- Resource-based URLs
- Client-server architecture

DATABASE SYSTEMS
Databases store and manage structured or unstructured data.

TYPES OF DATABASES
Relational Databases (SQL):
- MySQL
- PostgreSQL
- SQLite

NoSQL Databases:
- MongoDB
- Redis
- Cassandra

SQL VS NOSQL
SQL: databases use fixed schemas and strong consistency.
NoSQL: databases support flexible schemas and horizontal scalability.

DATA ENGINEERING
Data engineering focuses on collecting, transforming, validating, and storing data.

DATA PIPELINE STAGES
1. Data ingestion
2. Data cleaning
3. Data transformation
4. Data storage
5. Analytics or ML usage

ETL AND ELT
ETL stands for Extract, Transform, Load.
ELT loads raw data before transformation.

BIG DATA CONCEPTS
Big data is defined by:
- Volume
- Velocity
- Variety
- Veracity
- Value

BIG DATA TOOLS
- Hadoop
- Spark
- Kafka

DATA CLEANING TASKS
- Removing duplicates
- Handling missing values
- Normalizing formats
- Fixing inconsistencies

DATA STORAGE FORMATS
- CSV
- JSON
- Parquet
- Avro
- TXT

INDEXING
Indexes improve search and lookup performance.

TYPES OF INDEXES
- B-tree index
- Hash index
- Inverted index
- Vector index

INFORMATION RETRIEVAL (IR)
Information retrieval focuses on finding relevant documents based on a query.

IR COMPONENTS
- Query processing
- Ranking
- Similarity scoring
- Relevance estimation

SEMANTIC SEARCH
Semantic search understands meaning instead of exact keywords.
It relies on embeddings to find similar content.

EVALUATION METRICS
- Precision
- Recall
- F1-score
- Mean Reciprocal Rank (MRR)
- nDCG

SECURITY AND PRIVACY
Security ensures protection of data and systems.

SECURITY PRACTICES
- Encryption
- Authentication
- Authorization
- Access control
- Secure storage

LOGGING AND MONITORING
Used to track application behavior and detect issues.

DEPLOYMENT PIPELINES
Common environments:
- Development
- Staging
- Production

DEPLOYMENT TOOLS
- Docker
- CI/CD pipelines

REAL-WORLD APPLICATIONS
- Search engines
- Chatbots
- Recommendation systems
- Enterprise knowledge bases
- Document question-answering systems
- AI assistants

SUMMARY
Modern intelligent systems combine software engineering, data management, retrieval techniques, and language models to deliver accurate, scalable, and reliable applications.
"""
}

try:
    for file_path, txt in sample_txt.items():
        with open(file_path, "w") as f:
            f.write(txt)
    print("✓ Files written successfully")
except Exception as e:
    print(f"Error writing files: {e}")

# txt loaders
from langchain_community.document_loaders import DirectoryLoader, TextLoader

try:
    dir_loader = DirectoryLoader("../data/txt_files", glob="**/*.txt", loader_cls=TextLoader)
    documents = dir_loader.load()
    print(f"✓ Loaded {len(documents)} documents")
except Exception as e:
    print(f"Error loading documents: {e}")
    documents = []

# text splitting(chunking)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional


def split_documents(documents,
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""],
                    verbose: bool = True,
                    add_chunk_index: bool = True
                    ):
    """
    split documents into smaller chunks
     Parameters:
        documents: List of documents to split
        chunk_size: Maximum characters per chunk (default: 2000)
        chunk_overlap: Characters to overlap between chunks (default: 200)
        separators: Custom split hierarchy (default: ["\n\n", "\n", " ", ""])
        add_chunk_index: Add chunk number to metadata (default: True)
        verbose: Print detailed splitting information (default: True)
    
    Returns:
        List of split document chunks

    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            keep_separator=True
        )
        split_doc = text_splitter.split_documents(documents)

        if add_chunk_index:
            for idx, doc in enumerate(split_doc):
                doc.metadata['chunk_index'] = idx
                doc.metadata['total_chunks'] = len(split_doc)
        
        if verbose:
            print(f"✓ Split into {len(split_doc)} chunks")
        
        return split_doc
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []


def adv_split(
        documents,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        filter_emplty: bool = True
):
    """
    advanced splitting with filtering and validation
    """
    try:
        split_doc = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            verbose=False)
        if filter_emplty:
            original_len = len(split_doc)
            split_doc = [
                doc for doc in split_doc if len(doc.page_content.strip()) >= min_chunk_size
            ]
            removed = original_len - len(split_doc)
            print(f"Removed {removed} empty/short chunks")
        return split_doc
    except Exception as e:
        print(f"Error in advanced split: {e}")
        return []



# embeddings
import numpy as np
from sentence_transformers import SentenceTransformer  # my embedding model will be inside this
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import os

try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Global model loaded")
except Exception as e:
    print(f"Error loading global model: {e}")
    model = None


class EmbeddingManager:
    """handles document embedding generation using SentenceTransformer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"the model started loaded!")
            self.model = SentenceTransformer(self.model_name)
            print(f"model loaded successfully.{self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_embedding(self, texts: list[str]) -> np.ndarray:
        """generate embeddings for list of texts
        Args:
            texts:list of text strings to embed
        Returns:
            numpy array of embeddings with shape (len(texts),embedding_dim)    
        """
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            print(f"generating embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise



# vectorstore

class Vectorestore():
    """manages document embeddings in a chromadb vector store"""

    def __init__(self, collection_name: str = "documents", persist_directory: str = "../data/vectore"):
        """initialize the vector store 

        Args:
            collection_name:Name of the ChromaDB collection
            persist_directory:Directory to persist the vector store"""
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            # FIXED: Removed quotes from self.persist_directory
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name,
                                                                   metadata={"description": "document embedding for RAG appliaction"})
            print(f"vector store initialized succesfully.")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_document(self, documents: list[Any], embeddings: np.ndarray):
        """add documents and their embeddings to the vector store
        
        Args:
        document:list of langchain documents
        embeddings:corresponding document embeddings
        """
        try:
            if len(documents) != len(embeddings):
                print("error")
                raise ValueError(f"Documents count ({len(documents)}) != embeddings count ({len(embeddings)})")
            
            ids = []
            metadatas = []
            document_text = []
            embedding_list = []

            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(doc_id)

                metadata = dict(doc.metadata)
                metadata['doc_index'] = i
                metadata['content_length'] = len(doc.page_content)
                metadatas.append(metadata)

                document_text.append(doc.page_content)
                embedding_list.append(embedding.tolist())

            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=document_text,
                embeddings=embedding_list
            )
            print(f"✓ Added {len(documents)} documents to vector store")
        except Exception as e:
            print(f"Failed to add documents to vector store: {e}")
            raise

    def search(self, query_embedding, top_k=3):
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            print(f"✓ Search completed, found {len(results['documents'][0])} results")
            return results
        except Exception as e:
            print(f"Failed to search in vector store: {e}")
            raise




def build_prompt(context:str,question:str)->str:
    return f"""
     Build a prompt for the LLM with context and question ,you are an helpful assistant.
     use only context to ans the question,
     if the answer is not in the context then say i dont know
    
    context:{context}
    question:{question}
    """
from dotenv import load_dotenv

load_dotenv() 
from langchain_groq import ChatGroq
os.environ["GROQ_API_KEY"] = os.getenv("groq_api_key")
llm=ChatGroq(model="llama-3.3-70b-versatile")

chunks=adv_split(documents,
                 chunk_size=2000,
                 chunk_overlap=200,
                 min_chunk_size=50,
                 filter_emplty=True)
embedding_manager = EmbeddingManager()
text=[doc.page_content for doc in chunks]


embeddings=embedding_manager.generate_embedding(text)
vector_store = Vectorestore()
vector_store.add_document(chunks,embeddings)

query = input("tell me how can i help you:")
query_embedding = embedding_manager.generate_embedding([query])[0]

results = vector_store.search(query_embedding)

def generate_answer(question: str, results):
    context = "\n\n".join(results["documents"][0])
    prompt = build_prompt(context, question)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)
ans=generate_answer(query,results)    
print("\n" + "="*50)
print(f"Question: {query}")
print("="*50)
print(f"Answer: {ans}")
print("="*50)
        