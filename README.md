# Legal RAG using LangChain

A Retrieval-Augmented Generation (RAG) system for answering questions from legal PDF documents using LangChain, FAISS, Hugging Face embeddings, Cross-Encoder reranking, and Llama 3.2 through Ollama.

## Project Overview

This project implements a Legal RAG pipeline that retrieves relevant information from legal PDF documents and generates answers grounded in the retrieved context.

The system combines:

* Document loading and PDF processing
* Recursive text chunking
* Hugging Face embeddings
* FAISS vector search
* Top-5 document retrieval
* Cross-Encoder reranking
* Top-3 final document selection
* Prompt-based answer generation
* Llama 3.2 through ChatOllama
* Custom retrieval and answer-quality evaluation

The project also includes an evaluation layer to measure retrieval performance and the quality of generated answers.

---

## Architecture

### RAG Pipeline

```text
Legal PDF Documents
        ↓
DirectoryLoader + PyPDFLoader
        ↓
16 PDF Pages Loaded
        ↓
RecursiveCharacterTextSplitter
(chunk_size=800, overlap=100)
        ↓
53 Text Chunks
        ↓
Hugging Face Embeddings
(all-MiniLM-L6-v2)
        ↓
FAISS Vector Store
        ↓
Retriever
(Top 5 Documents)
        ↓
Cross-Encoder Reranker
(ms-marco-MiniLM-L-6-v2)
        ↓
Top 3 Final Documents
        ↓
Create Context
        ↓
ChatPromptTemplate
        ↓
Llama 3.2 through ChatOllama
        ↓
Final Legal Answer
```

### Evaluation Pipeline

```text
Test Questions
      ↓
Run RAG
      ↓
Retrieved Sources
      ↓
┌──────────────────────┐
│ Retrieval Evaluation │
│                      │
│ • Hit Rate           │
│ • Precision@3        │
│ • Recall@3           │
└──────────────────────┘
      ↓
┌──────────────────────┐
│ Answer Evaluation    │
│                      │
│ • Faithfulness       │
│ • Answer Relevance   │
└──────────────────────┘
      ↓
Results DataFrame
```

---

## RAG Workflow

### 1. Document Loading

Legal PDF documents are loaded recursively from the local `Legal` directory using:

* `DirectoryLoader`
* `PyPDFLoader`

The current dataset contains 16 PDF pages.

Only PDF files are loaded using:

```python
glob="**/*.pdf"
```

### 2. Text Chunking

The loaded documents are divided into smaller chunks using `RecursiveCharacterTextSplitter`.

Configuration:

```text
Chunk size: 800
Chunk overlap: 100
```

This produces 53 text chunks for the current document collection.

### 3. Embeddings

Each text chunk is converted into a numerical vector using the Hugging Face embedding model:

```text
all-MiniLM-L6-v2
```

These embeddings allow the system to perform semantic similarity search.

### 4. FAISS Vector Store

The generated embeddings are stored in a FAISS vector store.

FAISS is used for efficient similarity-based retrieval of relevant document chunks.

### 5. Initial Retrieval

For each user question, the retriever initially retrieves the top 5 relevant documents.

```text
initial_k = 5
```

### 6. Cross-Encoder Reranking

The retrieved documents are then reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The Cross-Encoder assigns relevance scores to the retrieved question-document pairs.

The top 3 documents after reranking are selected for the final context.

```text
initial retrieval → 5 documents
                    ↓
             Cross-Encoder
                    ↓
             final → 3 documents
```

### 7. Context Creation

The final reranked documents are combined to create the context provided to the language model.

### 8. Prompt Construction

A `ChatPromptTemplate` is used to construct the prompt containing the user's question and retrieved context.

### 9. Answer Generation

The final answer is generated using:

```text
Llama 3.2
     ↓
ChatOllama
```

The answer is generated using the retrieved legal context.

---

## Evaluation

The project includes a custom evaluation layer for measuring both retrieval performance and generated-answer quality.

### Retrieval Metrics

#### Hit Rate

Checks whether the expected source document appears among the retrieved documents.

```text
Hit Rate = 1 if expected source is retrieved
           0 otherwise
```

#### Precision@3

Measures how many of the retrieved top-3 results are relevant to the expected source.

```text
Precision@3 =
Relevant Retrieved Results / Total Retrieved Results
```

#### Recall@3

Checks whether the expected source is successfully retrieved within the top-3 results.

```text
Recall@3 = 1 if expected source is retrieved
           0 otherwise
```

### Answer Quality Metrics

#### Faithfulness

The project uses a custom `calculate_faithfulness()` function to evaluate whether the generated answer is supported by the retrieved context.

#### Answer Relevance

The project uses a custom `calculate_answer_relevance()` function to evaluate whether the generated answer is relevant to the user's question.

### Evaluation Results

For every test question, the evaluation pipeline stores:

* Question number
* Question
* Expected source
* Retrieved sources
* Reranking scores
* Generated answer
* Hit Rate
* Precision@3
* Recall@3
* Faithfulness
* Answer Relevance

The results are collected into a results structure that can be converted into a DataFrame for analysis.

---

## Technology Stack

### Programming Language

* Python

### NLP / RAG

* LangChain
* Hugging Face Embeddings
* Sentence Transformers
* FAISS
* Cross-Encoder Reranking

### Large Language Model

* Llama 3.2
* Ollama
* ChatOllama

### Document Processing

* PyPDFLoader
* DirectoryLoader
* RecursiveCharacterTextSplitter

### Data Processing

* Pandas

---

## Project Structure

```text
Legal-RAG-LangChain/
│
├── Legal_RAG_LangChain.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── Legal/
    └── Institute-provided legal PDF documents
```

The legal documents are not included in the public repository because they were provided by the institute.

---

## Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

The project uses Ollama for running the local Llama 3.2 model.

Make sure Ollama is installed and the required model is available locally before running the notebook.

---

## How to Run

1. Clone the repository.
2. Install the dependencies from `requirements.txt`.
3. Install and configure Ollama.
4. Download the required Llama 3.2 model.
5. Place your own legal PDF documents inside the `Legal` directory.
6. Open `Legal_RAG_LangChain.ipynb`.
7. Run the notebook cells in order.
8. Run the evaluation section using the provided test questions.
9. Review the retrieval and answer-quality metrics.

---

## Key Features

* PDF document ingestion
* Recursive text chunking
* Semantic embeddings
* FAISS vector search
* Top-5 retrieval
* Cross-Encoder reranking
* Top-3 context selection
* Local LLM inference using Llama 3.2
* LangChain-based RAG pipeline
* Custom retrieval evaluation
* Custom answer-quality evaluation
* Reranking score tracking

---

## Future Improvements

* Add support for additional document formats such as DOCX
* Improve evaluation using larger test-question sets
* Add citation-aware answer generation
* Experiment with different embedding models
* Compare different reranking models
* Add a web-based user interface
* Add automated evaluation reporting
* Experiment with larger and more capable local language models

---

## Disclaimer

This project is developed for educational and research purposes.

The system is not intended to provide professional legal advice. Generated answers should be verified against the original legal documents and, where appropriate, with a qualified legal professional.
