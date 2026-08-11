# Zepto Support Assistant (`/support_assistant`)

An offline-first, Retrieval-Augmented Generation (RAG) customer support assistant designed to answer questions using Zepto's policy corpus.

The application uses **ChromaDB** for local vector storage, **Sentence Transformers** for embeddings, **LangGraph** for workflow orchestration, **Pydantic** for structured response validation, and **FastAPI** for serving the API.

The system is designed with an offline-first graded baseline using `MOCK_LLM=1`, allowing the application to run without external API keys or network-dependent LLM calls.

---

# 1. Project Structure

```text
/support_assistant
│
├── main.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
└── db/
    ├── chroma.sqlite3
    └── [ChromaDB index files]
```

### Main Components

| Component          | Responsibility                       |
| ------------------ | ------------------------------------ |
| `main.py`          | FastAPI application and RAG workflow |
| `docs/`            | Zepto policy documents               |
| `db/`              | Persistent ChromaDB vector database  |
| `requirements.txt` | Python dependencies                  |
| `Dockerfile`       | Container configuration              |
| `README.md`        | Project documentation                |

---

# 2. Architecture

The RAG pipeline consists of four major stages:

```text
                    +------------------------+
                    |     POST /ask          |
                    +-----------+------------+
                                |
                                v
                    +------------------------+
                    |    classify_intent     |
                    |    Keyword Router       |
                    +-----------+------------+
                                |
                  +-------------+-------------+
                  |                           |
          policy_question              general_question
                  |                           |
                  v                           v
       +---------------------+       +-------------------+
       | retrieve_and_answer |       |   direct_answer   |
       |     ChromaDB        |       |   Canned Response |
       +----------+----------+       +---------+---------+
                  |                            |
                  +-------------+--------------+
                                |
                                v
                    +------------------------+
                    |   Pydantic Validation  |
                    | answer / sources /     |
                    | confidence             |
                    +------------------------+
                                |
                                v
                         JSON Response
```

---

# 3. RAG Workflow

## Stage 1 — Document Ingestion and Embedding

Document ingestion is handled by:

```text
init_db()
```

The application reads eight policy documents:

```text
docs/doc_01.txt
docs/doc_02.txt
docs/doc_03.txt
docs/doc_04.txt
docs/doc_05.txt
docs/doc_06.txt
docs/doc_07.txt
docs/doc_08.txt
```

The documents are converted into vector embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model runs locally on CPU.

The resulting vectors and document metadata are persisted in a local ChromaDB collection:

```text
zepto_policies
```

The ChromaDB data is stored under:

```text
./db/
```

Typical generated files include:

```text
db/
├── chroma.sqlite3
└── HNSW/vector index files
```

---

# 4. Intent Classification

Intent classification is handled by the LangGraph node:

```text
classify_intent
```

The application examines the user's query for policy-related keywords.

Examples include:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

The query is classified into one of two intents:

```text
policy_question
general_question
```

---

## Policy Question

Example:

```text
What is the return policy for damaged items?
```

The query is routed to:

```text
retrieve_and_answer
```

The system retrieves relevant policy information from ChromaDB.

---

## General Question

Example:

```text
What is the capital of France?
```

The query is routed to:

```text
direct_answer
```

The baseline system returns a fixed response indicating that it currently answers Zepto policy questions only.

---

# 5. Retrieval

The retrieval stage is implemented by:

```text
retrieve_and_answer
```

For policy-related questions, the application queries the ChromaDB collection using vector similarity.

The top **3 relevant document chunks** are retrieved.

The retrieval process is local and does not require an external LLM.

The retrieved information is then used to construct the policy response.

---

# 6. Response Generation

The application supports two modes.

```text
MOCK_LLM=1
MOCK_LLM=0
```

---

# 7. MOCK_LLM Mode

## Default Behavior

If `MOCK_LLM` is unset or configured as:

```bash
MOCK_LLM=1
```

the system runs in deterministic offline mode.

This is the **graded baseline mode**.

It requires:

* No API key
* No external LLM account
* No network connection for generation
* No external LLM service

---

## Intent Classification

A deterministic keyword-based routing mechanism is used.

For example:

```text
return
refund
delivery
tracking
cancel
```

can route the query to the policy retrieval path.

---

## Policy Response

The system retrieves the most relevant context and generates a deterministic response similar to:

```text
Based on the retrieved context: <retrieved snippet>
```

---

## General Response

For questions outside the supported Zepto policy domain, the baseline response is:

```text
I can only answer questions about Zepto policies right now.
```

---

## Pydantic Validation

The final response is validated using a Pydantic model containing:

```text
answer
sources
confidence
```

This guarantees a predictable JSON response structure.

---

# 8. Optional Real-LLM Mode

The application also supports optional LLM generation.

Configure:

```bash
MOCK_LLM=0
```

The real-LLM mode requires:

```text
GROQ_API_KEY
```

The configured model is:

```text
llama-3.3-70b-versatile
```

The LLM receives:

* System role
* Retrieved context
* User question
* Task instructions
* Output-format requirements
* Few-shot examples

The generated response is validated against the Pydantic schema.

If schema validation fails, the system can retry generation with corrective feedback for up to two retry attempts.

---

# 9. Response Schema

The API returns a structured JSON response.

Example:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_05"
  ],
  "confidence": 1.0
}
```

### Fields

| Field        | Type   | Description                             |
| ------------ | ------ | --------------------------------------- |
| `answer`     | string | Final support response                  |
| `sources`    | list   | Retrieved policy document identifiers   |
| `confidence` | float  | Confidence associated with the response |

Pydantic is used to validate this structure before the response is returned by the API.

---

# 10. Local Installation

Navigate to the support assistant directory:

```bash
cd support_assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 11. Running the FastAPI Application

The application can be started using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

The API will be available at:

```text
http://127.0.0.1:7860
```

FastAPI's interactive documentation is available at:

```text
http://127.0.0.1:7860/docs
```

---

# 12. Running in Offline Graded Mode

The recommended configuration is:

```bash
set MOCK_LLM=1
```

On Linux/macOS:

```bash
export MOCK_LLM=1
```

Then start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

This mode does not require `GROQ_API_KEY`.

---

# 13. Docker Execution

## Build the Image

From the `/support_assistant` directory:

```bash
docker build -t zepto-support-assistant .
```

---

## Run the Container

```bash
docker run -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

The API can then be accessed at:

```text
http://127.0.0.1:7860
```

---

# 14. API Endpoint

The primary endpoint is:

```text
POST /ask
```

Request format:

```json
{
  "query": "What is the return policy for damaged items?"
}
```

---

# 15. Test Case 1 — Policy Retrieval

### Request

```bash
curl -X POST "http://127.0.0.1:7860/ask" \
-H "Content-Type: application/json" \
-d '{"query": "What is the return policy for damaged items?"}'
```

### Example Response

```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_05"
  ],
  "confidence": 1.0
}
```

### Expected Behavior

The question contains policy-related terms such as:

```text
return
damaged
```

Therefore, the query should be classified as:

```text
policy_question
```

The system retrieves relevant policy documents from ChromaDB and returns the structured response.

---

# 16. Test Case 2 — General Question

### Request

```bash
curl -X POST "http://127.0.0.1:7860/ask" \
-H "Content-Type: application/json" \
-d '{"query": "What is the capital of France?"}'
```

### Expected Response

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

### Expected Behavior

The question does not contain supported Zepto policy keywords.

Therefore, it is classified as:

```text
general_question
```

The request is routed to:

```text
direct_answer
```

No policy retrieval is required for the baseline response.

---

# 17. Example Policy Questions

The following are examples of queries that should trigger the policy retrieval path:

```text
What is the delivery policy?

How can I request a refund?

Can I cancel my order?

What is the return policy?

How can I track my order?

What are the membership benefits?

What are the support hours?

Can I get a refund for damaged products?
```

---

# 18. Example General Questions

Examples of unsupported general questions include:

```text
What is the capital of France?

Who is the Prime Minister of India?

What is Python?

Explain machine learning.
```

In `MOCK_LLM=1` mode, these questions receive the deterministic general-question response.

---

# 19. Technology Stack

| Technology            | Purpose                            |
| --------------------- | ---------------------------------- |
| Python                | Application development            |
| FastAPI               | REST API framework                 |
| Uvicorn               | ASGI server                        |
| LangGraph             | Workflow orchestration             |
| ChromaDB              | Vector database                    |
| Sentence Transformers | Local embeddings                   |
| Pydantic              | Structured output validation       |
| Requests/HTTP client  | API communication where applicable |
| Docker                | Containerization                   |
| Groq                  | Optional external LLM provider     |

---

# 20. Why RAG?

A Retrieval-Augmented Generation architecture is used so that answers can be grounded in the supplied Zepto policy documents.

Instead of relying only on information learned during model training, the system:

```text
User Query
    ↓
Intent Classification
    ↓
Vector Retrieval
    ↓
Relevant Policy Context
    ↓
Answer Generation
    ↓
Pydantic Validation
    ↓
JSON Response
```

This makes the assistant better suited to policy-based customer support where responses should be based on a controlled document corpus.

---

# 21. Offline-First Design

The baseline system is intentionally designed to work without external LLM dependencies.

The offline workflow is:

```text
Policy Documents
       ↓
Local Embedding Model
       ↓
Local ChromaDB
       ↓
Keyword Intent Router
       ↓
Retrieved Context
       ↓
Deterministic Response
       ↓
Pydantic Validation
       ↓
FastAPI
```

This provides reproducible behavior during evaluation.

---

# 22. Data Persistence

ChromaDB is persisted locally under:

```text
./db/
```

The collection name is:

```text
zepto_policies
```

The database stores:

* Document embeddings
* Document text
* Metadata
* Vector-search indexes

This avoids rebuilding the vector index on every query when the persisted database is already available.

---

# 23. Error Handling

The application uses structured validation and controlled processing to reduce runtime failures.

Important protections include:

* Pydantic response validation
* Controlled intent routing
* Local vector retrieval
* LLM retry handling in real-LLM mode
* Deterministic fallback behavior in mock mode

In real-LLM mode, schema validation errors trigger corrective retries, with a maximum of two retry attempts.

---

# 24. Security and Configuration

API keys should never be hard-coded into source files.

For real-LLM mode, configure:

```text
GROQ_API_KEY
```

through an environment variable.

Example on Windows:

```bash
set GROQ_API_KEY=your_api_key
set MOCK_LLM=0
```

Example on Linux/macOS:

```bash
export GROQ_API_KEY=your_api_key
export MOCK_LLM=0
```

Do not commit API keys to Git.

---

# 25. API Testing Checklist

| Test                  | Expected Result                 |
| --------------------- | ------------------------------- |
| Policy question       | Routed to retrieval             |
| Return question       | Policy retrieval                |
| Refund question       | Policy retrieval                |
| Delivery question     | Policy retrieval                |
| Tracking question     | Policy retrieval                |
| General question      | Direct response                 |
| Missing/invalid query | API validation error            |
| ChromaDB retrieval    | Top relevant documents returned |
| Pydantic validation   | Structured JSON                 |
| `MOCK_LLM=1`          | No external LLM required        |
| `MOCK_LLM=0`          | Groq API used                   |

---

# 26. End-to-End Flow

```text
                 USER
                   |
                   v
              POST /ask
                   |
                   v
          +----------------+
          | Validate Input |
          +-------+--------+
                  |
                  v
         +-------------------+
         | classify_intent   |
         +---------+---------+
                   |
          +--------+--------+
          |                 |
          v                 v
      POLICY             GENERAL
          |                 |
          v                 v
  +---------------+   +---------------+
  | ChromaDB      |   | direct_answer |
  | Vector Search |   +-------+-------+
  +-------+-------+           |
          |                   |
          v                   |
  +---------------+           |
  | Top 3 Context |           |
  +-------+-------+           |
          |                   |
          +---------+---------+
                    |
                    v
          +-------------------+
          | Response Creation |
          +---------+---------+
                    |
                    v
          +-------------------+
          | Pydantic Model    |
          +---------+---------+
                    |
                    v
             JSON Response
```

---

# 27. Advantages

### Offline Baseline

The application can run without an external LLM API.

### Local Retrieval

Policy documents and embeddings are stored locally.

### Structured Responses

Pydantic ensures a predictable response format.

### Modular Architecture

LangGraph separates classification, retrieval, and response-generation stages.

### API Ready

FastAPI provides a clean REST interface.

### Container Ready

Docker allows the application to be packaged and executed consistently.

### Optional LLM Enhancement

The system can switch from deterministic responses to LLM-generated responses through:

```text
MOCK_LLM=0
```

---

# 28. Limitations

The baseline intent classifier uses keyword matching rather than a semantic classifier.

Therefore, a policy question that does not contain one of the recognized keywords may be incorrectly classified as a general question.

The deterministic `MOCK_LLM=1` mode also does not perform true generative reasoning. Its primary purpose is reproducibility and offline evaluation.

The quality of retrieved answers depends on:

* Document quality
* Document chunking
* Embedding quality
* Query wording
* Vector similarity

---

# 29. Future Improvements

Potential improvements include:

1. Replace keyword intent classification with a trained or LLM-based classifier.
2. Add semantic query expansion.
3. Improve document chunking strategy.
4. Add metadata filtering to ChromaDB retrieval.
5. Implement conversation history.
6. Add confidence calibration.
7. Add citation-level source tracking.
8. Add automated API tests using `pytest`.
9. Add monitoring and logging.
10. Add authentication and rate limiting for production deployment.
11. Add a frontend customer-support interface.
12. Evaluate retrieval quality using Recall@K and MRR.

---

# 30. Final Project Summary

The Zepto Support Assistant demonstrates an end-to-end RAG application combining:

```text
FastAPI
   +
LangGraph
   +
ChromaDB
   +
Sentence Transformers
   +
Pydantic
   +
Optional Groq LLM
```

The system provides an **offline-first, deterministic baseline** while also supporting an optional real-LLM mode.

The complete workflow is:

```text
Policy Documents
      ↓
Local Embeddings
      ↓
ChromaDB
      ↓
User Query
      ↓
Intent Classification
      ↓
Relevant Context Retrieval
      ↓
Response Generation
      ↓
Pydantic Validation
      ↓
FastAPI JSON Response
```

This architecture demonstrates practical implementation of Retrieval-Augmented Generation, vector search, workflow orchestration, structured output validation, API development, and containerized deployment.

