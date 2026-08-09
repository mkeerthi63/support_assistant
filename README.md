# Zepto Support Assistant (`/support_assistant`)

An offline-first, RAG-driven customer support assistant service built for Zepto's policy corpus. The application embeds and indexes policy documents locally, orchestrates query classification and context retrieval using **LangGraph**, enforces structured output schema validation with **Pydantic**, and serves queries via a **FastAPI** application.

---

## Architecture Description

The RAG pipeline operates across four sequential stages:

                      +------------------------+
                      |   POST Request /ask    |
                      +-----------+------------+
                                  |
                                  v
                      +------------------------+
                      |    classify_intent     |
                      |   (Keyword Router)     |
                      +----+--------------+----+
                           |              |
           policy_question |              | general_question
                           v              v
             +-------------------+  +-------------------+
             |retrieve_and_answer|  |   direct_answer   |
             | (ChromaDB Search) |  |(Canned Text Resp.)|
             +---------+---------+  +---------+---------+
                       |                      |
                       +----------+-----------+
                                  |
                                  v
                      +------------------------+
                      |  Pydantic JSON Model   |
                      | (answer/sources/conf.) |
                      +------------------------+

### 1. Ingestion & Embedding Stage
- **Handled by**: `init_db()` in `main.py`.
- **Process**: Reads 8 plain-text policy files (`doc_01.txt` through `doc_08.txt`) from the `./docs` directory.
- **Embedding**: Generates vector embeddings locally on CPU using `sentence-transformers/all-MiniLM-L6-v2`.
- **Storage**: Persists the embeddings, document texts, and metadata into a local **ChromaDB** collection named `zepto_policies` stored inside the `./db` directory (`chroma.sqlite3` and binary HNSW index files).

### 2. Intent Classification Stage
- **Handled by**: `classify_intent` node and `route_intent` conditional edge in `main.py`.
- **Process**: Inspects the incoming query for key policy terms (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`).
- **Routing**: Routes the graph state to either `policy_question` or `general_question`.

### 3. Retrieval Stage
- **Handled by**: `retrieve_and_answer` node in `main.py`.
- **Process**: Queries the ChromaDB vector collection using cosine similarity to fetch the top 3 most relevant document chunks based on the user's query. This retrieval stage runs locally in both mock and real-LLM modes.

### 4. Generation & Output Formatting Stage
- **Handled by**: `retrieve_and_answer` and `direct_answer` graph nodes in `main.py`.
- **Process**: Wraps the response into the validated `SupportResponse` Pydantic model (`answer`, `sources`, `confidence`).

---

## MOCK_LLM Toggle Behavior

Every LLM dependency is controlled by the `MOCK_LLM` environment variable:

* **Default Mode (`MOCK_LLM=1` or unset) — Graded Baseline**:
  - Requires **no API keys, accounts, or network connections**.
  - **Intent Classification**: Evaluates a deterministic keyword matching heuristic.
  - **Policy Answer**: Formats a canned string using the top retrieved document snippet (`Based on the retrieved context: <snippet>`).
  - **General Answer**: Returns a fixed canned response (`I can only answer questions about Zepto policies right now.`).
  - **JSON Output**: Populates the `SupportResponse` Pydantic model directly and deterministically.

* **Optional Real-LLM Mode (`MOCK_LLM=0`)**:
  - Requires `GROQ_API_KEY` set in the environment to call Groq's free API (`llama-3.3-70b-versatile`).
  - Uses the structured prompt template (including role, context, task, format constraints, and few-shot examples).
  - Enforces JSON validation with an automatic retry loop (up to 2 retry attempts with corrective feedback on schema validation errors).

---

## Local Setup & Execution

### Option A: Local Run via Python/Uvicorn

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
Start Server:

Bash:
uvicorn main:app --host 0.0.0.0 --port 7860
Option B: Local Run via Docker Container
Build Docker Image:

Bash:
docker build -t zepto-support-assistant .
Run Docker Container:

Bash:
docker run -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
Example Call Transcripts (Graded Baseline: MOCK_LLM=1)
Test Case 1: Policy Retrieval Query (policy_question)
HTTP Request:

Bash:
curl -X POST "[http://127.0.0.1:7860/ask](http://127.0.0.1:7860/ask)" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the return policy for damaged items?"}'
Raw JSON Response:

JSON:
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened",
  "sources": [
    "doc_02",
    "doc_06",
    "doc_05"
  ],
  "confidence": 1.0
}
Test Case 2: General Question (general_question)
HTTP Request:

Bash:
curl -X POST "[http://127.0.0.1:7860/ask](http://127.0.0.1:7860/ask)" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
