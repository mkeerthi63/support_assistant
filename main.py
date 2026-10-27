import os
import json
import glob
from typing import List, Optional, Literal, TypedDict
from pydantic import BaseModel, Field, ValidationError
from fastapi import FastAPI
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langgraph.graph import StateGraph, START, END

MOCK_LLM = os.getenv("MOCK_LLM", "1").strip() == "1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_RETRIES = 2

# "keyword" = original keyword router, "hybrid" = keywords OR semantic similarity to the policy corpus
ROUTER_MODE = os.getenv("ROUTER_MODE", "hybrid").strip().lower()
# Cosine distance below which a query is treated as a policy question even without a keyword
SEMANTIC_ROUTE_THRESHOLD = float(os.getenv("SEMANTIC_ROUTE_THRESHOLD", "0.75"))

embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection(
    name="zepto_policies",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

def init_db():
    if collection.count() == 0:
        doc_files = sorted(glob.glob("./docs/doc_*.txt"))
        ids, documents, metadatas = [], [], []
        for file_path in doc_files:
            doc_id = os.path.basename(file_path).replace(".txt", "")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            ids.append(doc_id)
            documents.append(content)
            metadatas.append({"source_doc": doc_id})
        if documents:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

init_db()

STRUCTURED_PROMPT_TEMPLATE = """
Role:
You are Zepto's Support Assistant.

Context:
{context}

Task:
Answer query using ONLY context:
{query}

Negative Constraints:
- Do NOT answer using information not present in context.
- If the context does not contain the answer, say so and return an empty sources list.
- Only cite source ids that appear in the context (e.g. "doc_02").

Format Requirements:
Return strictly JSON matching:
{{"answer": "...", "sources": ["..."], "confidence": 0.0}}
confidence is a number between 0 and 1 for how fully the context answers the query.

Example:
Query: Is phone support available?
Output: {{"answer": "No. Zepto does not offer phone support; use 24/7 in-app chat or email.", "sources": ["doc_08"], "confidence": 0.95}}
"""

class QueryRequest(BaseModel):
    query: str

class SupportResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

class SupportState(TypedDict):
    query: str
    classification: Optional[Literal["policy_question", "general_question"]]
    retrieved_docs: List[dict]
    response: Optional[SupportResponse]

KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

def retrieve(query: str, k: int = 3) -> List[dict]:
    results = collection.query(query_texts=[query], n_results=k, include=["documents", "distances"])
    ids = results["ids"][0] if results["ids"] else []
    texts = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results.get("distances") else [1.0] * len(ids)
    return [{"id": i, "text": t, "distance": d} for i, t, d in zip(ids, texts, distances)]

def similarity(doc: dict) -> float:
    return max(0.0, min(1.0, 1.0 - doc["distance"]))

def classify_intent(state: SupportState) -> SupportState:
    query_lower = state["query"].lower()
    is_policy = any(kw in query_lower for kw in KEYWORDS)
    if not is_policy and ROUTER_MODE == "hybrid":
        top = retrieve(state["query"], k=1)
        is_policy = bool(top) and top[0]["distance"] < SEMANTIC_ROUTE_THRESHOLD
    classification = "policy_question" if is_policy else "general_question"
    return {**state, "classification": classification}

def route_intent(state: SupportState) -> str:
    return state["classification"]

_groq_client = None

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client

def extractive_answer(docs: List[dict]) -> SupportResponse:
    if not docs:
        return SupportResponse(answer="No context available.", sources=[], confidence=0.0)
    return SupportResponse(
        answer=f"Based on the retrieved context: {docs[0]['text'][:200]}",
        sources=[d["id"] for d in docs],
        confidence=round(similarity(docs[0]), 3)
    )

def llm_answer(query: str, docs: List[dict]) -> SupportResponse:
    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    messages = [{"role": "user", "content": STRUCTURED_PROMPT_TEMPLATE.format(context=context, query=query)}]
    allowed_ids = {d["id"] for d in docs}

    for _ in range(MAX_RETRIES + 1):
        completion = _get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or ""
        try:
            resp = SupportResponse.model_validate_json(raw)
        except ValidationError as e:
            # Feed the schema error back so the model can correct its output
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Your JSON failed validation: {e.errors()}. Return corrected JSON only."})
            continue
        # Drop hallucinated citations that were not in the retrieved context
        resp.sources = [s for s in resp.sources if s in allowed_ids]
        return resp

    return extractive_answer(docs)

def retrieve_and_answer(state: SupportState) -> SupportState:
    docs = retrieve(state["query"], k=3)
    resp = extractive_answer(docs) if MOCK_LLM else llm_answer(state["query"], docs)
    return {**state, "retrieved_docs": docs, "response": resp}

def direct_answer(state: SupportState) -> SupportState:
    resp = SupportResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )
    return {**state, "retrieved_docs": [], "response": resp}

builder = StateGraph(SupportState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)

builder.add_edge(START, "classify_intent")
builder.add_conditional_edges("classify_intent", route_intent, {
    "policy_question": "retrieve_and_answer",
    "general_question": "direct_answer"
})
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

graph = builder.compile()

app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=SupportResponse)
def ask_question(request: QueryRequest):
    initial_state: SupportState = {
        "query": request.query,
        "classification": None,
        "retrieved_docs": [],
        "response": None
    }
    final_state = graph.invoke(initial_state)
    return final_state["response"]
