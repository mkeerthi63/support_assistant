import os
import json
import glob
from typing import List, Optional, Literal, TypedDict
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langgraph.graph import StateGraph, START, END

MOCK_LLM = os.getenv("MOCK_LLM", "1").strip() == "1"

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

Format Requirements:
Return strictly JSON matching:
{{"answer": "...", "sources": ["..."], "confidence": 1.0}}
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

def classify_intent(state: SupportState) -> SupportState:
    query_lower = state["query"].lower()
    if any(kw in query_lower for kw in KEYWORDS):
        classification = "policy_question"
    else:
        classification = "general_question"
    return {**state, "classification": classification}

def route_intent(state: SupportState) -> str:
    return state["classification"]

def retrieve_and_answer(state: SupportState) -> SupportState:
    results = collection.query(query_texts=[state["query"]], n_results=3)
    retrieved_ids = results["ids"][0] if results["ids"] else []
    retrieved_texts = results["documents"][0] if results["documents"] else []

    retrieved_docs = [{"id": r_id, "text": r_text} for r_id, r_text in zip(retrieved_ids, retrieved_texts)]
    top_snippet = retrieved_texts[0][:200] if retrieved_texts else "No context available."

    resp = SupportResponse(
        answer=f"Based on the retrieved context: {top_snippet}",
        sources=retrieved_ids,
        confidence=1.0
    )
    return {**state, "retrieved_docs": retrieved_docs, "response": resp}

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
