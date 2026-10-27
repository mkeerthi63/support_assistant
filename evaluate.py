"""Offline evaluation of retrieval quality and intent routing on a labeled query set.

Usage: python evaluate.py
"""
import json
import os

import main

with open(os.path.join(os.path.dirname(__file__), "eval_set.json"), encoding="utf-8") as f:
    EVAL_SET = json.load(f)


def evaluate_retrieval():
    policy = [e for e in EVAL_SET if e["doc"]]
    hit1 = hit3 = 0
    reciprocal_ranks = []
    misses = []
    for e in policy:
        ids = [d["id"] for d in main.retrieve(e["query"], k=3)]
        hit1 += ids[:1] == [e["doc"]]
        hit3 += e["doc"] in ids
        rank = ids.index(e["doc"]) + 1 if e["doc"] in ids else None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if ids[:1] != [e["doc"]]:
            misses.append((e["query"], e["doc"], ids))
    n = len(policy)
    return {"n": n, "hit@1": hit1 / n, "hit@3": hit3 / n, "mrr": sum(reciprocal_ranks) / n, "misses": misses}


def evaluate_router(mode):
    main.ROUTER_MODE = mode
    correct = 0
    errors = []
    for e in EVAL_SET:
        state = main.classify_intent({"query": e["query"], "classification": None, "retrieved_docs": [], "response": None})
        if state["classification"] == e["intent"]:
            correct += 1
        else:
            errors.append((e["query"], e["intent"], state["classification"]))
    return {"n": len(EVAL_SET), "accuracy": correct / len(EVAL_SET), "errors": errors}


if __name__ == "__main__":
    r = evaluate_retrieval()
    print(f"=== RETRIEVAL ({r['n']} labeled policy queries, top-k=3) ===")
    print(f"Hit@1: {r['hit@1']:.1%} | Hit@3: {r['hit@3']:.1%} | MRR: {r['mrr']:.3f}")
    for q, want, got in r["misses"]:
        print(f"  top-1 miss: {q!r} expected {want}, got {got}")

    original_mode = main.ROUTER_MODE
    for mode in ("keyword", "hybrid"):
        res = evaluate_router(mode)
        print(f"\n=== ROUTER: {mode} ({res['n']} labeled queries) ===")
        print(f"Accuracy: {res['accuracy']:.1%}")
        for q, want, got in res["errors"]:
            print(f"  misrouted: {q!r} expected {want}, got {got}")
    main.ROUTER_MODE = original_mode
