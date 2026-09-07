from __future__ import annotations

import os
from typing import Any, Dict
from typing_extensions import TypedDict
from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END

from langgraph_observe import instrument_fastapi, observe_graph

# 1. Any existing or new FastAPI backend application
app = FastAPI(title="Payment & Order Service")

# 2. Point to the completely separate LangGraph Observability Server
SERVER_URL = os.getenv("OBSERVE_SERVER_URL", "http://localhost:8765")
instrument_fastapi(app, server_url=SERVER_URL)


# 3. Any LangGraph workflow running inside this backend
class OrderState(TypedDict):
    order_id: str
    amount: float
    status: str
    fraud_check: str


def validate_order(state: OrderState) -> Dict[str, Any]:
    return {"status": "validated"}


def run_fraud_check(state: OrderState) -> Dict[str, Any]:
    risk = "high" if state["amount"] > 10000 else "low"
    return {"fraud_check": risk}


def finalize_order(state: OrderState) -> Dict[str, Any]:
    return {"status": "completed"}


builder = StateGraph(OrderState)
builder.add_node("validate_order", validate_order)
builder.add_node("run_fraud_check", run_fraud_check)
builder.add_node("finalize_order", finalize_order)

builder.add_edge(START, "validate_order")
builder.add_edge("validate_order", "run_fraud_check")
builder.add_edge("run_fraud_check", "finalize_order")
builder.add_edge("finalize_order", END)

order_graph = observe_graph(builder.compile(), name="OrderProcessingWorkflow")


@app.post("/api/checkout")
async def checkout_endpoint(payload: Dict[str, Any]):
    initial_state: OrderState = {
        "order_id": payload.get("order_id", "ord-101"),
        "amount": float(payload.get("amount", 250.0)),
        "status": "pending",
        "fraud_check": "pending",
    }
    result = await order_graph.ainvoke(initial_state)
    return {"message": "Order processed successfully", "order": result}


if __name__ == "__main__":
    import uvicorn
    print("\n📦 Starting Separate Backend on port 8000...")
    print(f"📡 Streaming traces to separate server at: {SERVER_URL}\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
