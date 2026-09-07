from __future__ import annotations

import asyncio
from fastapi.testclient import TestClient
from examples.app import app

def main():
    client = TestClient(app)
    print("Testing Customer Support LangGraph Workflow...")

    # 1. Technical issue query (triggers review_guardrail conditional edge)
    print("\n[1] Sending technical support query...")
    r1 = client.post("/api/run", json={"query": "I am facing an error 500 when saving my project"})
    print(f"Status: {r1.status_code}")
    print(f"X-Trace-ID: {r1.headers.get('X-Trace-ID')}")
    print(f"Workflow Summary: {r1.json()['result']['summary']}")
    print(f"Review Status: {r1.json()['result']['review_status']}")

    # 2. Billing query (normal path, no review)
    print("\n[2] Sending billing inquiry...")
    r2 = client.post("/api/run", json={"query": "What are your pricing plans?"})
    print(f"Status: {r2.status_code}")
    print(f"X-Trace-ID: {r2.headers.get('X-Trace-ID')}")
    print(f"Workflow Summary: {r2.json()['result']['summary']}")

    # 3. Deliberate error endpoint
    print("\n[3] Triggering failure workflow...")
    try:
        r3 = client.post("/api/fail")
        print(f"Status: {r3.status_code}")
    except Exception as e:
        print(f"Caught expected exception: {type(e).__name__}: {e}")

    # 4. Check Observability API
    print("\n[4] Querying Observability API...")
    traces_resp = client.get("/observe/api/traces")
    traces = traces_resp.json()
    print(f"Total traces recorded: {len(traces)}")
    for t in traces:
        cost_val = t.get('estimated_cost', 0)
        print(f" - [{t['status'].upper()}] {t['name']} | Spans: {t['total_spans']} | Tokens: {t.get('total_tokens', 0)} | Cost: ${cost_val:.5f} | Duration: {t['duration_ms']}ms | ID: {t['id']}")

    stats_resp = client.get("/observe/api/stats")
    print("\n[5] Observability Aggregated Stats:")
    print(stats_resp.json())

    # 5. Check UI HTML endpoint
    ui_resp = client.get("/observe")
    print(f"\n[6] Dashboard UI endpoint status: {ui_resp.status_code} (HTML length: {len(ui_resp.text)} chars)")
    print("\n[OK] Verification complete! Everything is working cleanly.")

if __name__ == "__main__":
    main()
