from fastapi.testclient import TestClient
from main import app
client = TestClient(app)
response = client.post("/ask", json={"query": "What is the policy for returns?"})
print(response.json())
