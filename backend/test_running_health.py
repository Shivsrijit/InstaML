import requests

try:
    res = requests.get("http://127.0.0.1:8000/health")
    print("Health check response:", res.status_code, res.json())
except Exception as e:
    print("Error calling health check:", e)
