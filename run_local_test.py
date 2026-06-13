from fastapi.testclient import TestClient
from app.main import app


def main():
    client = TestClient(app)
    r = client.post("/analyze", json={"text": "I love this product — excellent build and fast shipping."})
    print("status:", r.status_code)
    print(r.json())


if __name__ == "__main__":
    main()
