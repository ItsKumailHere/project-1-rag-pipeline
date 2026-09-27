import json
import httpx
import sys

def main():
    try:
        with open('data/golden_dataset.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)
        
    print(f"Found {len(data)} documents. Ingesting...")
    
    with httpx.Client(base_url="http://localhost:8000") as client:
        for doc in data:
            print(f"Ingesting {doc['document_id']}...")
            try:
                response = client.post("/api/v1/ingest", json=doc)
                response.raise_for_status()
                print(f"Success: {response.json()['message']}")
            except httpx.HTTPStatusError as e:
                print(f"Failed to ingest {doc['document_id']}: {e.response.text}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
