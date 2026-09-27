import httpx
import json

def test_query():
    query = "Why did the database go down in Q3 and how long was it down?"
    print(f"Querying: '{query}'")
    
    with httpx.Client(base_url="http://localhost:8000") as client:
        try:
            response = client.post(
                "/api/v1/query", 
                json={"query": query, "top_k": 3, "hybrid": False}
            )
            response.raise_for_status()
            result = response.json()
            
            print("\n--- Answer ---")
            print(result['answer'])
            
            print("\n--- Citations ---")
            for i, citation in enumerate(result['citations'], 1):
                print(f"{i}. Document: {citation['document_id']}")
                print(f"   Quote: \"{citation['quote']}\"")
                
        except Exception as e:
            print(f"Error: {e}")
            if hasattr(e, 'response'):
                print(e.response.text)

if __name__ == "__main__":
    test_query()
