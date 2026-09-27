import httpx
import json

def test_eval():
    test_cases = [
        {
            "question": "What is the deployment window for production?",
            "ground_truth": "The deployment window is 09:00 - 15:00 UTC, Monday through Thursday."
        },
        {
            "question": "How do I handle a duplicate charge that happened 2 minutes apart?",
            "ground_truth": "Refund the second charge immediately and apologize to the customer."
        }
    ]
    
    print(f"Running evaluation on {len(test_cases)} test cases...")
    
    with httpx.Client(base_url="http://localhost:8000", timeout=60.0) as client:
        try:
            response = client.post(
                "/api/v1/evaluate", 
                json={"test_cases": test_cases}
            )
            response.raise_for_status()
            result = response.json()
            
            print("\n--- Evaluation Metrics ---")
            print(json.dumps(result['metrics'], indent=2))
                
        except Exception as e:
            print(f"Error: {e}")
            if hasattr(e, 'response'):
                print(e.response.text)

if __name__ == "__main__":
    test_eval()
