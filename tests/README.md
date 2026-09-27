# Testing & Evaluation

This directory contains scripts to test the RAG pipeline end-to-end.

## Quick Start

1. Start the API server:
   ```bash
   cd ..
   uvicorn src.api.app:app --reload
   ```

2. Load the golden dataset:
   ```bash
   python load_data.py
   ```

3. Test a single query:
   ```bash
   python query_test.py
   ```

4. Run the quantitative evaluation (requires RAGAS):
   ```bash
   python eval_test.py
   ```

## About the Golden Dataset

The `data/golden_dataset.json` acts as our synthetic knowledge base. It contains realistic engineering and support documents designed specifically to test:
- Fact retrieval (e.g., "when is the deployment window?")
- Logical reasoning (e.g., "what to do if charges are 2 mins apart?")
- Root cause analysis (e.g., "why did the database go down?")
