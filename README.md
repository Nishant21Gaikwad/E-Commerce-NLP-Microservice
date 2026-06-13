# E-Commerce NLP Microservice

Containerized FastAPI microservice for product-review analysis.

The service takes raw customer review text and returns:
- sentiment: `POSITIVE`, `NEGATIVE`, or `NEUTRAL`
- key phrases: top extracted terms/phrases from the review

The primary NLP path uses Hugging Face Transformers (DistilBERT sentiment model, PyTorch backend). A lightweight fallback path is also included so the API can still respond if optional heavy packages are not available yet.

## Tech Stack

- Python
- FastAPI + Uvicorn
- PyTorch
- Hugging Face Transformers
- NLTK + RAKE
- Docker / Docker Compose
- Pytest

## Project Structure

```text
.
|-- app/
|   |-- __init__.py
|   `-- main.py                # FastAPI app and /analyze endpoint
|-- model/
|   |-- __init__.py
|   `-- inference.py           # Sentiment + keyphrase analyzer
|-- tests/
|   `-- test_inference.py
|-- run_local_test.py          # quick in-process endpoint smoke test
|-- requirements.txt
|-- Dockerfile
|-- docker-compose.yml
`-- README.md
```

## API

### `POST /analyze`

Request body:

```json
{
	"text": "I love this product. Fast shipping and excellent build quality."
}
```

Example response:

```json
{
	"sentiment": {
		"label": "POSITIVE",
		"score": 0.75
	},
	"key_phrases": [
		"love",
		"product",
		"excellent",
		"build",
		"fast"
	]
}
```

## Run Locally (Windows PowerShell)

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

In another terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/analyze" -Method POST -ContentType "application/json" -Body '{"text":"I love this product, excellent quality and fast delivery."}'
```

## Run with Docker

```bash
docker build -t ecom-nlp-microservice:latest .
docker run --rm -p 8080:8080 ecom-nlp-microservice:latest
```

Or with Compose:

```bash
docker compose up --build
```

## Test

```powershell
. .venv\Scripts\Activate.ps1
python -m pytest -q
```

## Notes

- First run may take longer because model artifacts and NLP resources can be downloaded.
- If transformer dependencies are unavailable, the service still returns structured output through the built-in fallback mode.

## Resume-Ready Project Summary

- Built a containerized FastAPI microservice for e-commerce review analysis with sentiment classification and key phrase extraction.
- Integrated Transformer-based sentiment inference (DistilBERT on PyTorch) and deployed via Uvicorn.
- Added test coverage, local/dev scripts, and Dockerized deployment for reproducible serving.
