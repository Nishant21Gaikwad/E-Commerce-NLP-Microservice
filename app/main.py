from fastapi import FastAPI
from pydantic import BaseModel

from model.inference import SentimentKeyphraseAnalyzer


class TextPayload(BaseModel):
    text: str


app = FastAPI(title="E-Commerce Review Sentiment & Intent Analyzer")
analyzer = SentimentKeyphraseAnalyzer()


@app.post("/analyze")
def analyze(payload: TextPayload):
    """Analyze product review text and return sentiment + key phrases."""
    result = analyzer.analyze(payload.text)
    return result
