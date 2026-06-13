import nltk
from typing import List, Dict

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency fallback
    pipeline = None

try:
    from rake_nltk import Rake
except Exception:  # pragma: no cover - optional dependency fallback
    Rake = None


class SentimentKeyphraseAnalyzer:
    """Lazy-loads a transformer sentiment pipeline and RAKE for keyphrase extraction.

    Loading is deferred until the first call to `analyze()` so the API starts quickly.
    """

    def __init__(self):
        self._sentiment = None
        self._rake = None

    def _ensure_loaded(self):
        if self._rake is None and Rake is not None:
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords")
            self._rake = Rake()

        if self._sentiment is None and pipeline is not None:
            # load the lightweight sentiment pipeline on first use
            self._sentiment = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )

    def analyze(self, text: str, max_phrases: int = 5) -> Dict:
        """Analyze text and return sentiment and key phrases.

        Returns:
            {"sentiment": {"label": str, "score": float}, "key_phrases": List[str]}
        """
        if not text or not text.strip():
            return {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "key_phrases": []}

        # ensure heavy resources are available, done lazily
        self._ensure_loaded()

        # limit input length to avoid extremely long tokenization
        snippet = text if len(text) <= 1000 else text[:1000]

        if self._sentiment is not None:
            s = self._sentiment(snippet)[0]
            label = s.get("label", "NEUTRAL")
            score = float(s.get("score", 0.0))
        else:
            lower = snippet.lower()
            positive_markers = ["love", "great", "excellent", "fast", "good", "amazing", "happy"]
            negative_markers = ["bad", "slow", "broken", "terrible", "hate", "poor", "refund"]
            pos_hits = sum(marker in lower for marker in positive_markers)
            neg_hits = sum(marker in lower for marker in negative_markers)
            if pos_hits > neg_hits:
                label = "POSITIVE"
                score = 0.75
            elif neg_hits > pos_hits:
                label = "NEGATIVE"
                score = 0.75
            else:
                label = "NEUTRAL"
                score = 0.5

        if self._rake is not None:
            self._rake.extract_keywords_from_text(text)
            phrases = self._rake.get_ranked_phrases()[:max_phrases]
        else:
            words = [word.strip(".,!?:;\"'()[]{}").lower() for word in text.split()]
            stop_words = {"the", "and", "is", "a", "an", "to", "for", "of", "with", "this", "that", "it", "in"}
            phrases = [word for word in words if len(word) > 3 and word not in stop_words][:max_phrases]

        return {"sentiment": {"label": label, "score": score}, "key_phrases": phrases}


__all__ = ["SentimentKeyphraseAnalyzer"]
