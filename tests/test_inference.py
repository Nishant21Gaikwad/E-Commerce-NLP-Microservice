from model.inference import SentimentKeyphraseAnalyzer


def test_analyze_basic():
    a = SentimentKeyphraseAnalyzer()
    res = a.analyze("I love this product. Fast shipping and great quality.")
    assert "sentiment" in res
    assert "key_phrases" in res
    assert isinstance(res["key_phrases"], list)
