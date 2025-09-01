from genai_analytics.rag.retriever import TFIDFRetriever

def test_retrieve():
    r = TFIDFRetriever.fit_from_folder('data/docs')
    out = r.retrieve('What is RAG?', k=2)
    assert len(out) == 2
