from rag.rag_eval import evaluate_batch, evaluate_rag_response
from rag.schemas import RetrievedChunk


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        title="PRSV info",
        category="symptoms",
        audience="farmer",
        text=text,
        tags=["prsv"],
        similarity_score=1.0,
    )


def test_faithful_answer_scores_higher_than_unrelated_answer() -> None:
    query = "What are PRSV leaf symptoms?"
    context_chunks = [_chunk("PRSV causes yellow mosaic patterns and leaf distortion on papaya.")]

    faithful_answer = "PRSV causes yellow mosaic patterns and leaf distortion on papaya leaves."
    unrelated_answer = "The weather today is sunny with a light breeze across the region."

    faithful_score = evaluate_rag_response(query, faithful_answer, context_chunks)
    unrelated_score = evaluate_rag_response(query, unrelated_answer, context_chunks)

    assert faithful_score.faithfulness >= unrelated_score.faithfulness
    assert faithful_score.answer_relevance >= unrelated_score.answer_relevance


def test_evaluate_batch_produces_averages() -> None:
    turns = [
        {
            "query": "How does PRSV spread?",
            "answer": "PRSV spreads through aphid vectors moving between plants.",
            "retrieved_chunks": [_chunk("PRSV spreads mainly through aphid vectors.")],
        },
        {
            "query": "How severe is this infection?",
            "answer": "The infection is moderate based on symptom coverage.",
            "retrieved_chunks": [_chunk("Symptom coverage determines severity classification.")],
        },
    ]

    report = evaluate_batch(turns)

    assert len(report["per_turn"]) == 2
    assert set(report["averages"].keys()) == {
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
    }


def test_evaluate_batch_empty_input() -> None:
    report = evaluate_batch([])
    assert report == {"per_turn": [], "averages": {}}
