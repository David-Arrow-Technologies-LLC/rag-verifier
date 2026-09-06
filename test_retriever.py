import pytest

from embedding_provider import FakeEmbeddingProvider
from retriever import Retriever

class StubEmbeddingProvider:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def encode(self, texts):
        return self.outputs.pop(0)


def make_retriever():
    chunks = {
        "mi-001": "Heart attack is related to blocked blood flow.",
        "pna-001": "Pneumonia is an infection of the lungs.",
        "htn-001": "Hypertension means elevated blood pressure."
    }

    provider = FakeEmbeddingProvider(
        embeddings={
            "What causes a heart attack?": [1.0, 0.0, 0.0],
            chunks["mi-001"]: [0.95, 0.05, 0.0],
            chunks["pna-001"]: [0.0, 1.0, 0.0],
            chunks["htn-001"]: [0.2, 0.1, 0.9]
        }
    )

    return Retriever(
        chunks=chunks,
        embedding_provider=provider
    )


def test_retrieval_ranks_most_similar_chunk_first():
    retriever = make_retriever()

    results = retriever.retrieve(
        "What causes a heart attack?",
        top_k=3
    )

    assert results[0]["chunk_id"] == "mi-001"


def test_top_k_limits_number_of_results():
    retriever = make_retriever()

    results = retriever.retrieve(
        "What causes a heart attack?",
        top_k=2
    )

    assert len(results) == 2


def test_results_are_sorted_by_score_descending():
    retriever = make_retriever()

    results = retriever.retrieve(
        "What causes a heart attack?",
        top_k=3
    )

    scores = [
        result["score"]
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True
    )


def test_min_score_filters_low_similarity_results():
    retriever = make_retriever()

    results = retriever.retrieve(
        "What causes a heart attack?",
        top_k=3,
        min_score=0.8
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "mi-001"


def test_invalid_top_k_is_rejected():
    retriever = make_retriever()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero"
    ):
        retriever.retrieve(
            "What causes a heart attack?",
            top_k=0
        )


def test_dimension_mismatch_is_rejected():
    retriever = make_retriever()

    with pytest.raises(
        ValueError,
        match="Embedding vectors must have the same dimension"
    ):
        retriever.cosine_similarity(
            [1.0, 0.0],
            [1.0, 0.0, 0.0]
        )


def test_zero_vector_returns_zero_similarity():
    retriever = make_retriever()

    score = retriever.cosine_similarity(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    )

    assert score == 0.0
@pytest.mark.integration
def test_real_embedding_provider_retrieves_relevant_chunk():
    from embedding_provider import (
        SentenceTransformerEmbeddingProvider
    )

    chunks = {
        "mi-001": (
            "A myocardial infarction occurs when blood flow "
            "to part of the heart muscle is blocked."
        ),
        "pna-001": (
            "Pneumonia is an infection that affects "
            "the lungs."
        ),
        "htn-001": (
            "Hypertension is persistently elevated "
            "arterial blood pressure."
        )
    }

    provider = SentenceTransformerEmbeddingProvider()

    retriever = Retriever(
        chunks=chunks,
        embedding_provider=provider
    )

    results = retriever.retrieve(
        "What happens when blood flow to the heart muscle is blocked?",
        top_k=3
    )

    assert results[0]["chunk_id"] == "mi-001"

def make_stub_retriever(outputs):
    return Retriever(
        chunks={
            "c1": "First chunk.",
            "c2": "Second chunk."
        },
        embedding_provider=StubEmbeddingProvider(
            outputs
        )
    )


def test_provider_returning_too_few_chunk_vectors_is_rejected():
    retriever = make_stub_retriever(
        outputs=[
            [[1.0, 0.0]],
            [[1.0, 0.0]]
        ]
    )

    with pytest.raises(
        ValueError,
        match="unexpected number of vectors"
    ):
        retriever.retrieve("query")


def test_provider_returning_too_many_chunk_vectors_is_rejected():
    retriever = make_stub_retriever(
        outputs=[
            [[1.0, 0.0]],
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5]
            ]
        ]
    )

    with pytest.raises(
        ValueError,
        match="unexpected number of vectors"
    ):
        retriever.retrieve("query")


def test_provider_returning_no_query_vector_is_rejected():
    retriever = make_stub_retriever(
        outputs=[
            [],
            [
                [1.0, 0.0],
                [0.0, 1.0]
            ]
        ]
    )

    with pytest.raises(
        ValueError,
        match="unexpected number of vectors"
    ):
        retriever.retrieve("query")


def test_provider_returning_inconsistent_chunk_dimensions_is_rejected():
    retriever = make_stub_retriever(
        outputs=[
            [[1.0, 0.0]],
            [
                [1.0, 0.0],
                [0.0, 1.0, 0.0]
            ]
        ]
    )

    with pytest.raises(
        ValueError,
        match="consistent dimensions"
    ):
        retriever.retrieve("query")


def test_query_and_chunk_dimension_mismatch_is_rejected():
    retriever = make_stub_retriever(
        outputs=[
            [[1.0, 0.0]],
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ]
        ]
    )

    with pytest.raises(
        ValueError,
        match="Query and chunk embeddings must have the same dimension"
    ):
        retriever.retrieve("query")


@pytest.mark.parametrize(
    "invalid_value",
    [
        "1.0",
        None,
        True
    ]
)
def test_provider_returning_nonnumeric_embedding_value_is_rejected(
    invalid_value
):
    retriever = make_stub_retriever(
        outputs=[
            [[1.0, 0.0]],
            [
                [invalid_value, 0.0],
                [0.0, 1.0]
            ]
        ]
    )

    with pytest.raises(
        ValueError,
        match="Embedding values must be numeric"
    ):
        retriever.retrieve("query")


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf")
    ]
)
def test_provider_returning_nonfinite_embedding_value_is_rejected(
    invalid_value
):
    retriever = make_stub_retriever(
        outputs=[
            [[1.0, 0.0]],
            [
                [invalid_value, 0.0],
                [0.0, 1.0]
            ]
        ]
    )

    with pytest.raises(
        ValueError,
        match="Embedding values must be finite"
    ):
        retriever.retrieve("query")


def test_provider_returning_nonlist_batch_is_rejected():
    retriever = make_stub_retriever(
        outputs=[
            (1.0, 0.0)
        ]
    )

    with pytest.raises(
        ValueError,
        match="Embedding provider output must be a list"
    ):
        retriever.retrieve("query")


def test_provider_returning_nonlist_vector_is_rejected():
    retriever = make_stub_retriever(
        outputs=[
            [(1.0, 0.0)]
        ]
    )

    with pytest.raises(
        ValueError,
        match="Each embedding vector must be a list"
    ):
        retriever.retrieve("query")


def test_provider_returning_empty_vector_is_rejected():
    retriever = make_stub_retriever(
        outputs=[
            [[]]
        ]
    )

    with pytest.raises(
        ValueError,
        match="Embedding vectors must not be empty"
    ):
        retriever.retrieve("query")


def test_expected_dimension_is_enforced_for_query_vectors():
    retriever = Retriever(
        chunks={"c1": "First chunk."},
        embedding_provider=StubEmbeddingProvider(
            outputs=[[[1.0, 0.0, 0.0]]]
        ),
        expected_dimension=2,
    )

    with pytest.raises(
        ValueError,
        match="does not match expected_dimension",
    ):
        retriever.retrieve("query")


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "3"])
def test_invalid_expected_dimension_is_rejected(invalid):
    with pytest.raises(
        ValueError,
        match="expected_dimension must be a positive integer",
    ):
        Retriever({}, object(), expected_dimension=invalid)
