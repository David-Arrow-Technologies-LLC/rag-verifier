import pytest

from verifier import RAGVerifier
from nli_provider import FakeNLIProvider


def make_verifier(
    default_scores=None,
    pass_threshold=0.90,
    fail_threshold=0.90
):
    chunks = {
        "mi-001": (
            "Myocardial infarction occurs when blood flow "
            "to part of the heart muscle is blocked or "
            "severely reduced."
        ),
        "mi-002": (
            "Symptoms may include chest discomfort, "
            "shortness of breath, nausea, or sweating."
        ),
        "pna-001": (
            "Pneumonia is an infection involving the lungs."
        )
    }

    provider = FakeNLIProvider(
        default_scores=default_scores
    )

    verifier = RAGVerifier(
        chunks=chunks,
        retrieved_ids={"mi-001", "mi-002"},
        pass_threshold=pass_threshold,
        fail_threshold=fail_threshold,
        nli_provider=provider
    )

    return verifier, provider


def test_fake_supported_claim_passes():
    verifier, _ = make_verifier({
        "contradiction": 0.01,
        "entailment": 0.98,
        "neutral": 0.01
    })

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [mi-001]"
    )

    assert result["status"] == "PASS"
    assert result["trusted_release"] is True


def test_fake_low_entailment_requires_review():
    verifier, _ = make_verifier({
        "contradiction": 0.02,
        "entailment": 0.58,
        "neutral": 0.40
    })

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [mi-001]"
    )

    assert result["status"] == "REVIEW"
    assert result["trusted_release"] is False


def test_fake_high_contradiction_fails():
    verifier, _ = make_verifier({
        "contradiction": 0.97,
        "entailment": 0.01,
        "neutral": 0.02
    })

    result = verifier.verify_answer(
        "A heart attack is caused by pneumonia. [mi-001]"
    )

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False


def test_fake_neutral_requires_review():
    verifier, _ = make_verifier({
        "contradiction": 0.01,
        "entailment": 0.01,
        "neutral": 0.98
    })

    result = verifier.verify_answer(
        "The heart attack occurred on Monday. [mi-001]"
    )

    assert result["status"] == "REVIEW"


def test_uncited_claim_fails_without_nli():
    verifier, _ = make_verifier()

    result = verifier.verify_answer(
        "A heart attack requires immediate surgery."
    )

    assert result["status"] == "FAIL"
    assert result["claims"][0]["reason"] == "UNCITED_FACTUAL_CLAIM"


def test_fabricated_citation_fails_without_nli():
    verifier, _ = make_verifier()

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [fake-999]"
    )

    verification = result["claims"][0]["verification"][0]

    assert result["status"] == "FAIL"
    assert verification["reason"] == "CITATION_NOT_FOUND"


def test_unretrieved_citation_fails_without_nli():
    verifier, _ = make_verifier()

    result = verifier.verify_answer(
        "Pneumonia involves the lungs. [pna-001]"
    )

    verification = result["claims"][0]["verification"][0]

    assert result["status"] == "FAIL"
    assert verification["reason"] == "CITATION_NOT_RETRIEVED"


def test_pair_specific_fake_response():
    verifier, provider = make_verifier()

    premise = verifier.chunks["mi-001"]
    hypothesis = "The heart attack occurred on Monday."

    provider.set_response(
        premise,
        hypothesis,
        {
            "contradiction": 0.01,
            "entailment": 0.01,
            "neutral": 0.98
        }
    )

    result = verifier.verify_citation(
        hypothesis,
        "mi-001"
    )

    assert result["status"] == "REVIEW"


def test_exact_pass_boundary():
    verifier, provider = make_verifier({
        "contradiction": 0.01,
        "entailment": 0.90,
        "neutral": 0.09
    })

    result = verifier.classify_nli_scores(
        provider.default_scores
    )

    assert result["status"] == "PASS"


def test_exact_fail_boundary():
    verifier, provider = make_verifier({
        "contradiction": 0.90,
        "entailment": 0.01,
        "neutral": 0.09
    })

    result = verifier.classify_nli_scores(
        provider.default_scores
    )

    assert result["status"] == "FAIL"


def test_provider_missing_label_fails_closed():
    verifier, _ = make_verifier(
        default_scores={
            "entailment": 0.98,
            "contradiction": 0.02
        }
    )

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [mi-001]"
    )

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False


def test_provider_extra_label_fails_closed():
    verifier, _ = make_verifier(
        default_scores={
            "entailment": 0.97,
            "contradiction": 0.01,
            "neutral": 0.01,
            "unexpected": 0.01
        }
    )

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [mi-001]"
    )

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False


def test_provider_nan_score_fails_closed():
    verifier, _ = make_verifier(
        default_scores={
            "entailment": float("nan"),
            "contradiction": 0.01,
            "neutral": 0.99
        }
    )

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [mi-001]"
    )

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False


def test_provider_infinite_score_fails_closed():
    verifier, _ = make_verifier(
        default_scores={
            "entailment": float("inf"),
            "contradiction": 0.0,
            "neutral": 0.0
        }
    )

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [mi-001]"
    )

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False


def test_provider_negative_score_fails_closed():
    verifier, _ = make_verifier(
        default_scores={
            "entailment": 0.95,
            "contradiction": -0.01,
            "neutral": 0.06
        }
    )

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [mi-001]"
    )

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False


def test_provider_score_above_one_fails_closed():
    verifier, _ = make_verifier(
        default_scores={
            "entailment": 1.01,
            "contradiction": 0.0,
            "neutral": -0.01
        }
    )

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [mi-001]"
    )

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False


def test_provider_non_numeric_score_fails_closed():
    verifier, _ = make_verifier(
        default_scores={
            "entailment": "0.98",
            "contradiction": 0.01,
            "neutral": 0.01
        }
    )

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [mi-001]"
    )

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False


def test_provider_invalid_probability_sum_fails_closed():
    verifier, _ = make_verifier(
        default_scores={
            "entailment": 0.80,
            "contradiction": 0.10,
            "neutral": 0.05
        }
    )

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [mi-001]"
    )

    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False


def test_provider_valid_probability_sum_is_accepted():
    verifier, _ = make_verifier(
        default_scores={
            "entailment": 0.98,
            "contradiction": 0.01,
            "neutral": 0.01
        }
    )

    result = verifier.verify_answer(
        "A heart attack occurs when blood flow is blocked. [mi-001]"
    )

    assert result["status"] == "PASS"
    assert result["trusted_release"] is True


# Decimal boundaries must not fragment the hypothesis or detach its citations.


@pytest.mark.parametrize("value", ["9.0", "-0.25", "+12.345", "0.001", "９.０"])
def test_decimal_claim_keeps_terminal_citations(value):
    verifier, _ = make_verifier()
    sentence = f"Synthetic value is {value}. [mi-001] [mi-002]"
    assert verifier.split_claims(sentence) == [sentence]
    assert verifier.decompose_atomic_claims(sentence) == [{
        "claim": f"Synthetic value is {value}.",
        "citations": ["mi-001", "mi-002"],
    }]


@pytest.mark.parametrize("terminal", [".", "!", "?"])
def test_decimal_does_not_merge_adjacent_sentences(terminal):
    verifier, _ = make_verifier()
    first = f"Synthetic value is 9.0{terminal} [mi-001]"
    second = "Another value is 2.5. [mi-002]"
    assert verifier.split_claims(first + "\n" + second) == [first, second]
    # Existing no-whitespace sentence-boundary behavior remains available.
    assert verifier.split_claims(first + second) == [first, second]


@pytest.mark.parametrize("suffix", ["", " [mi-001]"])
def test_unterminated_decimal_claim_is_retained(suffix):
    verifier, _ = make_verifier()
    text = "Synthetic value is 9.0" + suffix
    assert verifier.split_claims(text) == [text]


@pytest.mark.parametrize("scores,status", [
    ({"entailment": .98, "contradiction": .01, "neutral": .01}, "PASS"),
    ({"entailment": .01, "contradiction": .98, "neutral": .01}, "FAIL"),
    ({"entailment": .01, "contradiction": .01, "neutral": .98}, "REVIEW"),
])
def test_decimal_full_answer_uses_intact_hypothesis(scores, status):
    # Default contradiction makes accidental hypothesis fragmentation observable.
    verifier, provider = make_verifier({
        "entailment": .01, "contradiction": .98, "neutral": .01,
    })
    hypothesis = "Synthetic value is 9.0."
    provider.set_response(verifier.chunks["mi-001"], hypothesis, scores)
    answer = hypothesis + " [mi-001]"
    result = verifier.verify_answer(answer)
    assert result == verifier.verify_answer(answer)
    assert result["status"] == status
    assert result["trusted_release"] is (status == "PASS")
    assert result["citation_coverage"] == {
        "cited_claims": 1, "total_claims": 1,
        "coverage_ratio": 1.0, "coverage_percent": 100.0,
    }
    assert result["claims"][0]["claim"] == hypothesis
    assert result["claims"][0]["citations"] == ["mi-001"]


@pytest.mark.parametrize("citation,reason", [
    ("", "UNCITED_FACTUAL_CLAIM"),
    (" [fake-999]", "CITATION_NOT_FOUND"),
    (" [pna-001]", "CITATION_NOT_RETRIEVED"),
])
def test_decimal_claim_still_fails_closed(citation, reason):
    verifier, _ = make_verifier()
    result = verifier.verify_answer("Synthetic value is 9.0." + citation)
    assert result["status"] == "FAIL"
    assert result["trusted_release"] is False
    assert len(result["claims"]) == 1
    claim = result["claims"][0]
    assert claim["claim"] == "Synthetic value is 9.0."
    assert (claim["verification"][0]["reason"] if citation else claim["reason"]) == reason
