"""Deterministic, synthetic verifier demonstration; no real model or patient data.

Usage: python synthetic_verifier_demo.py
Output: JSON on stdout. This is fixture evidence, not model qualification.
"""
import json

from nli_provider import FakeNLIProvider
from release_policy import ReleasePolicy
from verifier import RAGVerifier


CHUNKS = {
    "syn-001": "The synthetic reference states that sample A is stored at 4 degrees Celsius.",
    "syn-002": "The synthetic reference states that sample B is stored at 20 degrees Celsius.",
}
RETRIEVED = {"syn-001"}
CASES = (
    ("supported", "Sample A is stored at 4 degrees Celsius. [syn-001]",
     {"entailment": 0.98, "contradiction": 0.01, "neutral": 0.01},
     "PASS", "RELEASE"),
    ("uncertain", "Sample A is stored at 4 degrees Celsius. [syn-001]",
     {"entailment": 0.40, "contradiction": 0.05, "neutral": 0.55},
     "REVIEW", "HOLD_FOR_REVIEW"),
    ("contradicted", "Sample A is stored at 40 degrees Celsius. [syn-001]",
     {"entailment": 0.01, "contradiction": 0.98, "neutral": 0.01},
     "FAIL", "BLOCK"),
    ("fabricated_citation", "Sample A is stored at 4 degrees Celsius. [fake-999]",
     None, "FAIL", "BLOCK"),
    ("unretrieved_citation", "Sample B is stored at 20 degrees Celsius. [syn-002]",
     None, "FAIL", "BLOCK"),
)


def run_cases():
    records = []
    for case_id, answer, scores, expected_status, expected_action in CASES:
        provider = FakeNLIProvider(default_scores=scores)
        verifier = RAGVerifier(
            chunks=CHUNKS, retrieved_ids=RETRIEVED, nli_provider=provider
        )
        verification = verifier.verify_answer(answer)
        decision = ReleasePolicy().decide(verification)
        record = {
            "case_id": case_id,
            "expected_status": expected_status,
            "observed_status": verification["status"],
            "expected_action": expected_action,
            "observed_action": decision["action"],
            "reason": verification["reason"],
            "citation_coverage": verification["citation_coverage"],
            "trusted_release": verification["trusted_release"],
        }
        if (record["observed_status"], record["observed_action"]) != (
            expected_status, expected_action
        ):
            raise AssertionError("Fixture expectation mismatch: " + case_id)
        records.append(record)
    return {
        "evidence_type": "synthetic_deterministic_fixture",
        "provider": "FakeNLIProvider",
        "clinical_validation": False,
        "cases": records,
        "summary": {"total": len(records), "matched": len(records)},
    }


if __name__ == "__main__":
    print(json.dumps(run_cases(), indent=2, sort_keys=True))
