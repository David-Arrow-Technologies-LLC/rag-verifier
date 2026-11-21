from synthetic_verifier_demo import run_cases


def test_synthetic_demo_expected_decisions():
    evidence = run_cases()
    assert evidence["clinical_validation"] is False
    assert evidence["provider"] == "FakeNLIProvider"
    assert evidence["summary"] == {"total": 5, "matched": 5}
    assert [(case["case_id"], case["observed_status"], case["observed_action"])
            for case in evidence["cases"]] == [
        ("supported", "PASS", "RELEASE"),
        ("uncertain", "REVIEW", "HOLD_FOR_REVIEW"),
        ("contradicted", "FAIL", "BLOCK"),
        ("fabricated_citation", "FAIL", "BLOCK"),
        ("unretrieved_citation", "FAIL", "BLOCK"),
    ]


def test_synthetic_demo_deterministic():
    assert run_cases() == run_cases()
