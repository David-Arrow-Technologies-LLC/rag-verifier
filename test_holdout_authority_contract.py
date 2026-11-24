import pytest
from research.holdout.authority_contract import (
    _tag, verify_test_attestation, verify_scientific_authority,
)

KEY = b"fictional-test-key-not-for-production"
EVIDENCE = "a" * 64
RECORD = "b" * 64


def fixture():
    payload = {"schema_version": 1, "environment": "TEST_ONLY",
               "issuer": "fixture-issuer", "key_id": "test-key",
               "attestation_id": "test-attestation", "subject": "fixture-reviewer",
               "role": "reviewer", "credential_reference": "fictional-credential",
               "independent": True, "conflict_free": True,
               "issued_at": 100, "expires_at": 200,
               "evidence_digest": EVIDENCE, "record_digest": RECORD}
    registry = {("fixture-issuer", "test-key"): {
        "key": KEY, "roles": {"reviewer"}, "subjects": {"fixture-reviewer"}}}
    return {"payload": payload, "tag": _tag(KEY, payload)}, registry


def verify(envelope, registry, revoked=(), now=150):
    return verify_test_attestation(envelope, registry, revoked, now,
        subject="fixture-reviewer", role="reviewer",
        evidence_digest=EVIDENCE, record_digest=RECORD)


def test_valid_fixture_never_confers_authority():
    envelope, registry = fixture()
    result = verify(envelope, registry)
    assert result["test_attestation_valid"]
    assert not result["scientific_approval"]
    assert not result["freeze_authorized"]
    assert not result["inference_authorized"]
    with pytest.raises(PermissionError):
        verify_scientific_authority(result)


@pytest.mark.parametrize("field,value", [
    ("credential_reference", "forged"), ("attestation_id", "altered"),
    ("expires_at", 999), ("issuer", "fixture-other"),
])
def test_unsigned_changes_rejected(field, value):
    envelope, registry = fixture()
    envelope["payload"][field] = value
    with pytest.raises(ValueError):
        verify(envelope, registry)


@pytest.mark.parametrize("field,value", [
    ("environment", "PRODUCTION"), ("independent", False),
    ("conflict_free", 1), ("role", "scientific_authorizer"),
    ("subject", "fixture-other"), ("evidence_digest", "c" * 64),
    ("record_digest", "c" * 64),
])
def test_even_authenticated_invalid_claims_rejected(field, value):
    envelope, registry = fixture()
    envelope["payload"][field] = value
    envelope["tag"] = _tag(KEY, envelope["payload"])
    with pytest.raises(ValueError):
        verify(envelope, registry)


def test_expiry_revocation_and_scope():
    envelope, registry = fixture()
    for now in (99, 200):
        with pytest.raises(ValueError):
            verify(envelope, registry, now=now)
    for revoked in ({("fixture-issuer", "test-key")},
                    {("fixture-issuer", "test-attestation")}):
        with pytest.raises(ValueError):
            verify(envelope, registry, revoked)
    with pytest.raises(ValueError):
        verify(envelope, {})
    registry[("fixture-issuer", "test-key")]["roles"] = {"adjudicator"}
    with pytest.raises(ValueError):
        verify(envelope, registry)
