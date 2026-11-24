from copy import deepcopy

import pytest

from research.holdout.annotation import digest
from research.holdout.authority_contract import _tag
from research.holdout.freeze_evidence import prepare_evidence, authorize_freeze
from research.holdout.approval_aggregation import aggregate_test_approvals
from test_holdout_freeze_evidence import fixture

KEY = b"fictional-aggregation-test-key-only"


def package():
    args = list(fixture())
    evidence = digest(prepare_evidence(*args))
    approval = {"authorizer_id": "fixture-d", "evidence_digest": evidence,
                "decision": "TEST_APPROVE", "rationale": "Synthetic fixture"}
    records = [("first", "fixture-a", "reviewer", args[4]),
               ("second", "fixture-b", "reviewer", args[5]),
               ("adjudicator", "fixture-c", "adjudicator", args[6]),
               ("authorizer", "fixture-d", "scientific_authorizer", approval)]
    registry, envelopes = {}, {}
    for slot, subject, role, record in records:
        payload = {"schema_version": 1, "environment": "TEST_ONLY",
                   "issuer": "fixture-issuer", "key_id": slot,
                   "attestation_id": slot, "subject": subject, "role": role,
                   "credential_reference": "fictional", "independent": True,
                   "conflict_free": True, "issued_at": 100, "expires_at": 200,
                   "evidence_digest": evidence, "record_digest": digest(record)}
        registry[("fixture-issuer", slot)] = {
            "key": KEY, "roles": {role}, "subjects": {subject}}
        envelopes[slot] = {"payload": payload, "tag": _tag(KEY, payload)}
    return args + [approval, envelopes, registry, set(), 150]


def test_complete_package_is_non_authorizing():
    args = package()
    before = deepcopy(args)
    result = aggregate_test_approvals(*args)
    assert result["test_package_complete"]
    assert len(result["participants"]) == 4
    assert not result["scientific_approval"]
    assert not result["independent_authorship_established"]
    assert not result["freeze_authorized"]
    assert not result["inference_authorized"]
    assert args == before
    with pytest.raises(PermissionError):
        authorize_freeze(result)


@pytest.mark.parametrize("mutation", ["missing", "extra", "swapped", "forged",
    "expired", "revoked", "record", "baseline", "conflict", "same_subject", "duplicate_id"])
def test_invalid_packages_rejected(mutation):
    args = package()
    approval, envelopes, registry, revoked = args[7:11]
    if mutation == "missing":
        del envelopes["second"]
    elif mutation == "extra":
        envelopes["extra"] = envelopes["first"]
    elif mutation == "swapped":
        envelopes["first"], envelopes["second"] = envelopes["second"], envelopes["first"]
    elif mutation == "forged":
        envelopes["first"]["tag"] = "0" * 64
    elif mutation == "expired":
        args[-1] = 200
    elif mutation == "revoked":
        revoked.add(("fixture-issuer", "first"))
    elif mutation == "record":
        args[4]["annotations"][0]["rationale"] = "Changed"
        args[6]["review_digests"][0] = digest(args[4])
    elif mutation == "baseline":
        args[1]["payload"]["cases"][0]["premise"] += " Changed"
    elif mutation == "conflict":
        approval["decision"] = "TEST_REJECT"
    elif mutation == "same_subject":
        approval["authorizer_id"] = "fixture-a"
    elif mutation == "duplicate_id":
        p = envelopes["second"]["payload"]
        p["attestation_id"] = "first"
        envelopes["second"]["tag"] = _tag(KEY, p)
    with pytest.raises(ValueError):
        aggregate_test_approvals(*args)
