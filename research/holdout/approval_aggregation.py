"""Aggregate fictional attestations only; never grant scientific authority."""
from research.holdout.annotation import digest
from research.holdout.authority_contract import verify_test_attestation
from research.holdout.freeze_evidence import prepare_evidence


def aggregate_test_approvals(candidate, baseline, packet, private, first, second,
                             resolution, approval, attestations, registry, revoked, now):
    evidence = prepare_evidence(candidate, baseline, packet, private, first, second, resolution)
    binding = digest(evidence)
    if set(approval) != {"authorizer_id", "evidence_digest", "decision", "rationale"}:
        raise ValueError("invalid approval schema")
    if approval["evidence_digest"] != binding or approval["decision"] != "TEST_APPROVE":
        raise ValueError("approval evidence or decision mismatch")
    if not isinstance(approval["rationale"], str) or not approval["rationale"].strip():
        raise ValueError("missing approval rationale")
    parties = [
        ("first", first["reviewer_id"], "reviewer", first),
        ("second", second["reviewer_id"], "reviewer", second),
        ("adjudicator", resolution["adjudicator_id"], "adjudicator", resolution),
        ("authorizer", approval["authorizer_id"], "scientific_authorizer", approval),
    ]
    subjects = [p[1] for p in parties]
    if any(not isinstance(s, str) or not s.strip() for s in subjects):
        raise ValueError("invalid subject")
    if len({s.strip().casefold() for s in subjects}) != 4:
        raise ValueError("four distinct subjects required")
    if set(attestations) != {p[0] for p in parties}:
        raise ValueError("exactly four attestation slots required")
    verified = {}
    seen = set()
    for slot, subject, role, record in parties:
        envelope = attestations[slot]
        verify_test_attestation(envelope, registry, revoked, now, subject=subject,
                                role=role, evidence_digest=binding, record_digest=digest(record))
        payload = envelope["payload"]
        identity = (payload["issuer"], payload["attestation_id"])
        if identity in seen:
            raise ValueError("duplicate attestation identity")
        seen.add(identity)
        verified[slot] = {"subject": subject, "role": role,
                          "record_digest": digest(record), "attestation_digest": digest(envelope)}
    return {"schema_version": 1, "environment": "TEST_ONLY",
            "evidence": evidence, "evidence_digest": binding,
            "verified_at": now, "participants": verified,
            "test_package_complete": True, "scientific_approval": False,
            "independent_authorship_established": False,
            "freeze_authorized": False, "inference_authorized": False}
