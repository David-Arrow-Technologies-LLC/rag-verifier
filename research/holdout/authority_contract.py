"""Test-only authority contract. No production trust root or freeze capability."""
import hashlib
import hmac
import json
import re

ROLES = {"reviewer", "adjudicator", "scientific_authorizer"}
DOMAIN = b"rag-verifier:holdout:test-authority:v1\x00"


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _tag(key, payload):
    return hmac.new(key, DOMAIN + _canonical(payload), hashlib.sha256).hexdigest()


def verify_test_attestation(envelope, registry, revoked, now, *, subject, role,
                            evidence_digest, record_digest):
    """Validate fictional HMAC attestations against an externally supplied test registry.

    Registry and clock must be trusted by the test harness; never use this as
    production identity verification. A shared secret is not a public signature.
    """
    if type(now) is not int or now < 0 or role not in ROLES:
        raise ValueError("invalid verification context")
    if set(envelope) != {"payload", "tag"}:
        raise ValueError("invalid envelope")
    p = envelope["payload"]
    fields = {"schema_version", "environment", "issuer", "key_id", "attestation_id",
              "subject", "role", "credential_reference", "independent",
              "conflict_free", "issued_at", "expires_at", "evidence_digest", "record_digest"}
    if set(p) != fields or type(p["schema_version"]) is not int or p["schema_version"] != 1 or p["environment"] != "TEST_ONLY":
        raise ValueError("invalid test attestation schema")
    for field in ("issuer", "key_id", "attestation_id", "subject", "credential_reference"):
        if not isinstance(p[field], str) or not p[field].strip() or p[field] != p[field].strip():
            raise ValueError("invalid identity field")
    if not p["subject"].startswith("fixture-") or not p["issuer"].startswith("fixture-"):
        raise ValueError("fictional identities required")
    if p["subject"] != subject or p["role"] != role:
        raise ValueError("subject or role mismatch")
    for field, expected in (("evidence_digest", evidence_digest), ("record_digest", record_digest)):
        if not isinstance(p[field], str) or not re.fullmatch(r"[0-9a-f]{64}", p[field]) or p[field] != expected:
            raise ValueError("evidence binding mismatch")
    if p["independent"] is not True or p["conflict_free"] is not True:
        raise ValueError("independence declarations required")
    if type(p["issued_at"]) is not int or type(p["expires_at"]) is not int or not 0 <= p["issued_at"] <= now < p["expires_at"]:
        raise ValueError("attestation expired or not yet valid")
    identity = (p["issuer"], p["key_id"])
    entry = registry.get(identity)
    if entry is None or identity in revoked or (p["issuer"], p["attestation_id"]) in revoked:
        raise ValueError("unknown or revoked authority")
    if role not in entry["roles"] or subject not in entry["subjects"]:
        raise ValueError("issuer scope mismatch")
    key = entry["key"]
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("invalid test key")
    tag = envelope["tag"]
    if not isinstance(tag, str) or not re.fullmatch(r"[0-9a-f]{64}", tag) or not hmac.compare_digest(tag, _tag(key, p)):
        raise ValueError("invalid authentication tag")
    return {"test_attestation_valid": True, "environment": "TEST_ONLY",
            "subject": subject, "role": role, "evidence_digest": evidence_digest,
            "record_digest": record_digest, "scientific_approval": False,
            "freeze_authorized": False, "inference_authorized": False}


def verify_scientific_authority(*args, **kwargs):
    """Production verification is unavailable until a real trust root is approved."""
    raise PermissionError("scientific authority unavailable: no approved trust root")
