"""Offline annotation preparation only. Never grants freeze or execution authority."""
import hashlib
import json
import secrets
from collections import Counter

from research.holdout.audit_candidate import LABEL_TO_STATUS


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def prepare(candidate):
    """Return private coordinator mapping and a separate blinded packet.
    Keep mapping and original candidate inaccessible to reviewers.
    """
    if candidate["payload_sha256"] != "PENDING_FREEZE_AFTER_REVIEW":
        raise ValueError("expected unfrozen candidate")
    cases = candidate["payload"]["cases"]
    if len(cases) != 90 or len({c["case_id"] for c in cases}) != 90:
        raise ValueError("invalid candidate coverage")
    binding = digest(candidate)
    rows = []
    mapping = {}
    for case in cases:
        token = secrets.token_hex(16)
        if token in mapping:
            raise ValueError("opaque ID collision")
        mapping[token] = case["case_id"]
        rows.append({"item_id": token, "premise": case["premise"],
                     "hypothesis": case["hypothesis"]})
    secrets.SystemRandom().shuffle(rows)
    packet = {"candidate_digest": binding, "items": rows}
    return packet, {"candidate_digest": binding, "packet_digest": digest(packet),
                    "mapping": mapping}


def validate_record(packet, record):
    if set(record) != {"reviewer_id", "packet_digest", "annotations"}:
        raise ValueError("invalid reviewer record")
    if not isinstance(record["reviewer_id"], str) or not record["reviewer_id"].strip():
        raise ValueError("missing reviewer ID")
    if record["packet_digest"] != digest(packet):
        raise ValueError("packet binding mismatch")
    expected = {row["item_id"] for row in packet["items"]}
    rows = record["annotations"]
    if len(rows) != len(expected) or {row["item_id"] for row in rows} != expected:
        raise ValueError("incomplete or duplicate annotations")
    for row in rows:
        if set(row) != {"item_id", "label", "rationale"}:
            raise ValueError("invalid annotation schema")
        if row["label"] not in LABEL_TO_STATUS:
            raise ValueError("invalid label")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError("missing rationale")
    return {row["item_id"]: row["label"] for row in rows}


def agreement(packet, first, second):
    a, b = validate_record(packet, first), validate_record(packet, second)
    if first["reviewer_id"].strip().casefold() == second["reviewer_id"].strip().casefold():
        raise ValueError("distinct reviewer identifiers required")
    n = len(a)
    if not n:
        raise ValueError("empty packet")
    matches = sum(a[k] == b[k] for k in a)
    ca, cb = Counter(a.values()), Counter(b.values())
    chance_numerator = sum(ca[k] * cb[k] for k in LABEL_TO_STATUS)
    denominator = n*n - chance_numerator
    return {"count": n, "matches": matches, "observed_agreement": matches/n,
            "kappa": (matches*n-chance_numerator)/denominator if denominator else None,
            "disagreements": sorted(k for k in a if a[k] != b[k])}


def adjudicate(packet, first, second, record):
    report = agreement(packet, first, second)
    a, b = validate_record(packet, first), validate_record(packet, second)
    if set(record) != {"adjudicator_id", "packet_digest", "review_digests", "resolutions"}:
        raise ValueError("invalid adjudication record")
    identity = record["adjudicator_id"]
    if not isinstance(identity, str) or not identity.strip() or identity.strip().casefold() in {
        first["reviewer_id"].strip().casefold(), second["reviewer_id"].strip().casefold()
    }:
        raise ValueError("distinct adjudicator identifier required")
    if record["packet_digest"] != digest(packet) or record["review_digests"] != [digest(first), digest(second)]:
        raise ValueError("adjudication binding mismatch")
    rows = record["resolutions"]
    if len(rows) != len(report["disagreements"]) or {r["item_id"] for r in rows} != set(report["disagreements"]):
        raise ValueError("incomplete or duplicate resolutions")
    for row in rows:
        if set(row) != {"item_id", "label", "rationale"} or row["label"] not in LABEL_TO_STATUS:
            raise ValueError("invalid resolution")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError("missing adjudication rationale")
        a[row["item_id"]] = row["label"]
    return {"agreement": report, "proposed_labels": a,
            "proposed_decisions": {k: LABEL_TO_STATUS[v] for k,v in a.items()},
            "candidate_digest": packet["candidate_digest"],
            "packet_digest": digest(packet), "review_digests": [digest(first), digest(second)],
            "adjudication_digest": digest(record),
            "freeze_authorized": False, "inference_authorized": False,
            "limitations": ["Identifiers are not authenticated identities",
                            "Independent authorship is not established",
                            "Criticality, leakage and scientific approval remain pending"]}
