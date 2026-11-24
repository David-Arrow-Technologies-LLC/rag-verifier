"""Non-authorizing freeze evidence preparation. No executable manifest is emitted."""
from collections import Counter
from copy import deepcopy
from itertools import combinations
from difflib import SequenceMatcher

from research.holdout.annotation import digest, adjudicate
from research.holdout.audit_candidate import CATEGORIES, LABEL_TO_STATUS, normalize


def verify_mapping(candidate, packet, private):
    if candidate["payload_sha256"] != "PENDING_FREEZE_AFTER_REVIEW":
        raise ValueError("candidate must remain unfrozen")
    if set(packet) != {"candidate_digest", "items"} or set(private) != {
        "candidate_digest", "packet_digest", "mapping"
    }:
        raise ValueError("invalid packet or mapping schema")
    if packet["candidate_digest"] != digest(candidate) or private["candidate_digest"] != digest(candidate) or private["packet_digest"] != digest(packet):
        raise ValueError("candidate or packet binding mismatch")
    cases = candidate["payload"]["cases"]
    by_id = {c["case_id"]: c for c in cases}
    items = packet["items"]
    mapping = private["mapping"]
    if len(cases) != 90 or len(by_id) != 90 or len(items) != 90 or len(mapping) != 90:
        raise ValueError("invalid mapping coverage")
    if len({r["item_id"] for r in items}) != 90 or set(mapping) != {r["item_id"] for r in items} or set(mapping.values()) != set(by_id):
        raise ValueError("mapping must be bijective")
    for row in items:
        case = by_id[mapping[row["item_id"]]]
        if set(row) != {"item_id", "premise", "hypothesis"} or any(row[k] != case[k] for k in ("premise", "hypothesis")):
            raise ValueError("mapped text mismatch")


def leakage_report(candidate, baseline):
    """Deterministic lexical screening, not proof of semantic independence."""
    def rows(document):
        return sorted((c["case_id"], normalize(c["premise"]), normalize(c["hypothesis"]))
                      for c in document["payload"]["cases"])
    current, reference = rows(candidate), rows(baseline)
    findings = []
    def compare(a, b, scope):
        score = SequenceMatcher(None, " || ".join(a[1:]), " || ".join(b[1:]), autojunk=False).ratio()
        if a[1:] == b[1:] or score >= .92:
            findings.append({"scope": scope, "first": a[0], "second": b[0],
                             "exact": a[1:] == b[1:], "similarity": round(score, 6)})
    for a, b in combinations(current, 2):
        compare(a, b, "within")
    for a in current:
        for b in reference:
            compare(a, b, "baseline")
    return {"candidate_digest": digest(candidate), "baseline_digest": digest(baseline),
            "threshold": .92, "findings": findings}


def prepare_evidence(candidate, baseline, packet, private, first, second, resolution):
    verify_mapping(candidate, packet, private)
    proposal = adjudicate(packet, first, second, resolution)
    payload = deepcopy(candidate["payload"])
    by_id = {c["case_id"]: c for c in payload["cases"]}
    for token, label in proposal["proposed_labels"].items():
        case = by_id[private["mapping"][token]]
        case["expected_label"] = label
        case["expected_status"] = LABEL_TO_STATUS[label]
    counts = Counter((c["category"], c["expected_label"]) for c in payload["cases"])
    expected = {(cat, label): 3 for cat in CATEGORIES for label in LABEL_TO_STATUS}
    if counts != expected:
        raise ValueError("adjudicated category/label balance mismatch; revise candidate before freeze")
    report = leakage_report(candidate, baseline)
    return {"schema_version": 1, "candidate_digest": digest(candidate),
            "mapping_digest": digest(private), "annotation_proposal_digest": digest(proposal),
            "leakage_report": report, "proposed_payload_digest": digest(payload),
            "critical_case_ids": sorted(c["case_id"] for c in payload["cases"] if c["critical"]),
            "required_external_evidence": ["authenticated reviewer and adjudicator credentials",
                "independence and conflict-of-interest declarations", "criticality review",
                "semantic leakage adjudication", "scientific approval", "explicit freeze authorization"],
            "freeze_authorized": False, "inference_authorized": False,
            "independent_authorship_established": False}


def authorize_freeze(*args, **kwargs):
    """No trust root or approved authority exists in this slice."""
    raise PermissionError("freeze disabled: authenticated authority not configured")
