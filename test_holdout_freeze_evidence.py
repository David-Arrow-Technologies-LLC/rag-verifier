import json
from copy import deepcopy

import pytest

from research.holdout.annotation import prepare, digest
from research.holdout.audit_candidate import CANDIDATE, BASELINE
from research.holdout.freeze_evidence import (
    verify_mapping, leakage_report, prepare_evidence, authorize_freeze,
)


def fixture():
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    packet, private = prepare(candidate)
    cases = {c["case_id"]: c for c in candidate["payload"]["cases"]}
    def review(identity):
        return {"reviewer_id": identity, "packet_digest": digest(packet),
                "annotations": [{"item_id": token, "label": cases[case_id]["expected_label"],
                                 "rationale": "Synthetic test fixture only"}
                                for token, case_id in private["mapping"].items()]}
    first, second = review("fixture-a"), review("fixture-b")
    resolution = {"adjudicator_id": "fixture-c", "packet_digest": digest(packet),
                  "review_digests": [digest(first), digest(second)], "resolutions": []}
    return candidate, baseline, packet, private, first, second, resolution


def test_mapping_and_non_authorizing_evidence():
    args = fixture()
    candidate = deepcopy(args[0])
    report = prepare_evidence(*args)
    assert report["candidate_digest"] == digest(candidate)
    assert report["proposed_payload_digest"] == digest(candidate["payload"])
    assert not report["freeze_authorized"]
    assert not report["inference_authorized"]
    assert not report["independent_authorship_established"]
    assert args[0] == candidate
    with pytest.raises(PermissionError):
        authorize_freeze(report, approved=True)


def test_mapping_swap_rejected():
    candidate, _, packet, private, *_ = fixture()
    a, b = list(private["mapping"])[:2]
    private["mapping"][a], private["mapping"][b] = private["mapping"][b], private["mapping"][a]
    with pytest.raises(ValueError, match="mapped text"):
        verify_mapping(candidate, packet, private)


def test_packet_tamper_even_with_recomputed_digest():
    candidate, _, packet, private, *_ = fixture()
    packet["items"][0]["hypothesis"] += " altered"
    private["packet_digest"] = digest(packet)
    with pytest.raises(ValueError, match="mapped text"):
        verify_mapping(candidate, packet, private)


def test_balance_rejected_after_agreed_relabel():
    args = fixture()
    first, second, resolution = args[4:]
    for record in (first, second):
        row = record["annotations"][0]
        row["label"] = "neutral" if row["label"] != "neutral" else "entailment"
    resolution["review_digests"] = [digest(first), digest(second)]
    with pytest.raises(ValueError, match="balance"):
        prepare_evidence(*args)


def test_leakage_report_deterministic_and_within_candidate():
    candidate, baseline, *_ = fixture()
    cases = candidate["payload"]["cases"]
    cases[1]["premise"] = cases[0]["premise"]
    cases[1]["hypothesis"] = cases[0]["hypothesis"]
    report = leakage_report(candidate, baseline)
    assert report == leakage_report(candidate, baseline)
    assert any(f["scope"] == "within" and f["exact"] for f in report["findings"])
