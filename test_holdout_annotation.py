import json
import pytest
from research.holdout.audit_candidate import CANDIDATE
from research.holdout.annotation import prepare, digest, agreement, adjudicate


def fixture():
    packet, private = prepare(json.loads(CANDIDATE.read_text(encoding="utf-8")))
    def record(identity):
        return {"reviewer_id": identity, "packet_digest": digest(packet),
                "annotations": [{"item_id": row["item_id"], "label": ["entailment", "neutral", "contradiction"][i % 3],
                                 "rationale": "Fictional test only"} for i, row in enumerate(packet["items"])]}
    return packet, private, record("fixture-a"), record("fixture-b")


def test_blinded_packet():
    packet, private, a, b = fixture()
    assert len(packet["items"]) == 90
    assert all(set(row) == {"item_id", "premise", "hypothesis"} for row in packet["items"])
    assert private["packet_digest"] == digest(packet)
    assert agreement(packet, a, b)["kappa"] == 1


def test_reject_incomplete_and_duplicate_reviewer():
    packet, _, a, b = fixture()
    with pytest.raises(ValueError, match="distinct"):
        agreement(packet, a, a)
    b["annotations"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        agreement(packet, a, b)


def test_adjudication_never_authorizes_freeze():
    packet, _, a, b = fixture()
    b["annotations"][0]["label"] = "neutral"
    record = {"adjudicator_id": "fixture-c", "packet_digest": digest(packet),
              "review_digests": [digest(a), digest(b)], "resolutions": []}
    with pytest.raises(ValueError, match="incomplete"):
        adjudicate(packet, a, b, record)
    record["resolutions"] = [{"item_id": b["annotations"][0]["item_id"],
                              "label": "neutral", "rationale": "Fictional test only"}]
    result = adjudicate(packet, a, b, record)
    assert not result["freeze_authorized"]
    assert not result["inference_authorized"]
    assert len(result["proposed_labels"]) == 90


def test_packet_tampering():
    packet, _, a, b = fixture()
    packet["items"][0]["premise"] += " changed"
    with pytest.raises(ValueError, match="binding"):
        agreement(packet, a, b)
