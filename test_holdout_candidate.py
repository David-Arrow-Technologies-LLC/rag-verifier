import json
from pathlib import Path
import pytest

from research.holdout.audit_candidate import audit, CANDIDATE, BASELINE


def test_holdout_candidate_static_checks():
    report = audit()
    assert report["candidate_cases"] == 90
    assert report["per_label"] == {"entailment": 30, "contradiction": 30, "neutral": 30}


def test_holdout_rejects_reused_baseline_pair(tmp_path):
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    candidate["payload"]["cases"][0]["premise"] = baseline["payload"]["cases"][0]["premise"]
    candidate["payload"]["cases"][0]["hypothesis"] = baseline["payload"]["cases"][0]["hypothesis"]
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        audit(candidate_path=path)


def test_holdout_rejects_label_imbalance(tmp_path):
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    candidate["payload"]["cases"][0]["expected_label"] = "neutral"
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="unbalanced"):
        audit(candidate_path=path)
