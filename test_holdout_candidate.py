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


def test_holdout_rejects_modified_model_revision(tmp_path):
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    candidate["payload"]["model"]["model_revision"] = "a" * 40
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned model"):
        audit(candidate_path=path)


def test_holdout_rejects_changed_release_threshold(tmp_path):
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    candidate["payload"]["policy"]["pass_threshold"] = 0.7
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="release thresholds"):
        audit(candidate_path=path)
