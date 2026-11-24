import json
from pathlib import Path
import pytest
from nli_qualification import NLIQualificationRunner, load_nli_qualification_manifest, payload_sha256
from test_nli_qualification import _ManifestProvider
from research import scientific_export as exporter


def fixture(monkeypatch):
    # Fictional test only: substitute the pinned evidence digest in process.
    path = Path("nli_deberta_qualification_v2.json")
    manifest = load_nli_qualification_manifest(path)
    p = dict(artifact_schema_version=2, source_revision=exporter.SOURCE,
             runtime={"fixture": True}, dependency_lock={}, ci_supply_chain={},
             **manifest.qualify(NLIQualificationRunner(manifest, _ManifestProvider)))
    raw = exporter.encode(dict(payload=p, payload_sha256=payload_sha256(p)))
    monkeypatch.setattr(exporter, "EVIDENCE_SHA", exporter.sha(raw))
    return path.read_bytes(), raw


def test_export_exact_bytes_and_counts(monkeypatch):
    dataset, evidence = fixture(monkeypatch)
    raw, artifacts = exporter.export_historical(dataset, evidence)
    m = json.loads(raw)
    assert exporter.export_historical(dataset, evidence) == (raw, artifacts)
    assert artifacts["dataset"] == dataset
    assert artifacts["case_results"] == evidence
    assert m["metrics"] == dict(total=30, expected_non_pass=20, false_pass=0,
                                label_correct=30, decision_correct=30,
                                critical_total=10, critical_correct=10)
    assert m["study"] == "DA-RAG-01A"
    assert all(not m[k] for k in ("scientific_approval", "freeze_authorized",
                                  "inference_authorized", "clinical_use_authorized", "translation_authorized"))
    for ref in m["artifacts"]:
        assert ref["sha256"] == exporter.sha(artifacts[ref["kind"]])
        assert ref["byte_length"] == len(artifacts[ref["kind"]])


def test_reject_changed_inputs(monkeypatch):
    dataset, evidence = fixture(monkeypatch)
    with pytest.raises(ValueError):
        exporter.export_historical(dataset, evidence+b" ")
    bad = json.loads(dataset)
    bad["payload"]["policy"]["pass_threshold"] = 0.1
    bad["payload_sha256"] = payload_sha256(bad["payload"])
    with pytest.raises(ValueError):
        exporter.export_historical(exporter.encode(bad), evidence)


def test_reject_rehashed_wrong_record(monkeypatch):
    dataset, evidence = fixture(monkeypatch)
    bad = json.loads(evidence)
    bad["payload"]["qualification_record"]["metrics"]["label_accuracy"] = 0.5
    bad["payload_sha256"] = payload_sha256(bad["payload"])
    raw = exporter.encode(bad)
    monkeypatch.setattr(exporter, "EVIDENCE_SHA", exporter.sha(raw))
    with pytest.raises(ValueError):
        exporter.export_historical(dataset, raw)
