import json
import re

import pytest

from nli_provider import LoadedNLIModelDescriptor
from nli_qualification import NLIQualificationRunner, VersionedNLIQualificationManifest, build_nli_qualification_runner, load_nli_qualification_manifest
from run_nli_qualification import write_qualification_evidence


MANIFEST_PATH = "nli_deberta_qualification.json"


class _FakeProvider:
    def __init__(self, model_id, revision):
        self.loaded_model_descriptor = LoadedNLIModelDescriptor(model_id, revision, "huggingface-sequence-classification", ("contradiction", "entailment", "neutral"))

    def predict(self, premise, hypothesis):
        if "not " in hypothesis:
            return {"contradiction": 0.98, "entailment": 0.01, "neutral": 0.01}
        if hypothesis == premise:
            return {"contradiction": 0.01, "entailment": 0.98, "neutral": 0.01}
        return {"contradiction": 0.01, "entailment": 0.01, "neutral": 0.98}


def test_repository_manifest_is_valid_and_pinned():
    manifest = load_nli_qualification_manifest(MANIFEST_PATH)
    assert manifest.payload()["model"]["model_revision"] == "fa2804872c3b4bd748f38c0185cc85775361e735"
    with open(MANIFEST_PATH, encoding="utf-8") as manifest_file:
        assert manifest.payload_sha256 == json.load(manifest_file)["payload_sha256"]


def test_manifest_qualifies_with_matching_observed_provider():
    manifest = load_nli_qualification_manifest(MANIFEST_PATH)
    result = manifest.qualify(NLIQualificationRunner(manifest, _FakeProvider))
    assert result["qualification_record"]["qualification"]["status"] == "PASS"
    assert len(result["qualification_record"]["case_results"]) == 9
    assert all(case["expected_label"] == case["predicted_label"] for case in result["qualification_record"]["case_results"])


def test_manifest_rejects_forged_case_evidence():
    manifest = load_nli_qualification_manifest(MANIFEST_PATH)
    runner = NLIQualificationRunner(manifest, _FakeProvider)
    record = runner.run()
    record["case_results"][0]["observed_status"] = "FAIL"
    with pytest.raises(ValueError, match="decision mismatch|case"):
        manifest._validate_record(record)


def test_evidence_writer_binds_source_runtime_and_digest(tmp_path, monkeypatch):
    manifest = load_nli_qualification_manifest(MANIFEST_PATH)
    qualification = manifest.qualify(NLIQualificationRunner(manifest, _FakeProvider))
    monkeypatch.setattr("run_nli_qualification.qualify_nli_manifest", lambda _path: qualification)
    monkeypatch.setattr("run_nli_qualification.importlib.metadata.version", lambda package: f"pinned-{package}")
    output = tmp_path / "evidence.json"
    evidence = write_qualification_evidence(MANIFEST_PATH, "a" * 40, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document == evidence
    assert document["payload"]["source_revision"] == "a" * 40
    assert document["payload"]["runtime"]["transformers"] == "pinned-transformers"
    assert re.fullmatch(r"[0-9a-f]{64}", document["payload_sha256"])


def test_evidence_writer_rejects_unpinned_source_revision(tmp_path):
    with pytest.raises(ValueError, match="source_revision"):
        write_qualification_evidence(MANIFEST_PATH, "main", tmp_path / "evidence.json")


def test_manifest_rejects_payload_tampering():
    with open(MANIFEST_PATH, encoding="utf-8") as manifest_file:
        document = json.load(manifest_file)
    document["payload"]["policy"]["pass_threshold"] = 0.1
    with pytest.raises(ValueError, match="payload digest mismatch"):
        VersionedNLIQualificationManifest(document)


def test_runner_rejects_forged_loaded_revision():
    class ForgedProvider(_FakeProvider):
        def __init__(self, model_id, revision):
            super().__init__(model_id, "f" * 40)

    manifest = load_nli_qualification_manifest(MANIFEST_PATH)
    with pytest.raises(ValueError, match="identity does not match"):
        NLIQualificationRunner(manifest, ForgedProvider).run()


def test_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"payload": {}, "payload": {}, "payload_sha256": "x"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_nli_qualification_manifest(path)


@pytest.mark.integration
@pytest.mark.nli_integration
def test_pinned_real_nli_model_qualifies():
    manifest = load_nli_qualification_manifest(MANIFEST_PATH)
    result = manifest.qualify(build_nli_qualification_runner(manifest))
    assert result["qualification_record"]["qualification"]["status"] == "PASS"
