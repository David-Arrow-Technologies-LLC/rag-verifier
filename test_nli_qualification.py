import json
import re

import pytest

from nli_provider import LoadedNLIModelDescriptor
from nli_qualification import NLIQualificationRunner, VersionedNLIQualificationManifest, build_nli_qualification_runner, load_nli_qualification_manifest
from run_nli_qualification import main, resolve_source_revision, write_qualification_evidence


MANIFEST_PATH = "nli_deberta_qualification.json"
MANIFEST_V2_PATH = "nli_deberta_qualification_v2.json"


class _FakeProvider:
    def __init__(self, model_id, revision):
        self.loaded_model_descriptor = LoadedNLIModelDescriptor(model_id, revision, "huggingface-sequence-classification", ("contradiction", "entailment", "neutral"))

    def predict(self, premise, hypothesis):
        if "not " in hypothesis:
            return {"contradiction": 0.98, "entailment": 0.01, "neutral": 0.01}
        if hypothesis == premise:
            return {"contradiction": 0.01, "entailment": 0.98, "neutral": 0.01}
        return {"contradiction": 0.01, "entailment": 0.01, "neutral": 0.98}


class _ManifestProvider:
    def __init__(self, model_id, revision):
        self.loaded_model_descriptor = LoadedNLIModelDescriptor(model_id, revision, "huggingface-sequence-classification", ("contradiction", "entailment", "neutral"))
        manifest = load_nli_qualification_manifest(MANIFEST_V2_PATH).payload()
        self.labels = {(case["premise"], case["hypothesis"]): case["expected_label"] for case in manifest["cases"]}

    def predict(self, premise, hypothesis):
        label = self.labels[(premise, hypothesis)]
        return {candidate: 0.98 if candidate == label else 0.01 for candidate in ("contradiction", "entailment", "neutral")}


def test_repository_manifest_is_valid_and_pinned():
    manifest = load_nli_qualification_manifest(MANIFEST_PATH)
    assert manifest.payload()["model"]["model_revision"] == "fa2804872c3b4bd748f38c0185cc85775361e735"
    with open(MANIFEST_PATH, encoding="utf-8") as manifest_file:
        assert manifest.payload_sha256 == json.load(manifest_file)["payload_sha256"]


def test_v2_manifest_is_balanced_versioned_and_critical():
    manifest = load_nli_qualification_manifest(MANIFEST_V2_PATH)
    payload = manifest.payload()
    assert payload["schema_version"] == 2
    assert payload["benchmark_version"] == "synthetic-medical-nli-v2"
    assert len(payload["cases"]) == 30
    assert {label: sum(case["expected_label"] == label for case in payload["cases"]) for label in manifest.LABELS} == {
        "contradiction": 10,
        "entailment": 10,
        "neutral": 10,
    }
    assert any(case["critical"] for case in payload["cases"])


def test_v2_runner_emits_stratified_metrics_and_passes_critical_cases():
    manifest = load_nli_qualification_manifest(MANIFEST_V2_PATH)
    result = manifest.qualify(NLIQualificationRunner(manifest, _ManifestProvider))
    record = result["qualification_record"]
    assert record["metrics"]["per_label_accuracy"] == {"contradiction": 1.0, "entailment": 1.0, "neutral": 1.0}
    assert record["metrics"]["critical_case_accuracy"] == 1.0
    assert record["qualification"]["status"] == "PASS"


def test_v2_rejects_qualification_that_hides_critical_failure():
    manifest = load_nli_qualification_manifest(MANIFEST_V2_PATH)
    record = NLIQualificationRunner(manifest, _ManifestProvider).run()
    critical = next(case for case in record["case_results"] if case["critical"])
    critical["scores"] = {"contradiction": 0.01, "entailment": 0.01, "neutral": 0.98}
    critical["predicted_label"] = "neutral"
    critical["observed_status"] = "REVIEW"
    with pytest.raises(ValueError, match="metrics do not match case results"):
        manifest._validate_record(record)


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


def test_manifest_rejects_metrics_that_contradict_case_evidence():
    manifest = load_nli_qualification_manifest(MANIFEST_PATH)
    record = NLIQualificationRunner(manifest, _FakeProvider).run()
    record["case_results"][0]["scores"] = {
        "contradiction": 0.01,
        "entailment": 0.01,
        "neutral": 0.98,
    }
    record["case_results"][0]["predicted_label"] = "neutral"
    record["case_results"][0]["observed_status"] = "REVIEW"
    with pytest.raises(ValueError, match="metrics do not match case results"):
        manifest._validate_record(record)


def test_evidence_writer_binds_source_runtime_and_digest(tmp_path, monkeypatch):
    manifest = load_nli_qualification_manifest(MANIFEST_PATH)
    qualification = manifest.qualify(NLIQualificationRunner(manifest, _FakeProvider))
    monkeypatch.setattr("run_nli_qualification.qualify_nli_manifest", lambda _path: qualification)
    monkeypatch.setattr("run_nli_qualification.importlib.metadata.version", lambda package: f"pinned-{package}")
    monkeypatch.setattr("run_nli_qualification.resolve_source_revision", lambda _path: "a" * 40)
    output = tmp_path / "evidence.json"
    evidence = write_qualification_evidence(MANIFEST_PATH, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document == evidence
    assert document["payload"]["source_revision"] == "a" * 40
    assert document["payload"]["runtime"]["transformers"] == "pinned-transformers"
    assert re.fullmatch(r"[0-9a-f]{64}", document["payload_sha256"])


def test_source_revision_rejects_dirty_worktree(monkeypatch):
    class Result:
        stdout = "?? forged.py\n"

    monkeypatch.setattr("run_nli_qualification.subprocess.run", lambda *args, **kwargs: Result())
    with pytest.raises(ValueError, match="clean"):
        resolve_source_revision(".")


def test_source_revision_is_derived_from_clean_checkout(monkeypatch):
    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    results = iter((Result(""), Result("b" * 40 + "\n")))
    monkeypatch.setattr("run_nli_qualification.subprocess.run", lambda *args, **kwargs: next(results))
    assert resolve_source_revision(".") == "b" * 40


def test_cli_fails_after_writing_failed_qualification(tmp_path, monkeypatch):
    evidence = {"payload": {"qualification_record": {"qualification": {"status": "FAIL"}}}, "payload_sha256": "a" * 64}
    monkeypatch.setattr("run_nli_qualification.write_qualification_evidence", lambda *args: evidence)
    monkeypatch.setattr("sys.argv", ["run_nli_qualification.py", "--manifest", MANIFEST_PATH, "--output", str(tmp_path / "evidence.json")])
    assert main() == 1


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
    manifest = load_nli_qualification_manifest(MANIFEST_V2_PATH)
    result = manifest.qualify(build_nli_qualification_runner(manifest))
    assert result["qualification_record"]["qualification"]["status"] == "PASS"
