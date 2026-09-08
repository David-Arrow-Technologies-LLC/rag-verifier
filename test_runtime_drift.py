import copy
import json

import pytest

from runtime_drift import DECISION_CURRENT, DECISION_REQUALIFY, evaluate_runtime_drift, load_runtime_drift_policy, payload_sha256


POLICY = "runtime_drift_policy.json"


def _evidence():
    payload = {
        "source_revision": "a" * 40,
        "release_policy_version": "rag-v28-unified-release-v1",
        "release_manifest_sha256": "b" * 64,
        "ci_supply_chain": {
            "manifest_sha256": "c" * 64,
            "runner_image_os": "ubuntu24",
            "runner_image_version": "20260901.1",
        },
        "dependency_lock": {"sha256": "d" * 64, "installed_packages_sha256": "e" * 64},
        "runtime": {
            "python": "3.12.13",
            "sentence-transformers": "6.0.0",
            "torch": "2.14.0",
            "transformers": "5.16.1",
            "tokenizers": "0.23.2",
            "safetensors": "0.8.0",
            "huggingface-hub": "1.8.0",
        },
        "components": {
            "embedding-minilm": {"manifest_sha256": "f" * 64},
            "nli-deberta-v2": {"manifest_sha256": "0" * 64},
        },
        "qualification": {"status": "PASS"},
    }
    return {"payload": payload, "payload_sha256": payload_sha256(payload)}


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


def test_repository_policy_is_valid_and_digest_bound():
    assert load_runtime_drift_policy(POLICY)["payload"]["policy_version"] == "rag-v29-runtime-drift-v1"


def test_rehashed_policy_cannot_remove_a_trigger(tmp_path):
    document = json.loads(open(POLICY, encoding="utf-8").read())
    document["payload"]["triggers"] = document["payload"]["triggers"][1:]
    document["payload_sha256"] = payload_sha256(document["payload"])
    path = tmp_path / "policy.json"
    _write(path, document)
    with pytest.raises(ValueError, match="implemented trigger set"):
        load_runtime_drift_policy(path)


def test_identical_runtime_remains_current(tmp_path):
    evidence = _evidence()
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, evidence)
    _write(observed, evidence)
    result = evaluate_runtime_drift(POLICY, baseline, observed)
    assert result["payload"]["decision"] == DECISION_CURRENT
    assert result["payload"]["triggers"] == []


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (("source_revision",), "source-revision-changed"),
        (("ci_supply_chain", "runner_image_version"), "runner-image-changed"),
        (("dependency_lock", "installed_packages_sha256"), "installed-inventory-changed"),
        (("runtime", "torch"), "torch-changed"),
        (("components", "nli-deberta-v2", "manifest_sha256"), "nli-manifest-changed"),
    ],
)
def test_governed_drift_requires_requalification(tmp_path, path, expected):
    baseline_document = _evidence()
    observed_document = copy.deepcopy(baseline_document)
    value = observed_document["payload"]
    for part in path[:-1]:
        value = value[part]
    value[path[-1]] = "changed"
    observed_document["payload_sha256"] = payload_sha256(observed_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    result = evaluate_runtime_drift(POLICY, baseline, observed)
    assert result["payload"]["decision"] == DECISION_REQUALIFY
    assert [item["id"] for item in result["payload"]["triggers"]] == [expected]


def test_tampered_evidence_fails_closed(tmp_path):
    baseline_document = _evidence()
    observed_document = copy.deepcopy(baseline_document)
    observed_document["payload"]["runtime"]["python"] = "forged"
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    with pytest.raises(ValueError, match="payload digest mismatch"):
        evaluate_runtime_drift(POLICY, baseline, observed)


def test_non_pass_baseline_fails_closed(tmp_path):
    baseline_document = _evidence()
    baseline_document["payload"]["qualification"]["status"] = "FAIL"
    baseline_document["payload_sha256"] = payload_sha256(baseline_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, _evidence())
    with pytest.raises(ValueError, match="not qualified"):
        evaluate_runtime_drift(POLICY, baseline, observed)


def test_missing_monitored_field_fails_closed(tmp_path):
    baseline_document = _evidence()
    observed_document = copy.deepcopy(baseline_document)
    del observed_document["payload"]["runtime"]["torch"]
    observed_document["payload_sha256"] = payload_sha256(observed_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    with pytest.raises(ValueError, match="missing monitored field"):
        evaluate_runtime_drift(POLICY, baseline, observed)
