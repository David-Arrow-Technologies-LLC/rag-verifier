import copy
import json

import pytest

from runtime_drift import (
    DECISION_CURRENT,
    DECISION_REQUALIFY,
    canonical_payload,
    evaluate_runtime_drift,
    load_runtime_drift_policy,
    payload_sha256,
)


POLICY = "runtime_drift_policy.json"


def _evidence():
    payload = {
        "artifact_schema_version": 1,
        "source_revision": "a" * 40,
        "release_policy_version": "rag-v28-unified-release-v1",
        "release_manifest_sha256": "b" * 64,
        "ci_supply_chain": {
            "manifest_path": "ci_supply_chain_manifest.json",
            "manifest_sha256": "c" * 64,
            "runner_image_os": "ubuntu24",
            "runner_image_version": "20260901.1",
        },
        "dependency_lock": {
            "path": "requirements-integration.lock",
            "sha256": "d" * 64,
            "installed_packages": {
                "sentence-transformers": "6.0.0",
                "torch": "2.14.0",
                "transformers": "5.16.1",
                "tokenizers": "0.23.2",
                "safetensors": "0.8.0",
                "huggingface-hub": "1.8.0",
            },
            "installed_packages_sha256": "",
        },
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
            "embedding-minilm": {
                "manifest_sha256": "f" * 64,
                "qualification_record": {
                    "model": {"model_id": "embedding/model"},
                    "benchmark": {"benchmark_id": "embedding-benchmark"},
                    "metrics": {"recall_at_k": 1.0},
                    "qualification": {"status": "PASS", "reason": "QUALIFICATION_THRESHOLDS_SATISFIED"},
                },
            },
            "nli-deberta-v2": {
                "manifest_sha256": "0" * 64,
                "qualification_record": {
                    "model": {"model_id": "nli/model"},
                    "benchmark": {"benchmark_id": "nli-benchmark"},
                    "policy": {"pass_threshold": 0.8},
                    "case_results": [{"case_id": "NLI-001"}],
                    "metrics": {"label_accuracy": 1.0},
                    "qualification": {"status": "PASS", "reason": "QUALIFICATION_THRESHOLDS_SATISFIED"},
                },
            },
        },
        "qualification": {
            "status": "PASS",
            "component_statuses": {"embedding-minilm": "PASS", "nli-deberta-v2": "PASS"},
        },
    }
    payload["dependency_lock"]["installed_packages_sha256"] = payload_sha256(
        payload["dependency_lock"]["installed_packages"]
    )
    return {"payload": payload, "payload_sha256": payload_sha256(payload)}


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


def _evaluate(baseline, observed):
    baseline_document = json.loads(baseline.read_text(encoding="utf-8"))
    return evaluate_runtime_drift(POLICY, baseline, observed, baseline_document["payload_sha256"])


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


@pytest.mark.parametrize("invalid_version", [True, 1.0], ids=["boolean", "float"])
def test_rehashed_non_integer_policy_schema_version_fails_closed(tmp_path, invalid_version):
    document = json.loads(open(POLICY, encoding="utf-8").read())
    document["payload"]["schema_version"] = invalid_version
    document["payload_sha256"] = payload_sha256(document["payload"])
    path = tmp_path / "policy.json"
    _write(path, document)
    with pytest.raises(ValueError, match="runtime drift policy payload is invalid"):
        load_runtime_drift_policy(path)


def test_identical_runtime_remains_current(tmp_path):
    evidence = _evidence()
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, evidence)
    _write(observed, evidence)
    result = _evaluate(baseline, observed)
    assert result["payload"]["decision"] == DECISION_CURRENT
    assert result["payload"]["triggers"] == []


@pytest.mark.parametrize(
    ("path", "changed_value", "expected"),
    [
        (("source_revision",), "9" * 40, "source-revision-changed"),
        (("ci_supply_chain", "runner_image_version"), "20260909.1", "runner-image-changed"),
        (("components", "nli-deberta-v2", "manifest_sha256"), "9" * 64, "nli-manifest-changed"),
    ],
)
def test_governed_drift_requires_requalification(tmp_path, path, changed_value, expected):
    baseline_document = _evidence()
    observed_document = copy.deepcopy(baseline_document)
    value = observed_document["payload"]
    for part in path[:-1]:
        value = value[part]
    value[path[-1]] = changed_value
    observed_document["payload_sha256"] = payload_sha256(observed_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    result = _evaluate(baseline, observed)
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
        _evaluate(baseline, observed)


def test_runtime_and_inventory_drift_require_requalification(tmp_path):
    baseline_document = _evidence()
    observed_document = copy.deepcopy(baseline_document)
    observed_document["payload"]["runtime"]["torch"] = "2.15.0"
    observed_document["payload"]["dependency_lock"]["installed_packages"]["torch"] = "2.15.0"
    observed_document["payload"]["dependency_lock"]["installed_packages_sha256"] = payload_sha256(
        observed_document["payload"]["dependency_lock"]["installed_packages"]
    )
    observed_document["payload_sha256"] = payload_sha256(observed_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    result = _evaluate(baseline, observed)
    assert result["payload"]["decision"] == DECISION_REQUALIFY
    assert [item["id"] for item in result["payload"]["triggers"]] == [
        "installed-inventory-changed",
        "torch-changed",
    ]


def test_non_pass_baseline_fails_closed(tmp_path):
    baseline_document = _evidence()
    baseline_document["payload"]["components"]["embedding-minilm"]["qualification_record"]["qualification"]["status"] = "FAIL"
    baseline_document["payload"]["qualification"]["component_statuses"]["embedding-minilm"] = "FAIL"
    baseline_document["payload"]["qualification"]["status"] = "FAIL"
    baseline_document["payload_sha256"] = payload_sha256(baseline_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, _evidence())
    with pytest.raises(ValueError, match="not qualified"):
        _evaluate(baseline, observed)


def test_missing_monitored_field_fails_closed(tmp_path):
    baseline_document = _evidence()
    observed_document = copy.deepcopy(baseline_document)
    del observed_document["payload"]["runtime"]["torch"]
    observed_document["payload_sha256"] = payload_sha256(observed_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    with pytest.raises(ValueError, match="runtime evidence is invalid"):
        _evaluate(baseline, observed)


def test_rehashed_truncated_release_evidence_fails_closed(tmp_path):
    truncated = _evidence()
    del truncated["payload"]["artifact_schema_version"]
    truncated["payload_sha256"] = payload_sha256(truncated["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, truncated)
    _write(observed, truncated)
    with pytest.raises(ValueError, match="payload schema is invalid"):
        evaluate_runtime_drift(POLICY, baseline, observed, truncated["payload_sha256"])


@pytest.mark.parametrize("invalid_version", [True, 1.0], ids=["boolean", "float"])
def test_rehashed_non_integer_release_schema_version_fails_closed(tmp_path, invalid_version):
    baseline_document = _evidence()
    observed_document = copy.deepcopy(baseline_document)
    observed_document["payload"]["artifact_schema_version"] = invalid_version
    observed_document["payload_sha256"] = payload_sha256(observed_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    with pytest.raises(ValueError, match="payload schema is invalid"):
        _evaluate(baseline, observed)


def test_rehashed_alternate_workflow_manifest_path_fails_closed(tmp_path):
    baseline_document = _evidence()
    observed_document = copy.deepcopy(baseline_document)
    observed_document["payload"]["ci_supply_chain"]["manifest_path"] = "alternate.json"
    observed_document["payload_sha256"] = payload_sha256(observed_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    with pytest.raises(ValueError, match="CI supply-chain identity is invalid"):
        _evaluate(baseline, observed)


def test_unavailable_runner_identity_fails_closed(tmp_path):
    evidence = _evidence()
    evidence["payload"]["ci_supply_chain"]["runner_image_version"] = "unavailable"
    evidence["payload_sha256"] = payload_sha256(evidence["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, evidence)
    _write(observed, evidence)
    with pytest.raises(ValueError, match="runner identity is unavailable"):
        evaluate_runtime_drift(POLICY, baseline, observed, evidence["payload_sha256"])


@pytest.mark.parametrize("unknown", ["unknown", "Unavailable", "  ", "n/a"])
def test_unknown_runner_sentinels_fail_closed(tmp_path, unknown):
    evidence = _evidence()
    evidence["payload"]["ci_supply_chain"]["runner_image_os"] = unknown
    evidence["payload_sha256"] = payload_sha256(evidence["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, evidence)
    _write(observed, evidence)
    with pytest.raises(ValueError, match="runner identity is unavailable"):
        evaluate_runtime_drift(POLICY, baseline, observed, evidence["payload_sha256"])


@pytest.mark.parametrize("component", ["embedding-minilm", "nli-deberta-v2"])
def test_truncated_component_record_fails_closed(tmp_path, component):
    baseline_document = _evidence()
    observed_document = copy.deepcopy(baseline_document)
    observed_document["payload"]["components"][component]["qualification_record"] = {
        "qualification": {"status": "PASS"}
    }
    observed_document["payload_sha256"] = payload_sha256(observed_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    with pytest.raises(ValueError, match="component qualification is invalid"):
        _evaluate(baseline, observed)


def test_complete_component_record_change_requires_requalification(tmp_path):
    baseline_document = _evidence()
    observed_document = copy.deepcopy(baseline_document)
    observed_document["payload"]["components"]["nli-deberta-v2"]["qualification_record"]["metrics"]["label_accuracy"] = 0.9
    observed_document["payload_sha256"] = payload_sha256(observed_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    result = _evaluate(baseline, observed)
    assert result["payload"]["decision"] == DECISION_REQUALIFY
    trigger = result["payload"]["triggers"]
    assert [item["id"] for item in trigger] == ["nli-record-changed"]
    assert set(trigger[0]) == {"id", "path", "qualified_sha256", "observed_sha256"}


@pytest.mark.parametrize("qualified_value", [1, 1.0], ids=["integer", "float"])
def test_nested_boolean_number_alias_requires_requalification(tmp_path, qualified_value):
    baseline_document = _evidence()
    baseline_document["payload"]["components"]["embedding-minilm"]["qualification_record"]["metrics"]["recall_at_k"] = qualified_value
    baseline_document["payload_sha256"] = payload_sha256(baseline_document["payload"])
    observed_document = copy.deepcopy(baseline_document)
    observed_document["payload"]["components"]["embedding-minilm"]["qualification_record"]["metrics"]["recall_at_k"] = True
    observed_document["payload_sha256"] = payload_sha256(observed_document["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    result = _evaluate(baseline, observed)
    assert result["payload"]["decision"] == DECISION_REQUALIFY
    assert [item["id"] for item in result["payload"]["triggers"]] == ["embedding-record-changed"]


def test_fabricated_rehashed_baseline_requires_trusted_digest(tmp_path):
    evidence = _evidence()
    fabricated = copy.deepcopy(evidence)
    fabricated["payload"]["source_revision"] = "9" * 40
    fabricated["payload_sha256"] = payload_sha256(fabricated["payload"])
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, fabricated)
    _write(observed, fabricated)
    with pytest.raises(ValueError, match="does not match trusted digest"):
        evaluate_runtime_drift(POLICY, baseline, observed, evidence["payload_sha256"])


@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_canonical_payload_rejects_non_finite_numbers(non_finite):
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_payload({"value": non_finite})


@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_non_finite_release_evidence_fails_closed(tmp_path, non_finite):
    baseline_document = _evidence()
    observed_document = copy.deepcopy(baseline_document)
    observed_document["payload"]["components"]["embedding-minilm"]["qualification_record"]["metrics"][
        "recall_at_k"
    ] = non_finite
    observed_document["payload_sha256"] = "0" * 64
    baseline = tmp_path / "baseline.json"
    observed = tmp_path / "observed.json"
    _write(baseline, baseline_document)
    _write(observed, observed_document)
    with pytest.raises(ValueError, match="non-finite JSON constant is invalid"):
        evaluate_runtime_drift(
            POLICY,
            baseline,
            observed,
            baseline_document["payload_sha256"],
        )


@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_non_finite_policy_fails_closed(tmp_path, non_finite):
    document = json.loads(open(POLICY, encoding="utf-8").read())
    document["payload"]["schema_version"] = non_finite
    document["payload_sha256"] = "0" * 64
    path = tmp_path / "policy.json"
    _write(path, document)
    with pytest.raises(ValueError, match="non-finite JSON constant is invalid"):
        load_runtime_drift_policy(path)
