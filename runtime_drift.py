import hashlib
import json
import re
from pathlib import Path

from supply_chain_policy import reject_duplicate_json_keys


SUPPORTED_POLICY_VERSION = "rag-v29-runtime-drift-v1"
DECISION_CURRENT = "CURRENT"
DECISION_REQUALIFY = "REQUALIFICATION_REQUIRED"
POLICY_FIELDS = {"schema_version", "policy_version", "triggers"}
TRIGGER_FIELDS = {"id", "path"}
RELEASE_FIELDS = {
    "artifact_schema_version", "release_policy_version", "release_manifest_sha256",
    "source_revision", "ci_supply_chain", "dependency_lock", "runtime", "components",
    "qualification",
}
RUNTIME_FIELDS = {
    "python", "sentence-transformers", "torch", "transformers", "tokenizers",
    "safetensors", "huggingface-hub",
}
EXPECTED_TRIGGERS = (
    ("source-revision-changed", "source_revision"),
    ("release-policy-changed", "release_policy_version"),
    ("release-manifest-changed", "release_manifest_sha256"),
    ("workflow-policy-changed", "ci_supply_chain.manifest_sha256"),
    ("runner-os-changed", "ci_supply_chain.runner_image_os"),
    ("runner-image-changed", "ci_supply_chain.runner_image_version"),
    ("dependency-lock-changed", "dependency_lock.sha256"),
    ("installed-inventory-changed", "dependency_lock.installed_packages_sha256"),
    ("python-runtime-changed", "runtime.python"),
    ("sentence-transformers-changed", "runtime.sentence-transformers"),
    ("torch-changed", "runtime.torch"),
    ("transformers-changed", "runtime.transformers"),
    ("tokenizers-changed", "runtime.tokenizers"),
    ("safetensors-changed", "runtime.safetensors"),
    ("huggingface-hub-changed", "runtime.huggingface-hub"),
    ("embedding-manifest-changed", "components.embedding-minilm.manifest_sha256"),
    ("nli-manifest-changed", "components.nli-deberta-v2.manifest_sha256"),
    ("qualification-status-changed", "qualification.status"),
)


def canonical_payload(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def payload_sha256(value):
    return hashlib.sha256(canonical_payload(value).encode()).hexdigest()


def _load_envelope(path, label):
    document = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_json_keys)
    if not isinstance(document, dict) or set(document) != {"payload", "payload_sha256"}:
        raise ValueError(f"{label} envelope is invalid")
    if not isinstance(document["payload"], dict):
        raise ValueError(f"{label} payload is invalid")
    digest = document["payload_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{label} digest is invalid")
    if payload_sha256(document["payload"]) != digest:
        raise ValueError(f"{label} payload digest mismatch")
    return document


def _is_digest(value, length=64):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _validate_release_evidence(document, label):
    payload = document["payload"]
    if set(payload) != RELEASE_FIELDS or payload["artifact_schema_version"] != 1:
        raise ValueError(f"{label} payload schema is invalid")
    if not _is_digest(payload["source_revision"], 40):
        raise ValueError(f"{label} source revision is invalid")
    if not isinstance(payload["release_policy_version"], str) or not payload["release_policy_version"]:
        raise ValueError(f"{label} release policy version is invalid")
    if not _is_digest(payload["release_manifest_sha256"]):
        raise ValueError(f"{label} release manifest digest is invalid")

    supply = payload["ci_supply_chain"]
    if not isinstance(supply, dict) or set(supply) != {
        "manifest_path", "manifest_sha256", "runner_image_os", "runner_image_version"
    }:
        raise ValueError(f"{label} CI supply-chain evidence is invalid")
    if not isinstance(supply["manifest_path"], str) or not supply["manifest_path"] or not _is_digest(supply["manifest_sha256"]):
        raise ValueError(f"{label} CI supply-chain identity is invalid")
    for field in ("runner_image_os", "runner_image_version"):
        if not isinstance(supply[field], str) or not supply[field] or supply[field] == "unavailable":
            raise ValueError(f"{label} runner identity is unavailable")

    dependency = payload["dependency_lock"]
    if not isinstance(dependency, dict) or set(dependency) != {
        "path", "sha256", "installed_packages", "installed_packages_sha256"
    }:
        raise ValueError(f"{label} dependency evidence is invalid")
    packages = dependency["installed_packages"]
    if (
        dependency["path"] != "requirements-integration.lock"
        or not _is_digest(dependency["sha256"])
        or not isinstance(packages, dict)
        or not packages
        or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in packages.items())
        or not _is_digest(dependency["installed_packages_sha256"])
        or payload_sha256(packages) != dependency["installed_packages_sha256"]
    ):
        raise ValueError(f"{label} dependency identity is invalid")

    runtime = payload["runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != RUNTIME_FIELDS
        or any(not isinstance(value, str) or not value for value in runtime.values())
    ):
        raise ValueError(f"{label} runtime evidence is invalid")
    for package in RUNTIME_FIELDS - {"python"}:
        if packages.get(package) != runtime[package]:
            raise ValueError(f"{label} runtime does not match installed inventory: {package}")

    components = payload["components"]
    if not isinstance(components, dict) or set(components) != {"embedding-minilm", "nli-deberta-v2"}:
        raise ValueError(f"{label} component evidence is invalid")
    component_statuses = {}
    for name, component in components.items():
        if not isinstance(component, dict) or set(component) != {"manifest_sha256", "qualification_record"}:
            raise ValueError(f"{label} component evidence is invalid: {name}")
        if not _is_digest(component["manifest_sha256"]):
            raise ValueError(f"{label} component manifest identity is invalid: {name}")
        record = component["qualification_record"]
        if not isinstance(record, dict) or not isinstance(record.get("qualification"), dict):
            raise ValueError(f"{label} component qualification is invalid: {name}")
        status = record["qualification"].get("status")
        if status not in {"PASS", "FAIL"}:
            raise ValueError(f"{label} component status is invalid: {name}")
        component_statuses[name] = status

    qualification = payload["qualification"]
    if (
        not isinstance(qualification, dict)
        or set(qualification) != {"status", "component_statuses"}
        or qualification["status"] not in {"PASS", "FAIL"}
        or qualification["component_statuses"] != component_statuses
        or (qualification["status"] == "PASS") != all(value == "PASS" for value in component_statuses.values())
    ):
        raise ValueError(f"{label} aggregate qualification is invalid")
    return document


def load_runtime_drift_policy(path):
    document = _load_envelope(path, "runtime drift policy")
    payload = document["payload"]
    if set(payload) != POLICY_FIELDS or payload["schema_version"] != 1:
        raise ValueError("runtime drift policy payload is invalid")
    if payload["policy_version"] != SUPPORTED_POLICY_VERSION:
        raise ValueError("unsupported runtime drift policy version")
    triggers = payload["triggers"]
    if not isinstance(triggers, list) or not triggers or any(not isinstance(item, dict) for item in triggers):
        raise ValueError("runtime drift triggers are invalid")
    if any(set(item) != TRIGGER_FIELDS for item in triggers):
        raise ValueError("runtime drift trigger is invalid")
    ids = [item["id"] for item in triggers]
    paths = [item["path"] for item in triggers]
    if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
        raise ValueError("runtime drift triggers must be unique")
    if any(not isinstance(value, str) or not value for value in ids + paths):
        raise ValueError("runtime drift trigger values are invalid")
    if tuple((item["id"], item["path"]) for item in triggers) != EXPECTED_TRIGGERS:
        raise ValueError("runtime drift policy does not match the implemented trigger set")
    return document


def _resolve(payload, dotted_path):
    value = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"release evidence is missing monitored field: {dotted_path}")
        value = value[part]
    return value


def evaluate_runtime_drift(policy_path, qualified_evidence_path, observed_evidence_path, expected_qualified_sha256):
    if not _is_digest(expected_qualified_sha256):
        raise ValueError("trusted qualified evidence digest is invalid")
    policy = load_runtime_drift_policy(policy_path)
    qualified = _validate_release_evidence(
        _load_envelope(qualified_evidence_path, "qualified release evidence"),
        "qualified release evidence",
    )
    observed = _validate_release_evidence(
        _load_envelope(observed_evidence_path, "observed release evidence"),
        "observed release evidence",
    )
    if qualified["payload_sha256"] != expected_qualified_sha256:
        raise ValueError("qualified release evidence does not match trusted digest")
    qualified_payload = qualified["payload"]
    observed_payload = observed["payload"]
    if _resolve(qualified_payload, "qualification.status") != "PASS":
        raise ValueError("baseline release evidence is not qualified")
    triggered = []
    for trigger in policy["payload"]["triggers"]:
        path = trigger["path"]
        before = _resolve(qualified_payload, path)
        after = _resolve(observed_payload, path)
        if before != after:
            triggered.append({"id": trigger["id"], "path": path, "qualified": before, "observed": after})
    decision = DECISION_REQUALIFY if triggered else DECISION_CURRENT
    payload = {
        "artifact_schema_version": 1,
        "policy_version": policy["payload"]["policy_version"],
        "policy_sha256": policy["payload_sha256"],
        "qualified_evidence_sha256": qualified["payload_sha256"],
        "observed_evidence_sha256": observed["payload_sha256"],
        "decision": decision,
        "triggers": triggered,
    }
    return {"payload": payload, "payload_sha256": payload_sha256(payload)}
