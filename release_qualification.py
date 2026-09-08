import json
import re
from pathlib import Path

from healthcare_real_model_qualification import qualify_sentence_transformer_manifest
from nli_qualification import canonical_payload, payload_sha256
from run_nli_qualification import build_qualification_evidence
from supply_chain_policy import file_sha256, reject_duplicate_json_keys


RELEASE_MANIFEST_FIELDS = {"schema_version", "release_policy_version", "components", "supply_chain"}
SUPPORTED_RELEASE_POLICY_VERSION = "rag-v28-unified-release-v1"


def load_release_manifest(path, repository_path="."):
    document = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_json_keys)
    if not isinstance(document, dict) or set(document) != {"payload", "payload_sha256"}:
        raise ValueError("release manifest envelope is invalid")
    payload = document["payload"]
    if not isinstance(payload, dict) or set(payload) != RELEASE_MANIFEST_FIELDS or payload["schema_version"] != 1:
        raise ValueError("release manifest payload is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", document["payload_sha256"]):
        raise ValueError("release manifest digest is invalid")
    if payload_sha256(payload) != document["payload_sha256"]:
        raise ValueError("release manifest payload digest mismatch")
    if payload["release_policy_version"] != SUPPORTED_RELEASE_POLICY_VERSION:
        raise ValueError("unsupported release policy version")
    root = Path(repository_path)
    component_entries = payload["components"]
    if not isinstance(component_entries, list) or len(component_entries) != 2 or any(not isinstance(item, dict) for item in component_entries):
        raise ValueError("release components are invalid")
    if len({item.get("name") for item in component_entries}) != len(component_entries):
        raise ValueError("release component names must be unique")
    for component in component_entries:
        if set(component) != {"name", "manifest_path", "file_sha256", "payload_sha256", "required_status"}:
            raise ValueError("release component is invalid")
        if component["required_status"] != "PASS":
            raise ValueError("release component must fail closed on non-PASS status")
        if any(re.fullmatch(r"[0-9a-f]{64}", component[field]) is None for field in ("file_sha256", "payload_sha256")):
            raise ValueError("release component digest is invalid")
        manifest_path = root / component["manifest_path"]
        if file_sha256(manifest_path) != component["file_sha256"]:
            raise ValueError(f"release component file digest mismatch: {component['name']}")
        component_document = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_json_keys)
        if component_document.get("payload_sha256") != component["payload_sha256"]:
            raise ValueError(f"release component payload digest mismatch: {component['name']}")
    supply_chain = payload["supply_chain"]
    if set(supply_chain) != {"workflow_manifest_path", "workflow_manifest_sha256", "lock_sha256"}:
        raise ValueError("release supply-chain declaration is invalid")
    if file_sha256(root / supply_chain["workflow_manifest_path"]) != supply_chain["workflow_manifest_sha256"]:
        raise ValueError("release workflow manifest digest mismatch")
    if set(supply_chain["lock_sha256"]) != {"requirements-ci.lock", "requirements-integration.lock"}:
        raise ValueError("release lock paths are invalid")
    for lock_name, digest in supply_chain["lock_sha256"].items():
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("release lock digest is invalid")
        if file_sha256(root / lock_name) != digest:
            raise ValueError(f"release lock digest mismatch: {lock_name}")
    return document


def build_release_qualification(manifest_path, repository_path="."):
    manifest = load_release_manifest(manifest_path, repository_path)
    components = {item["name"]: item for item in manifest["payload"]["components"]}
    if set(components) != {"embedding-minilm", "nli-deberta-v2"}:
        raise ValueError("release manifest must declare exactly the governed components")
    root = Path(repository_path)
    embedding = qualify_sentence_transformer_manifest(root / components["embedding-minilm"]["manifest_path"])
    nli_evidence = build_qualification_evidence(root / components["nli-deberta-v2"]["manifest_path"], repository_path)
    if embedding["manifest_sha256"] != components["embedding-minilm"]["payload_sha256"]:
        raise ValueError("embedding qualification manifest identity mismatch")
    if nli_evidence["payload"]["manifest_sha256"] != components["nli-deberta-v2"]["payload_sha256"]:
        raise ValueError("NLI qualification manifest identity mismatch")
    records = {
        "embedding-minilm": embedding,
        "nli-deberta-v2": {
            "manifest_sha256": nli_evidence["payload"]["manifest_sha256"],
            "qualification_record": nli_evidence["payload"]["qualification_record"],
        },
    }
    statuses = {
        "embedding-minilm": embedding["qualification_record"]["qualification"]["status"],
        "nli-deberta-v2": nli_evidence["payload"]["qualification_record"]["qualification"]["status"],
    }
    status = "PASS" if all(value == "PASS" for value in statuses.values()) else "FAIL"
    payload = {
        "artifact_schema_version": 1,
        "release_policy_version": manifest["payload"]["release_policy_version"],
        "release_manifest_sha256": manifest["payload_sha256"],
        "source_revision": nli_evidence["payload"]["source_revision"],
        "ci_supply_chain": nli_evidence["payload"]["ci_supply_chain"],
        "dependency_lock": nli_evidence["payload"]["dependency_lock"],
        "runtime": nli_evidence["payload"]["runtime"],
        "components": records,
        "qualification": {"status": status, "component_statuses": statuses},
    }
    return {"payload": payload, "payload_sha256": payload_sha256(payload)}


def write_release_qualification(manifest_path, output_path, repository_path="."):
    evidence = build_release_qualification(manifest_path, repository_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_payload(evidence) + "\n", encoding="utf-8")
    return evidence
