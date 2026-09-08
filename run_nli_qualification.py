import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path

from nli_qualification import canonical_payload, payload_sha256, qualify_nli_manifest
from supply_chain_policy import file_sha256, validate_lockfile


RUNTIME_PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub")


def resolve_source_revision(repository_path):
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("qualification source worktree must be clean")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("checked-out source revision is not a lowercase commit SHA")
    return revision


def build_qualification_evidence(manifest_path, repository_path="."):
    source_revision = resolve_source_revision(repository_path)
    lock_path = Path(repository_path) / "requirements-integration.lock"
    validate_lockfile(lock_path)
    result = qualify_nli_manifest(manifest_path)
    runtime = {"python": platform.python_version()}
    for package in RUNTIME_PACKAGES:
        runtime[package] = importlib.metadata.version(package)
    payload = {
        "artifact_schema_version": 2,
        "source_revision": source_revision,
        "dependency_lock": {
            "path": "requirements-integration.lock",
            "sha256": file_sha256(lock_path),
        },
        "runtime": runtime,
        **result,
    }
    return {"payload": payload, "payload_sha256": payload_sha256(payload)}


def write_qualification_evidence(manifest_path, output_path, repository_path="."):
    evidence = build_qualification_evidence(manifest_path, repository_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_payload(evidence) + "\n", encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description="Run and retain governed real-NLI qualification evidence")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", default=".")
    args = parser.parse_args()
    evidence = write_qualification_evidence(args.manifest, args.output, args.repository)
    print(json.dumps({"artifact": args.output, "payload_sha256": evidence["payload_sha256"]}, sort_keys=True))
    if evidence["payload"]["qualification_record"]["qualification"]["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
