import argparse
import importlib.metadata
import json
import platform
import re
from pathlib import Path

from nli_qualification import canonical_payload, payload_sha256, qualify_nli_manifest


RUNTIME_PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub")


def build_qualification_evidence(manifest_path, source_revision):
    if not isinstance(source_revision, str) or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ValueError("source_revision must be a 40-character lowercase commit SHA")
    result = qualify_nli_manifest(manifest_path)
    runtime = {"python": platform.python_version()}
    for package in RUNTIME_PACKAGES:
        runtime[package] = importlib.metadata.version(package)
    payload = {
        "artifact_schema_version": 1,
        "source_revision": source_revision,
        "runtime": runtime,
        **result,
    }
    return {"payload": payload, "payload_sha256": payload_sha256(payload)}


def write_qualification_evidence(manifest_path, source_revision, output_path):
    evidence = build_qualification_evidence(manifest_path, source_revision)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_payload(evidence) + "\n", encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description="Run and retain governed real-NLI qualification evidence")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence = write_qualification_evidence(args.manifest, args.source_revision, args.output)
    print(json.dumps({"artifact": args.output, "payload_sha256": evidence["payload_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
