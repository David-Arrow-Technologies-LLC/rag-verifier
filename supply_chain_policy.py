import hashlib
import json
import re
from pathlib import Path


ACTION_SHA = re.compile(r"^[0-9a-f]{40}$")
PINNED_REQUIREMENT = re.compile(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s;]+(?:\s*;.*)?$")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_workflow_actions(path):
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        match = re.match(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", line)
        if not match:
            continue
        reference = match.group(1)
        if reference.startswith("./"):
            continue
        if "@" not in reference or not ACTION_SHA.fullmatch(reference.rsplit("@", 1)[1]):
            raise ValueError(f"{path}:{line_number}: action reference must use a full commit SHA")


def validate_requirement_input(path):
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            include = line[3:].strip()
            if include != "requirements-ci.txt":
                raise ValueError(f"{path}:{line_number}: requirement include is not approved")
            continue
        if not PINNED_REQUIREMENT.fullmatch(line):
            raise ValueError(f"{path}:{line_number}: dependency must use an exact version")


def validate_lockfile(path):
    logical = Path(path).read_text(encoding="utf-8").replace("\\\n", " ")
    packages = 0
    for line_number, raw in enumerate(logical.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        requirement, separator, hashes = line.partition(" --hash=")
        if not PINNED_REQUIREMENT.fullmatch(requirement.strip()):
            raise ValueError(f"{path}:{line_number}: lock entry must use an exact version")
        hash_tokens = f"--hash={hashes}".split() if separator else []
        if not hash_tokens or any(
            re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", token) is None
            for token in hash_tokens
        ):
            raise ValueError(f"{path}:{line_number}: lock entry must include SHA-256 hashes")
        packages += 1
    if packages == 0:
        raise ValueError(f"{path}: lockfile must contain at least one package")
    return packages


def validate_repository(root="."):
    root = Path(root)
    workflow_root = root / ".github" / "workflows"
    workflow_paths = sorted(set(workflow_root.glob("*.yml")) | set(workflow_root.glob("*.yaml")))
    if not workflow_paths:
        raise ValueError("repository must contain workflow files")
    for path in workflow_paths:
        validate_workflow_actions(path)
    for name in ("requirements-ci.txt", "requirements-integration.txt"):
        validate_requirement_input(root / name)
    locks = {}
    for name in ("requirements-ci.lock", "requirements-integration.lock"):
        path = root / name
        locks[name] = {"packages": validate_lockfile(path), "sha256": file_sha256(path)}
    return {"workflows": len(workflow_paths), "locks": locks}


if __name__ == "__main__":
    print(json.dumps(validate_repository(), sort_keys=True))
