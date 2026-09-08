import hashlib
import json
import re
from pathlib import Path


ACTION_SHA = re.compile(r"^[0-9a-f]{40}$")
PINNED_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==(?P<version>[^\s;]+)$"
)
USES_KEY = re.compile(r"(?:^|[,{\s])[\"']?uses[\"']?\s*:")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_package_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pinned_requirement(line, path, line_number):
    match = PINNED_REQUIREMENT.fullmatch(line)
    if match is None:
        raise ValueError(f"{path}:{line_number}: dependency must use an exact version")
    return normalize_package_name(match.group("name")), match.group("version")


def validate_workflow_actions(path):
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        content = line.split("#", 1)[0]
        match = re.fullmatch(r"\s*(?:-\s*)?uses:\s*([^\s]+)\s*", content)
        if not match:
            if USES_KEY.search(content):
                raise ValueError(f"{path}:{line_number}: action reference must use canonical block syntax")
            continue
        reference = match.group(1)
        if reference.startswith("./"):
            continue
        if "@" not in reference or not ACTION_SHA.fullmatch(reference.rsplit("@", 1)[1]):
            raise ValueError(f"{path}:{line_number}: action reference must use a full commit SHA")


def validate_requirement_input(path):
    requirements = {}
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            include = line[3:].strip()
            if include != "requirements-ci.txt":
                raise ValueError(f"{path}:{line_number}: requirement include is not approved")
            continue
        name, version = parse_pinned_requirement(line, path, line_number)
        if name in requirements:
            raise ValueError(f"{path}:{line_number}: duplicate dependency declaration")
        requirements[name] = version
    return requirements


def read_lock_versions(path):
    logical = Path(path).read_text(encoding="utf-8").replace("\\\n", " ")
    versions = {}
    for line_number, raw in enumerate(logical.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        requirement, separator, hashes = line.partition(" --hash=")
        try:
            name, version = parse_pinned_requirement(requirement.strip(), path, line_number)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: lock entry must use an exact version") from error
        hash_tokens = f"--hash={hashes}".split() if separator else []
        if not hash_tokens or any(
            re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", token) is None
            for token in hash_tokens
        ):
            raise ValueError(f"{path}:{line_number}: lock entry must include SHA-256 hashes")
        if name in versions:
            raise ValueError(f"{path}:{line_number}: duplicate lock entry")
        versions[name] = version
    if not versions:
        raise ValueError(f"{path}: lockfile must contain at least one package")
    return versions


def validate_lockfile(path):
    return len(read_lock_versions(path))


def validate_repository(root="."):
    root = Path(root)
    workflow_root = root / ".github" / "workflows"
    workflow_paths = sorted(set(workflow_root.glob("*.yml")) | set(workflow_root.glob("*.yaml")))
    if not workflow_paths:
        raise ValueError("repository must contain workflow files")
    for path in workflow_paths:
        validate_workflow_actions(path)
    ci_requirements = validate_requirement_input(root / "requirements-ci.txt")
    integration_requirements = {
        **ci_requirements,
        **validate_requirement_input(root / "requirements-integration.txt"),
    }
    locks = {}
    for name, expected in (
        ("requirements-ci.lock", ci_requirements),
        ("requirements-integration.lock", integration_requirements),
    ):
        path = root / name
        versions = read_lock_versions(path)
        for package, version in expected.items():
            if versions.get(package) != version:
                raise ValueError(f"{name}: {package} does not match its requirement input")
        locks[name] = {"packages": len(versions), "sha256": file_sha256(path)}
    return {"workflows": len(workflow_paths), "locks": locks}


if __name__ == "__main__":
    print(json.dumps(validate_repository(), sort_keys=True))
