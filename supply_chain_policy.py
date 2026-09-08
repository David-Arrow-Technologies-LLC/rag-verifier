import hashlib
import json
import re
from pathlib import Path


ACTION_SHA = re.compile(r"^[0-9a-f]{40}$")
PINNED_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?=="
    r"(?P<version>[0-9]+(?:[A-Za-z0-9.!+_-]*[A-Za-z0-9])?)$"
)
USES_KEY = re.compile(r"(?:^|[,{\s])[\"']?uses[\"']?\s*:")
EXACT_HEAD_REF = "ref: ${{ github.event.pull_request.head.sha || github.sha }}"
VALIDATE_COMMAND = "run: python supply_chain_policy.py"
WORKFLOW_MANIFEST = "ci_supply_chain_manifest.json"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def reject_duplicate_json_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def validate_workflow_manifest(root, workflow_paths):
    manifest_path = root / WORKFLOW_MANIFEST
    document = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json_keys,
    )
    if set(document) != {"payload", "payload_sha256"}:
        raise ValueError("workflow manifest envelope is invalid")
    payload = document["payload"]
    if file_sha256_bytes(canonical_json(payload).encode()) != document["payload_sha256"]:
        raise ValueError("workflow manifest payload digest mismatch")
    if payload.get("schema_version") != 1 or set(payload) != {"schema_version", "workflows"}:
        raise ValueError("workflow manifest payload is invalid")
    expected_paths = {path.relative_to(root).as_posix() for path in workflow_paths}
    entries = payload["workflows"]
    if not isinstance(entries, list) or {entry.get("path") for entry in entries} != expected_paths:
        raise ValueError("workflow manifest paths do not match repository workflows")
    for entry in entries:
        if set(entry) != {"path", "sha256"} or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None:
            raise ValueError("workflow manifest entry is invalid")
        if file_sha256(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"workflow bytes do not match manifest: {entry['path']}")
    return file_sha256(manifest_path)


def file_sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def normalize_package_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pinned_requirement(line, path, line_number):
    match = PINNED_REQUIREMENT.fullmatch(line)
    if match is None:
        raise ValueError(f"{path}:{line_number}: dependency must use an exact version")
    return normalize_package_name(match.group("name")), match.group("version")


def validate_workflow_actions(path):
    references = []
    block_scalar_indent = None
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        indent = len(line) - len(line.lstrip())
        if block_scalar_indent is not None:
            if not line.strip() or indent > block_scalar_indent:
                continue
            block_scalar_indent = None
        content = line.split("#", 1)[0]
        if re.search(r":\s*[|>]?[+-]?\s*$", content) and re.search(r":\s*[|>][+-]?\s*$", content):
            block_scalar_indent = indent
        if re.match(r"^\s*(?:-\s*)?[\"']", content) or re.search(r"(?:^|\s)[&*][A-Za-z0-9_-]+", content):
            raise ValueError(f"{path}:{line_number}: unsupported YAML key, anchor, or alias syntax")
        if "<<:" in content or re.match(r"^\s*-\s*[\[{]", content):
            raise ValueError(f"{path}:{line_number}: unsupported YAML flow or merge syntax")
        match = re.fullmatch(r"\s*(?:-\s*)?uses:\s*([^\s]+)\s*", content)
        if not match:
            if USES_KEY.search(content):
                raise ValueError(f"{path}:{line_number}: action reference must use canonical block syntax")
            continue
        reference = match.group(1)
        if reference.startswith("./"):
            raise ValueError(f"{path}:{line_number}: local actions are not supported by the policy")
        if "@" not in reference or not ACTION_SHA.fullmatch(reference.rsplit("@", 1)[1]):
            raise ValueError(f"{path}:{line_number}: action reference must use a full commit SHA")
        references.append(reference)
    return references


def validate_workflow_commands(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    stripped = [line.strip() for line in lines]
    if stripped.count(VALIDATE_COMMAND) != 1:
        raise ValueError(f"{path}: workflow must run the supply-chain validator exactly once")
    expected_lock = "requirements-ci.lock" if Path(path).name == "rag-verifier-unit.yml" else "requirements-integration.lock"
    expected_install = f"run: python install_locked_requirements.py {expected_lock}"
    if stripped.count(expected_install) != 1:
        raise ValueError(f"{path}: workflow must use the approved locked-install wrapper exactly once")
    if stripped.index(VALIDATE_COMMAND) > stripped.index(expected_install):
        raise ValueError(f"{path}: supply-chain validation must precede dependency installation")
    if stripped.count(EXACT_HEAD_REF) != 1:
        raise ValueError(f"{path}: workflow must check out the exact event head")
    if any(re.match(r"^(?:container|services):", line) for line in stripped):
        raise ValueError(f"{path}: workflow containers and services are not supported")


def read_requirement_input(path):
    requirements = {}
    includes = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            include = line[3:].strip()
            if include != "requirements-ci.txt":
                raise ValueError(f"{path}:{line_number}: requirement include is not approved")
            includes.append(include)
            continue
        name, version = parse_pinned_requirement(line, path, line_number)
        if name in requirements:
            raise ValueError(f"{path}:{line_number}: duplicate dependency declaration")
        requirements[name] = version
    return requirements, includes


def validate_requirement_input(path):
    return read_requirement_input(path)[0]


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
    workflow_manifest_sha256 = validate_workflow_manifest(root, workflow_paths)
    for path in workflow_paths:
        validate_workflow_actions(path)
        validate_workflow_commands(path)
    ci_requirements, ci_includes = read_requirement_input(root / "requirements-ci.txt")
    integration_direct, integration_includes = read_requirement_input(root / "requirements-integration.txt")
    if ci_includes:
        raise ValueError("requirements-ci.txt: includes are not permitted")
    if integration_includes != ["requirements-ci.txt"]:
        raise ValueError("requirements-integration.txt: must include requirements-ci.txt exactly once")
    duplicate = set(ci_requirements) & set(integration_direct)
    if duplicate:
        raise ValueError(f"requirements-integration.txt: duplicate included dependency: {sorted(duplicate)[0]}")
    integration_requirements = {**ci_requirements, **integration_direct}
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
    return {
        "workflows": len(workflow_paths),
        "workflow_manifest_sha256": workflow_manifest_sha256,
        "locks": locks,
    }


if __name__ == "__main__":
    print(json.dumps(validate_repository(), sort_keys=True))
