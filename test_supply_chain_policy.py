import pytest

from supply_chain_policy import validate_lockfile, validate_repository, validate_requirement_input, validate_workflow_actions


def test_repository_supply_chain_is_immutable_and_hash_locked():
    result = validate_repository()
    assert result["workflows"] == 3
    assert result["locks"]["requirements-ci.lock"]["packages"] > 0
    assert result["locks"]["requirements-integration.lock"]["packages"] > 0


def test_floating_action_reference_is_rejected(tmp_path):
    workflow = tmp_path / "workflow.yml"
    workflow.write_text("steps:\n  - uses: actions/checkout@v4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="full commit SHA"):
        validate_workflow_actions(workflow)


def test_unhashed_lock_entry_is_rejected(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("pytest==9.1.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_lockfile(lock)


def test_non_sha256_lock_hash_is_rejected(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("pytest==9.1.1 --hash=sha256:not-a-digest\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_lockfile(lock)


def test_version_range_is_rejected(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pytest>=9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact version"):
        validate_requirement_input(requirements)


def test_unapproved_requirement_include_is_rejected(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("-r ../untrusted.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not approved"):
        validate_requirement_input(requirements)
