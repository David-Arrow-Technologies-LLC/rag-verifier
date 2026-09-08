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


@pytest.mark.parametrize(
    "step",
    (
        "  - {name: Checkout, uses: actions/checkout@v4}\n",
        "  - uses : actions/checkout@v4\n",
        "  - 'uses': actions/checkout@v4\n",
    ),
)
def test_noncanonical_action_mapping_is_rejected(tmp_path, step):
    workflow = tmp_path / "workflow.yml"
    workflow.write_text("steps:\n" + step, encoding="utf-8")
    with pytest.raises(ValueError, match="canonical block syntax"):
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


@pytest.mark.parametrize("declared", ("pytest==8.4.2", "pytest==9.1.1\nnew-package==1.0.0"))
def test_stale_or_missing_direct_requirement_is_rejected(tmp_path, declared):
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "gate.yml").write_text(
        "steps:\n  - uses: actions/checkout@" + "a" * 40 + "\n", encoding="utf-8"
    )
    (tmp_path / "requirements-ci.txt").write_text(declared + "\n", encoding="utf-8")
    (tmp_path / "requirements-integration.txt").write_text(
        "-r requirements-ci.txt\nsentence-transformers==6.0.0\n", encoding="utf-8"
    )
    digest = "a" * 64
    (tmp_path / "requirements-ci.lock").write_text(
        f"pytest==9.1.1 --hash=sha256:{digest}\n", encoding="utf-8"
    )
    (tmp_path / "requirements-integration.lock").write_text(
        f"pytest==9.1.1 --hash=sha256:{digest}\n"
        f"sentence-transformers==6.0.0 --hash=sha256:{digest}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match its requirement input"):
        validate_repository(tmp_path)
