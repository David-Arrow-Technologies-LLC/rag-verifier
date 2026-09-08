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
    with pytest.raises(ValueError, match="canonical block syntax|unsupported YAML"):
        validate_workflow_actions(workflow)


def test_escaped_yaml_action_key_is_rejected(tmp_path):
    workflow = tmp_path / "workflow.yml"
    workflow.write_text('steps:\n  - "\\x75ses": actions/checkout@v4\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported YAML"):
        validate_workflow_actions(workflow)


def test_local_action_is_rejected_until_recursive_validation_exists(tmp_path):
    workflow = tmp_path / "workflow.yml"
    workflow.write_text("steps:\n  - uses: ./.github/actions/local\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local actions"):
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


def test_wildcard_version_is_rejected(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pytest==9.*\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact version"):
        validate_requirement_input(requirements)


def test_unapproved_requirement_include_is_rejected(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("-r ../untrusted.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not approved"):
        validate_requirement_input(requirements)


@pytest.mark.parametrize("declared", ("pytest==8.4.2", "pytest==9.1.1\nnew-package==1.0.0"))
def test_stale_or_missing_direct_requirement_is_rejected(tmp_path, declared):
    _write_valid_repository(tmp_path)
    (tmp_path / "requirements-ci.txt").write_text(declared + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its requirement input"):
        validate_repository(tmp_path)


def test_repository_rejects_missing_integration_include(tmp_path):
    _write_valid_repository(tmp_path)
    (tmp_path / "requirements-integration.txt").write_text("sentence-transformers==6.0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must include"):
        validate_repository(tmp_path)


def test_repository_rejects_conflict_with_included_requirement(tmp_path):
    _write_valid_repository(tmp_path)
    (tmp_path / "requirements-integration.txt").write_text(
        "-r requirements-ci.txt\npytest==8.4.2\nsentence-transformers==6.0.0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate included dependency"):
        validate_repository(tmp_path)


def test_repository_rejects_unhashed_pip_install(tmp_path):
    _write_valid_repository(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "rag-verifier-unit.yml"
    workflow.write_text(workflow.read_text(encoding="utf-8") + "  - run: python -m pip install untrusted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash-locked install"):
        validate_repository(tmp_path)


def _write_valid_repository(root):
    workflow_root = root / ".github" / "workflows"
    workflow_root.mkdir(parents=True)
    workflow_root.joinpath("rag-verifier-unit.yml").write_text(
        "steps:\n"
        "  - uses: actions/checkout@" + "a" * 40 + "\n"
        "    with:\n"
        "      ref: ${{ github.event.pull_request.head.sha || github.sha }}\n"
        "  - name: Validate\n"
        "    run: python supply_chain_policy.py\n"
        "  - name: Install\n"
        "    run: python -m pip install --require-hashes -r requirements-ci.lock\n",
        encoding="utf-8",
    )
    root.joinpath("requirements-ci.txt").write_text("pytest==9.1.1\n", encoding="utf-8")
    root.joinpath("requirements-integration.txt").write_text(
        "-r requirements-ci.txt\nsentence-transformers==6.0.0\n", encoding="utf-8"
    )
    digest = "a" * 64
    root.joinpath("requirements-ci.lock").write_text(
        f"pytest==9.1.1 --hash=sha256:{digest}\n", encoding="utf-8"
    )
    root.joinpath("requirements-integration.lock").write_text(
        f"pytest==9.1.1 --hash=sha256:{digest}\n"
        f"sentence-transformers==6.0.0 --hash=sha256:{digest}\n",
        encoding="utf-8",
    )
