import pytest

from install_locked_requirements import install


def test_installer_rejects_unapproved_lock():
    with pytest.raises(ValueError, match="approved repository lock"):
        install("../untrusted.lock")


def test_installer_uses_exact_fail_closed_pip_arguments(tmp_path, monkeypatch):
    lock = tmp_path / "requirements-ci.lock"
    lock.write_text("pytest==9.1.1 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    observed = {}
    monkeypatch.setattr(
        "install_locked_requirements.subprocess.run",
        lambda command, check: observed.update({"command": command, "check": check}),
    )
    install("requirements-ci.lock")
    assert observed["command"][2:] == ["pip", "install", "--require-hashes", "-r", "requirements-ci.lock"]
    assert observed["check"] is True
