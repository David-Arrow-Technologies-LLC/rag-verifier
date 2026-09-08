import subprocess
import sys
from pathlib import Path

from supply_chain_policy import validate_lockfile


APPROVED_LOCKS = {"requirements-ci.lock", "requirements-integration.lock"}


def install(lock_name):
    if lock_name not in APPROVED_LOCKS or Path(lock_name).name != lock_name:
        raise ValueError("dependency installation requires an approved repository lock")
    validate_lockfile(lock_name)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--require-hashes", "-r", lock_name],
        check=True,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: install_locked_requirements.py <approved-lock>")
    install(sys.argv[1])
