# Reproducible CI supply chain

RAG-V27 binds repository-controlled CI execution to a digest-verified workflow
manifest, immutable GitHub Action commits, exact Python patch versions, and
complete SHA-256-hashed Python dependency locks. Human-maintained requirement
inputs keep their direct dependencies exact; CI installs only through the
repository's allowlisted wrapper using pip's `--require-hashes` mode.

`supply_chain_policy.py` is standard-library-only and runs before third-party
dependency installation. It fails closed on floating action references,
unsupported YAML action mappings, local actions, unapproved install commands,
non-exact dependency inputs, invalid include graphs, input/lock drift, unhashed
lock entries, empty locks, missing exact-head checkout, or missing workflows.
The real-NLI evidence records the exact integration lock filename and digest and
fails unless every locked distribution is installed at its declared version; the
verified installed-package inventory and its digest are retained in the artifact.

GitHub's managed `ubuntu-24.04` host image cannot be selected by immutable image
digest and is therefore outside the byte-for-byte reproducibility boundary. The
NLI artifact records GitHub's `ImageOS` and `ImageVersion` identities so a run can
be traced to that managed substrate; repository-defined job and service
containers are prohibited.

To update dependencies, edit the applicable `.txt` input, regenerate both locks
with a reviewed resolver, run the policy and complete test suite, and require a
fresh real-model qualification artifact. Dependency refreshes must never be
combined with benchmark, threshold, manifest, or verifier-policy changes.
