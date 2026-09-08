# Reproducible CI supply chain

RAG-V27 binds CI execution to immutable GitHub Action commits and complete,
SHA-256-hashed Python dependency locks. Human-maintained requirement inputs keep
their direct dependencies exact; CI installs only from the generated lockfiles
with `--require-hashes`.

`supply_chain_policy.py` is standard-library-only and runs before third-party
dependency installation. It fails closed on floating action references,
non-exact dependency inputs, unapproved includes, unhashed lock entries, empty
locks, or missing workflows. The real-NLI evidence records the exact integration
lock filename and digest used by the qualifying source revision.

To update dependencies, edit the applicable `.txt` input, regenerate both locks
with a reviewed resolver, run the policy and complete test suite, and require a
fresh real-model qualification artifact. Dependency refreshes must never be
combined with benchmark, threshold, manifest, or verifier-policy changes.
