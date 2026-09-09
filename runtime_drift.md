# Runtime drift and requalification

RAG-V29 compares a previously qualified `PASS` release bundle with an observed
release environment under a versioned, digest-bound trigger policy. The prior
qualification remains `CURRENT` only when every governed identity is unchanged.
The caller must provide the baseline payload SHA-256 from an independent trusted
release record; a self-consistent artifact is not treated as self-authenticating.

Changes to source, qualification policy or manifests, CI workflow policy,
runner image, dependency lock or installed inventory, Python or ML runtime
packages, component manifests, or qualification status produce
`REQUALIFICATION_REQUIRED`. Missing fields, malformed evidence, digest
tampering, duplicate JSON keys, and non-PASS baselines fail closed.
Truncated release schemas, inconsistent aggregate/component decisions, altered
installed inventories, and missing or `unavailable` runner identities also fail
closed before comparison.
The complete MiniLM and NLI qualification records are monitored as canonical
objects, so any record change triggers requalification without copying large
case-level records into the drift result.

After successful requalification, the newly produced PASS bundle becomes the
new baseline. This is engineering release evidence and does not establish
clinical validity or authorize autonomous clinical use.
