# Runtime drift and requalification

RAG-V29 compares a previously qualified `PASS` release bundle with an observed
release environment under a versioned, digest-bound trigger policy. The prior
qualification remains `CURRENT` only when every governed identity is unchanged.

Changes to source, qualification policy or manifests, CI workflow policy,
runner image, dependency lock or installed inventory, Python or ML runtime
packages, component manifests, or qualification status produce
`REQUALIFICATION_REQUIRED`. Missing fields, malformed evidence, digest
tampering, duplicate JSON keys, and non-PASS baselines fail closed.

After successful requalification, the newly produced PASS bundle becomes the
new baseline. This is engineering release evidence and does not establish
clinical validity or authorize autonomous clinical use.
