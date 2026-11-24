# Freeze evidence preparation (research only)

This slice verifies the private mapping against the exact candidate and packet, recomputes adjudication, checks the 10-by-3 category/label balance, and produces deterministic lexical leakage findings against the supplied baseline and within the candidate. It reports the digest of the proposed payload without emitting an executable manifest or modifying the candidate.

The coordinator must use the existing candidate audit and pinned baseline. This API does not establish baseline authenticity, reviewer credentials, independent authorship, semantic leakage clearance, criticality approval or scientific approval. The report is not a freeze readiness certificate. The candidate remains unfrozen.

## External evidence contract (planned, not authenticated here)

A later authority implementation must bind verified reviewer identities, credential issuer and verification evidence, independence/conflict declarations, exact annotation and adjudication digests, candidate and baseline identities, criticality rationale, disposition of every leakage finding, and scientific approval to the proposed payload digest. A separately trusted authorizer must approve the exact version and scope. Do not accept caller-supplied booleans or self-issued credentials as authorization. The current authorize_freeze function always denies, including for test fixtures.

A future authorized manifest generator must reuse VersionedNLIQualificationManifest and its canonical payload digest, with a new benchmark version, and enforce the preregistered false-PASS gate before any confirmatory claim. No inference or clinical use is permitted by this slice.
