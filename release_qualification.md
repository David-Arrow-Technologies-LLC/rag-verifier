# Unified release qualification

RAG-V28 produces one canonical, digest-bound release qualification bundle from
the governed MiniLM retrieval and DeBERTa NLI qualifications. The bundle binds
the exact source revision, static release manifest, CI workflow policy,
dependency locks, installed distribution inventory, managed runner identity,
component records, component decisions, and overall decision.

The overall status is `PASS` only when both governed components return `PASS`.
Missing, changed, duplicate, unknown, mismatched, or non-PASS inputs fail closed.
This is engineering release evidence; it does not establish clinical validity or
authorize autonomous clinical use.
