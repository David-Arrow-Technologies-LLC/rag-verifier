# Versioned healthcare qualification manifest

This contract binds one embedding qualification run to canonical, digest-protected inputs.

The manifest payload contains the schema version, benchmark identity and version, `top_k`, pinned model metadata, chunk corpus, queries, and relevance judgments. Its SHA-256 digest is verified before any provider is loaded.

Qualification also verifies that the supplied runner uses exactly the benchmark identity, version, and `top_k` declared by the manifest. The result couples the governed qualification record with the verified manifest digest.

This is a deterministic governance slice. A subsequent integration slice may add an approved real healthcare corpus and execute a pinned external model, but those assets must enter through this manifest contract.
