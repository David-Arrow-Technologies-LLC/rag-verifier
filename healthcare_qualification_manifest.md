# Versioned healthcare qualification manifest

This contract binds one embedding qualification run to canonical, digest-protected inputs.

The manifest payload contains the schema version, benchmark identity and version, `top_k`, pinned model metadata, qualification-policy thresholds, chunk corpus, queries, and relevance judgments. Sentence-transformer revisions must be immutable lowercase 40-character commit SHAs. Its SHA-256 digest is verified before any provider is loaded.

Qualification also verifies that the supplied runner uses exactly the benchmark identity, version, `top_k`, and policy thresholds declared by the manifest. The returned record is validated against the manifest and manifest-derived policy before the result couples it with the verified digest.

This is a deterministic governance slice. A subsequent integration slice may add an approved real healthcare corpus and execute a pinned external model, but those assets must enter through this manifest contract.
