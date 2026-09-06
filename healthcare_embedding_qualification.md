# Healthcare embedding qualification

This slice connects the governed healthcare retrieval benchmark to a concrete embedding-provider qualification run.

The runner:

- binds candidate metadata to the identity and revision observed by the provider factory;
- preflights both query and document encoders and enforces the declared dimension on every benchmark embedding;
- rejects relevance judgments that do not exist in the supplied chunk corpus before loading a provider;
- executes the existing `HealthcareRetrievalBenchmark` through the production `Retriever` contract;
- evaluates the resulting metrics through the existing `ModelQualificationPolicy`; and
- emits the existing governed `ModelQualificationRecord`.

Operational model-loading or inference failures remain operational failures. They are not converted into a model `FAIL` qualification result, because `FAIL` is reserved for a completed benchmark whose metrics fail policy thresholds.

The deterministic tests use `FakeEmbeddingProvider`. Actual healthcare model candidates and externally sourced benchmark corpora remain a subsequent integration/evaluation step so that model downloads and network access do not enter the deterministic unit gate.
