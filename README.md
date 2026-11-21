# DA Tech RAG Verifier

**Evidence-grounded AI engineering | Self-verifying Retrieval-Augmented Generation (RAG)**

RAG Verifier is a Python research and engineering project for checking whether generated answers are supported by retrieved evidence and for making policy-governed release decisions. It is part of David Arrow Technologies' broader work on trustworthy and auditable AI.

> **Safety boundary:** Engineering and research use only. The repository's tests and qualification artifacts do **not** establish clinical validity, medical-device clearance, production readiness, or authorization for autonomous patient care.

## Start here: technical portfolio walkthrough

**[Explore the RAG Verifier portfolio walkthrough](docs/portfolio-walkthrough.md)** — a concise guide to the architecture, five deterministic synthetic verification cases, release-policy decisions, GitHub CI evidence, and user-reported local execution provenance. The demonstration uses `FakeNLIProvider` and is **not** an end-to-end live-model benchmark or clinical validation.

## Why verification matters

A fluent AI answer may contain unsupported statements, misleading citations, or evidence that was not actually retrieved. RAG Verifier investigates claim-level evidence checks and explicit release decisions rather than assuming that retrieval alone guarantees a correct answer.

## Architecture

```text
Question + indexed reference chunks
                |
           Retrieval
                |
            Reranking
                |
         Context building
                |
       Answer generation
                |
    Claim and citation checks
       + NLI-based evidence
                |
       Release policy gate
          /     |      \
       RELEASE REVIEW   BLOCK
```

The diagram describes the implemented orchestration structure and decision concepts. It is not a live clinical deployment diagram.

## Source components

| Component | Purpose | Source |
| --- | --- | --- |
| RAG orchestration | Coordinates retrieval, reranking, context, generation and verification | [`self_verifying_rag.py`](self_verifying_rag.py) |
| Claim verification | Citation extraction, claim splitting and evidence/NLI checks | [`verifier.py`](verifier.py) |
| Release decisions | Maps verification outcomes to RELEASE, HOLD_FOR_REVIEW or BLOCK | [`release_policy.py`](release_policy.py) |
| Synthetic example | Demonstrates verifier inputs and structured outputs | [`run_demo.py`](run_demo.py) |
| Automated tests | Unit and integration coverage for pipeline components and policy behavior | [`test_unit_verifier.py`](test_unit_verifier.py), [`test_self_verifying_rag.py`](test_self_verifying_rag.py), [`test_release_policy.py`](test_release_policy.py) |

Additional source modules cover retrieval, embeddings, reranking, context building, model providers, and evaluation.

## Engineering assurance

- **Deterministic unit gate:** [GitHub Actions workflow](.github/workflows/rag-verifier-unit.yml) checks supply-chain policy, installs hashed locked dependencies and runs `python -m pytest -m "not integration" -v`.
- **Real-model NLI qualification:** [Qualification methodology](nli_qualification.md) describes pinned model revisions, synthetic cases, threshold checks and digest-bound evidence artifacts.
- **Unified release qualification:** [Release qualification](release_qualification.md) combines governed retrieval and NLI component decisions and fails closed on mismatches or non-PASS evidence.
- **Supply-chain controls:** [Reproducibility policy](supply_chain.md) documents immutable action references, pinned Python versions, hashed locks and their limitations.
- **Runtime drift:** [Runtime drift documentation](runtime_drift.md) describes checks on runtime qualification assumptions.

These are implemented test and qualification **mechanisms**. This README does not assert that every current CI run passed; consult the relevant workflow run and exact commit for results.

## Reproducing the deterministic test gate

Use a supported Python environment matching the [unit workflow](.github/workflows/rag-verifier-unit.yml). The repository pins its CI interpreter to Python 3.13.14 and uses hashed requirements.

```bash
python supply_chain_policy.py
python install_locked_requirements.py requirements-ci.lock
python -m pytest -m "not integration" -v
```

These commands reproduce the repository-defined deterministic test steps when run in a compatible environment. They are **not** a substitute for the separate real-model integration and release qualification gates. Check workflow and lockfile constraints before running locally. No claim is made here that these commands were executed as part of this README change.

A standalone example is available in [`run_demo.py`](run_demo.py), which invokes the verifier on synthetic medical-reference excerpts. Running the example may require additional model dependencies and access to the pinned NLI model; it is not part of the lightweight deterministic gate.

## Evaluation and limitations

- NLI predictions and citation presence are **not** guarantees of medical correctness.
- A supported claim can still be outdated, incomplete, or clinically inappropriate.
- Engineering fixtures and synthetic cases are not prospective clinical validation.
- Reproducibility is bounded by the documented runtime, model, and supply-chain assumptions.
- Human oversight, source provenance, privacy, and clinical governance remain essential for any future clinical application.
- This repository must not be used to diagnose, prescribe, triage, or independently treat patients.

## Further technical reading

- [NLI qualification](nli_qualification.md)
- [Healthcare MiniLM retrieval qualification](healthcare_minilm_qualification.md)
- [Unified release qualification](release_qualification.md)
- [Supply-chain controls](supply_chain.md)
- [Runtime drift](runtime_drift.md)
- [Decimal claim-boundary handling](decimal-claim-boundaries.md)

## About

Developed as part of **David Arrow Technologies (DA Tech)** healthcare AI and responsible AI engineering research.

**Olanrewaju Onala, MBBS** — Physician, Founder and AI Platform Architect.

*This project is an engineering research system, not a clinically validated medical service.*
