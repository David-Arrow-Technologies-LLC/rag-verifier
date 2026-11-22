# Real-model NLI qualification — evidence audit and portfolio extension plan

## Purpose and scope

Extend the [portfolio walkthrough](portfolio-walkthrough.md) with **auditable real-model qualification evidence**, without duplicating existing qualification infrastructure or confusing synthetic-fixture results with measured NLI performance. This document is an inspection-derived plan, **not a new execution report**.

## Existing implementation (inspected on `main` at `8840a7608c6c048239898a0a26960147a06d720c`)

- [`nli_provider.py`](../nli_provider.py): Hugging Face NLI provider loads a model at a resolved pinned revision and exposes label probabilities; a separate `FakeNLIProvider` supports deterministic unit tests.
- [`nli_deberta_qualification_v2.json`](../nli_deberta_qualification_v2.json): digest-bound, schema-v2 synthetic medical-text benchmark, with entailment/contradiction/neutral labels, authored expected decisions, adversarial categories, critical cases, and qualification thresholds.
- Pinned model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`.
- [`nli_qualification.py`](../nli_qualification.py): validates manifest structure, digest, model identity, per-case outputs, and recomputed decision metrics.
- [`run_nli_qualification.py`](../run_nli_qualification.py): qualification entry point.
- [NLI integration workflow](../.github/workflows/rag-verifier-nli-integration.yml): checks out the exact PR head, validates supply-chain controls, installs locked integration dependencies, executes the real-model qualification, and uploads `qualification-evidence/nli-deberta-v2.json` with 90-day retention.
- [Existing qualification methodology](../nli_qualification.md) and [unified release qualification](../release_qualification.md) describe the current governance gates.

**Do not** implement a parallel model loader, benchmark runner, evidence schema, or CI workflow unless an independently demonstrated gap requires it.

## Evidence required before publishing real-model performance claims

1. Identify an actual completed NLI Integration Gate run and capture the run URL, job ID, exact PR head SHA, completion timestamp, and conclusion.
2. Retrieve the retained `nli-qualification-<head-sha>` artifact from **that same run**. If it has expired or is absent, explicitly record the gap and schedule a separately authorized rerun.
3. Inspect the artifact's exact source revision, manifest SHA-256, model ID/revision, runtime package versions, per-case expected and observed labels/decisions, and aggregate metrics. Compare these against the source manifest and runner contract.
4. Recompute metrics from the retained per-case results; do not infer accuracy from workflow success alone. Record any failures, per-label performance, critical-case mismatches, and abstentions.
5. Hash the downloaded artifact bytes, record their SHA-256 and provenance, and preserve the unmodified evidence artifact where permitted. A digest displayed in a report must not be represented as independently verified unless the original bytes were examined.
6. Only then create a separate, revision-bound **observed real-model results** page. Include precise denominator, benchmark limitations, and run identity.

## Evaluation questions for the next independent study

- How does the pinned model perform across entailment, contradiction, and neutral cases, not merely in aggregate?
- What is the false-PASS rate for contradictory or unsupported claims? Which critical cases fail?
- How do dosage, numbers/units, negation, temporal order, laterality, allergy, and paraphrase categories behave?
- How often does the release policy abstain (`REVIEW`) on uncertain inputs, and what are the resulting coverage/risk tradeoffs?
- Are model probability outputs well calibrated? A calibration claim requires a separate sufficiently sized evaluation dataset and method.
- Does performance persist on an independent, non-overlapping, pre-registered synthetic holdout? Existing authored qualification fixtures alone cannot establish generalization.

## Controlled next slices

**Slice A — read-only evidence audit:** inspect the most recent relevant successful run and its artifact; publish findings in a review note without code or workflow mutation.

**Slice B — results documentation:** if the artifact is available and verified, prepare a separate documentation-only PR with a source-bound metrics table, case-level error analysis, and limitations.

**Slice C — evaluation expansion (separate authorization):** design an independent synthetic holdout and pre-register pass/fail, false-PASS, abstention, and category-level measures; implement only after reviewing overlap with the existing schema and tests.

## Boundaries

These are authored synthetic medical-text engineering cases, **not** patient records, clinical benchmark validation, or authorization for diagnosis, treatment, triage, or autonomous care. A passing CI gate supports only the exact defined engineering qualification criteria, not general clinical reliability. No new run, artifact download, performance metric, or clinical validation is claimed by this document.
