# Slice C — preregistered independent synthetic NLI holdout protocol

**Status:** protocol proposal for review, not an executed study or a claim of model performance.

## Research question

Does the fixed, revision-pinned DeBERTa NLI provider and the unchanged verifier release policy maintain acceptable performance on independently authored synthetic medical-text premise/hypothesis pairs that do not overlap the existing 30-case qualification set?

## Fixed baseline and leakage boundary

- Baseline source: `11742ae6fba045937d117a1ac84f62d343fd62a5`.
- Existing training/evaluation exposure reference: `nli_deberta_qualification_v2.json`, 30 authored cases. Do not reuse its premise/hypothesis pairs or simple lexical substitutions.
- Frozen model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`.
- Frozen release thresholds: entailment PASS >= 0.9; contradiction FAIL >= 0.9, with remaining uncertain cases reviewed according to existing policy. Do not tune on holdout.
- Existing `run_nli_qualification.py` and `nli_qualification.py` remain the authoritative execution/evidence contracts. Inspect extension compatibility before implementation; avoid creating a duplicate runner.

## Sampling and authorship

Target **90 new synthetic pairs**, balanced across expected labels (30 entailment, 30 contradiction, 30 neutral). Within each label, target at least three cases from each of ten predeclared categories: comorbidity/independent conditions, medication/dose, laboratory/numeric units, negation, temporal sequence, laterality, allergy, demographic attributes, missing evidence, and paraphrase/semantic equivalence. Category/label feasibility must be reviewed before freezing; record justified exceptions without changing the total label balance.

Cases must be fictional, contain no PHI, and be authored and reviewed independently of the baseline case texts and observed model errors. The authors may target error *categories* revealed by prior results but must not copy or lightly rewrite the known failing cases. Freeze the complete manifest and SHA-256 before inference. Maintain a provenance ledger and a deterministic leakage report comparing normalized strings and near-duplicates against baseline and within holdout.

Two independent reviewers should label expected NLI class and release decision while blinded to model outputs; resolve disagreements before manifest freeze, preserving original labels and adjudication history. If independent reviewers are unavailable, mark the dataset **unblinded/single-annotator** and prohibit an independent-annotation claim.

## Primary outcomes (descriptive)

1. **Incorrect automatic PASS rate on expected non-PASS cases:** count observed PASS among expected contradiction or neutral / total expected contradiction plus neutral (planned denominator 60). **Safety gate: zero** observed incorrect PASS; any occurrence is a failed gate and requires case-level investigation.
2. **NLI label accuracy:** exact matches / all 90. Prespecified engineering target >= 0.90.
3. **Verifier decision accuracy:** exact PASS/REVIEW/FAIL matches / all 90. Prespecified engineering target >= 0.90.
4. **Critical-case integrity:** pre-mark critical cases before inference; all must match both expected label and expected decision. A single critical failure fails this gate.

The study must report raw counts and denominators even when gates fail. These thresholds are research engineering acceptance criteria, not estimates of clinical safety or validated operating points.

## Secondary outcomes

- Full 3×3 NLI confusion matrix and 3×3 decision confusion matrix.
- Per-label and per-category accuracy with numerator/denominator.
- Incorrect BLOCK on expected REVIEW; unnecessary REVIEW/BLOCK on expected PASS.
- Abstention (`REVIEW`) frequency overall and by expected label.
- False-PASS rate with an exact binomial 95% interval, explicitly noting the small sample and nonrepresentative sampling.
- Critical and noncritical error lists, including predicted probabilities and decision threshold crossings.

No post-hoc threshold adjustment, case removal, relabeling, or benchmark expansion may be used to convert a failed primary gate into a pass. Any exploratory analysis must be marked post-hoc.

## Execution and audit gates

1. **Read-only inspection:** confirm existing manifest schema, evaluator, test coverage, source SHA and CI execution cost/permissions.
2. **Dataset PR:** submit the holdout cases, independent annotation evidence, duplicate/leakage tests, and manifest digest. Review and freeze before inference.
3. **Execution PR / explicit approval:** run the pinned model in an isolated nonclinical environment, capture exact source revision, dependency lock, model revision, case-level evidence, workflow identity and artifact SHA-256. Avoid automatic model execution on the dataset-authoring PR before freeze.
4. **Independent audit:** download original artifact, verify digest, recompute every metric from case records, investigate all discrepancies, and publish a revision-bound results report.
5. **Governance:** keep clinical use, real patient data, production release and autonomous decisions prohibited. A passing holdout does not establish clinical validity or deployment readiness.

## Integration and portfolio implications

Publish the protocol as **planned Slice C research** until the dataset is frozen and inference completed. Once audited, publish results with limitations and a link to the exact CI artifact. Keep K2A's separately governed integration/staging qualification distinct from RAG Verifier's NLI evaluation; neither project's test results prove the other's deployment readiness.
