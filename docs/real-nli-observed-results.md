# Observed real-model NLI qualification — revision-bound evidence report

**Evidence type:** observed Hugging Face model inference on 30 repository-authored synthetic medical-text pairs. **Not** the deterministic `FakeNLIProvider` fixture, an independent holdout, or clinical validation.

## Source and chain of custody

| Field | Observed evidence |
| --- | --- |
| GitHub Actions run | [RAG Verifier NLI Integration Gate, run 45019205079](https://github.com/David-Arrow-Technologies-LLC/rag-verifier/actions/runs/45019205079) |
| Qualification job | `qualify-nli`, job ID `137198138548`; conclusion `success` |
| Evaluated source revision | `1211dbd6588fb0ae189c98dc6135b3e8c5b14a6b` (PR #38 head, **not** the later squash-merge SHA) |
| Artifact | GitHub artifact ID `14753376012`, `nli-qualification-1211dbd6588fb0ae189c98dc6135b3e8c5b14a6b` |
| ZIP SHA-256 | `c50725e32e9d0a0f09ce4ebb20ab240269cb5e1cd9252223621853bb2f81d734` |
| Extracted `nli-deberta-v2.json` SHA-256 | `61046c9d607364bde334a33f1bd4de101a66478ecd5a9ef98036cb5f4ab52f9c` |
| Evidence payload SHA-256 (internal) | `6c38258e74eef99fbad73cf322699ec77d1674db6d88319b605ecbc4a9d1d095` |
| Manifest SHA-256 (reported by evidence) | `5f2f90d865c067f32ce902fe0fa01cc7508b81a6f3aca35b800619f55c5b3b13` |
| Benchmark | `nli-verifier-contract`, `synthetic-medical-nli-v2`, 30 authored cases |
| Model | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| Pinned model revision | `6f5cf0a2b59cabb106aca4c287eed12e357e90eb` |
| Policy thresholds | `pass_threshold=0.9`, `fail_threshold=0.9` |
| Runtime | Python 3.12.13, PyTorch 2.14.0, Transformers 5.16.1, tokenizers 0.23.2 |

The artifact was downloaded from the specified successful run, its ZIP bytes were SHA-256 hashed and matched GitHub's published artifact digest, and the contained JSON bytes were separately hashed. Reported aggregate results were independently recalculated from the 30 case-result records. These checks establish **artifact integrity and internal metric consistency**; they do not constitute an independent model rerun or independent verification of the authored ground-truth labels.

## Observed results

| Measure | Recomputed numerator / denominator | Value |
| --- | ---: | ---: |
| NLI label accuracy | 28 / 30 | 93.33% |
| Verifier decision accuracy | 27 / 30 | 90.00% |
| Entailment label accuracy | 10 / 10 | 100.00% |
| Contradiction label accuracy | 10 / 10 | 100.00% |
| Neutral label accuracy | 8 / 10 | 80.00% |
| Critical cases correct on both label and decision | 10 / 10 | 100.00% |
| Incorrect `PASS` decisions on expected `FAIL` or `REVIEW` cases | 0 / 20 | 0.00% |

The qualification record reports `PASS` with reason `QUALIFICATION_THRESHOLDS_SATISFIED`. This is a **thresholded engineering qualification** for this specific authored benchmark, not a claim of error-free inference. In particular, the 0/20 incorrect-PASS observation has a very small denominator and does not establish a population false-release rate.

### Per-case discrepancies

| Case | Category | Expected label / decision | Observed label / decision | Interpretation |
| --- | --- | --- | --- | --- |
| `ENT-07` | negated-finding | entailment / `PASS` | entailment / `REVIEW` | Label correct, but entailment score `0.78564453125` is below the 0.9 PASS threshold; conservative abstention |
| `NEU-01` | comorbidity | neutral / `REVIEW` | contradiction / `FAIL` | Neutral evidence incorrectly treated as contradiction; contradiction score `0.9931640625` |
| `NEU-04` | comorbidity | neutral / `REVIEW` | contradiction / `FAIL` | Neutral evidence incorrectly treated as contradiction; contradiction score `0.998046875` |

All ten expected contradiction cases were classified `FAIL`. Eight of ten expected neutral cases were classified `REVIEW`; two were classified `FAIL`. Nine of ten expected entailment cases were classified `PASS`; one was classified `REVIEW`. Thus the observed decisions were **9 PASS, 9 REVIEW, 12 FAIL** (expected: 10 PASS, 10 REVIEW, 10 FAIL). No observed non-PASS expected case was automatically passed.

**Interpretive caution:** A `FAIL` on a neutral case is conservative with respect to preventing unsupported release, but it mischaracterizes absent evidence as contradiction. This matters for truthful explanations and downstream review workflows. A `REVIEW` on a supported case may increase unnecessary human review.

## Reproduction and review procedure

1. Open the exact [workflow run](https://github.com/David-Arrow-Technologies-LLC/rag-verifier/actions/runs/45019205079) and locate artifact ID `14753376012`; verify the run is associated with head SHA `1211dbd6588fb0ae189c98dc6135b3e8c5b14a6b`.
2. Download the original artifact ZIP while retained. Hash the ZIP bytes and compare to the recorded ZIP digest above; extract `nli-deberta-v2.json` and hash its exact bytes separately.
3. Parse `payload.qualification_record.case_results` and independently count expected-versus-observed label and status matches, grouped by expected label and `critical`. Do not substitute the summary metrics for case-level recomputation.
4. Compare the artifact's model revision, benchmark, thresholds, source revision, and manifest digest with the exact revision's [qualification manifest](../nli_deberta_qualification_v2.json) and [runner](../nli_qualification.py). The links point to current repository files; **use the historical SHA** when reproducing the run.
5. Treat the CI conclusion as an execution signal, not as independent proof of clinical correctness or external validity.

## Limitations and next evaluation

- All 30 examples are repository-authored synthetic medical-text pairs. They are not a representative clinical dataset or a blinded, independently annotated holdout.
- The observed percentages are descriptive on this fixed sample. No confidence intervals, generalization, calibration, sensitivity/specificity on clinical cases, or prospective clinical safety claims are established.
- The artifact records model outputs, not independent re-execution; verification here covers the recorded evidence and arithmetic.
- No patient data, patient-care authorization, deployment, or medical-device clearance is implied.

**Next separately authorized step:** design and preregister an independent non-overlapping synthetic holdout with additional comorbidity/neutral, negation, dosage, numeric, temporal, laterality, and missing-evidence cases. Report per-label confusion, incorrect automatic PASS, unnecessary BLOCK, abstention rate, and category-level failures. Reuse the existing qualification infrastructure after inspecting its extension points.
