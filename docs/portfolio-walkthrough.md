# Portfolio walkthrough — evidence-grounded RAG verification

**Audience:** AI engineers, research collaborators, and technical reviewers.  
**Demonstration type:** deterministic, fictional, nonclinical fixture.  
**Claim:** the implemented verifier and release-policy functions produce the documented decisions on five authored synthetic cases; this does **not** establish real-model accuracy or clinical effectiveness.

## The engineering problem

Retrieval-augmented generation can produce fluent but unsupported claims. A citation may be present yet fabricated, may point to evidence that was never retrieved, or may contradict the answer. The project separates claim/evidence checking from an explicit, auditable release-policy decision.

## System architecture

The repository's broader [orchestration design](../README.md) comprises:

```text
Question + indexed chunks
          |
       Retrieval
          |
       Reranking
          |
    Context building
          |
    Answer generation
          |
 Claim / citation checks + NLI
          |
      ReleasePolicy
     /      |       \
 RELEASE  HOLD      BLOCK
        FOR_REVIEW
```

**Important scope distinction:** the standalone fixture does **not** execute the full retrieval → reranking → generation pipeline. It directly exercises [`RAGVerifier`](../verifier.py) and [`ReleasePolicy`](../release_policy.py) with [`FakeNLIProvider`](../nli_provider.py), authored evidence chunks, and a fixed set of retrieved IDs.

## Five-case demonstration

The script [`synthetic_verifier_demo.py`](../synthetic_verifier_demo.py) uses fictional sample-storage statements and checks expected status/action pairs:

| Case | Condition being exercised | Verifier status | Policy action |
| --- | --- | --- | --- |
| Supported | Authored fake entailment supports a cited, retrieved chunk | PASS | RELEASE |
| Uncertain | Authored fake neutral score requires review | REVIEW | HOLD_FOR_REVIEW |
| Contradicted | Authored fake contradiction opposes the cited statement | FAIL | BLOCK |
| Fabricated citation | Citation ID is absent from the source dictionary | FAIL | BLOCK |
| Unretrieved citation | Citation exists but was not in the retrieved-ID set | FAIL | BLOCK |

The fixture asserts all five expected outcomes. **RELEASE is a synthetic software-policy return value**, not approval to release content into a clinical or production system.

### What the outcomes show

The user-supplied local transcript reported **5/5 matched**, with `trusted_release = true` only for the supported fixture. Every case reported 100% citation *coverage*, including cases that failed verification. Coverage therefore must not be confused with evidence validity or claim support.

## Evidence chain

| Evidence class | Source | What is actually established |
| --- | --- | --- |
| Source implementation | [Verifier](../verifier.py), [policy](../release_policy.py), [fixture](../synthetic_verifier_demo.py) | Inspectable behavior and authored expectations |
| GitHub CI | [Exact unit workflow run](https://github.com/David-Arrow-Technologies-LLC/rag-verifier/actions/runs/44948210193), [CI evidence](synthetic-ci-evidence.md) | At tested PR revision `ef198757da7cf7d995a9c5be447c4366ce734d49`: 511 passed, 34 deselected; both fixture tests passed |
| User-reported local run | [Standalone execution record](standalone-synthetic-execution.md) | At `50303396f69f022d076f7e7fcf78470c7a362985`: two test executions passed, five cases matched, and a full SHA-256 digest was supplied |
| Artifact-level independent verification | Not yet available | Original JSON file was not uploaded and independently hashed; no authentic execution screenshot was supplied |

User-reported SHA-256 for local `synthetic-demo-output.json`:

```text
39CA231F56920BDBB665B444028E4E18633EE251252D08F89D00433E197ECC9D
```

This hash is **not** a separately verified GitHub Actions artifact digest. The earlier [CI evidence page](synthetic-ci-evidence.md) records the state *at the time it was authored*, before the subsequent local execution was reported; the [standalone record](standalone-synthetic-execution.md) documents that later report.

## Reproduce the fixture

From the repository root in a compatible Python environment:

```powershell
python synthetic_verifier_demo.py > synthetic-demo-output.json
python -m pytest test_synthetic_verifier_demo.py -q
git rev-parse HEAD
(Get-FileHash .\synthetic-demo-output.json -Algorithm SHA256).Hash
Get-Content .\synthetic-demo-output.json
```

Record the exact commit and full digest; output bytes can differ across encodings/newline conventions. For independent artifact verification, retain the original JSON file and rehash those exact bytes.

## Technical significance

1. **Evidence integrity:** a present citation is not necessarily valid or retrieved.
2. **Explicit abstention:** ambiguous evidence maps to human review rather than automatic release.
3. **Fail-closed behavior:** contradicted, fabricated, and unretrieved evidence maps to BLOCK.
4. **Traceability:** source revision, test run, fixture expectations, and provenance are distinguishable.
5. **Responsible claims:** synthetic fixture results are not represented as live-model or clinical validation.

## Limitations and next validation gates

- Fake NLI scores are authored inputs, **not** measured model confidence.
- The five fixtures do not measure sensitivity, specificity, calibration, real-world retrieval quality, or generalization.
- The walkthrough does not demonstrate a live end-to-end pipeline, prospective evaluation, independent scientific review, or clinical authorization.
- Before any real-world application, separate model qualification, independent evaluation, privacy/security controls, domain-specific risk assessment, and appropriate governance are required.

**Portfolio summary:** RAG Verifier demonstrates how a Python verification layer can distinguish supported, uncertain, and invalid evidence and map these outcomes to governed software release decisions, with reproducible synthetic tests and clearly attributed evidence provenance.
