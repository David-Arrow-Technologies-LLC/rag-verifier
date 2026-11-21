# Synthetic RAG Verifier demonstration

This demonstration exercises the **existing** `RAGVerifier` and `ReleasePolicy` with `FakeNLIProvider` and entirely fictional sample-storage reference text. It is not an end-to-end live retrieval, generative-model, or real-NLI benchmark.

## Run

From the repository root, with a compatible Python environment:

```bash
python synthetic_verifier_demo.py > synthetic-demo-output.json
python -m pytest test_synthetic_verifier_demo.py -q
git rev-parse HEAD
sha256sum synthetic-demo-output.json
```

Record the exact checked-out commit SHA and output SHA-256 alongside the JSON if publishing an execution result. The output file is generated locally; **no execution result, screenshot, digest, or CI PASS is asserted by this documentation**.

## Fixture cases

| Case | Simulated evidence behavior | Expected verifier | Expected policy |
| --- | --- | --- | --- |
| supported | Fake NLI entailment 0.98 | PASS | RELEASE |
| uncertain | Fake NLI neutral 0.55 | REVIEW | HOLD_FOR_REVIEW |
| contradicted | Fake NLI contradiction 0.98 | FAIL | BLOCK |
| fabricated_citation | Citation absent from source dictionary | FAIL | BLOCK |
| unretrieved_citation | Citation exists but was not retrieved | FAIL | BLOCK |

`RELEASE` here means **only** that the engineering release-policy function returns its release action for a synthetic fixture. It does not mean clinical release, authorization, deployment, or real-world trustworthiness.

## Evidence interpretation

- This is a deterministic demonstration of branching and fail-closed handling.
- The scores are authored fixture inputs, **not** measured model confidence or accuracy.
- No patient data, medical treatment advice, licensed medical documents, or live model calls are involved.
- Tests must execute at the exact revision before describing them as passed.
- A screenshot for the public portfolio should be captured from an actual run of the script, not a fabricated dashboard.
