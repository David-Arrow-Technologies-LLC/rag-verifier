# Synthetic verifier — CI execution evidence (PR #33)

## Provenance

- Source repository: `David-Arrow-Technologies-LLC/rag-verifier`
- Tested PR head commit: `ef198757da7cf7d995a9c5be447c4366ce734d49`
- Merge commit on `main`: `ceb878e26f7e06d40d50feeb34b1995e84eaddfc`
- GitHub Actions run: [RAG Verifier Unit Gate #44948210193](https://github.com/David-Arrow-Technologies-LLC/rag-verifier/actions/runs/44948210193)
- Job ID: `136980112694`
- Job conclusion: `success`
- Evidence source: GitHub Actions job log, retrieved and inspected on 2026-11-21.

## Observed test evidence

The workflow executed `python -m pytest -m "not integration" -v`. The job log includes:

```text
test_synthetic_verifier_demo.py::test_synthetic_demo_expected_decisions PASSED [ 92%]
test_synthetic_verifier_demo.py::test_synthetic_demo_deterministic PASSED [ 93%]
====================== 511 passed, 34 deselected in 0.54s ======================
```

The same PR head also had successful NLI Integration, MiniLM Integration and Unified Release Qualification workflows. These are **separate gates**; their success does not imply clinical validity.

## Interpretation

The deterministic tests exercised the five expected synthetic outcomes in `synthetic_verifier_demo.py` via `run_cases()` and checked repeatability. This is **real CI test evidence**, not a synthetic JSON transcript and not an independent clinical benchmark.

**Not yet captured:** a separately executed `python synthetic_verifier_demo.py` stdout JSON file, its SHA-256 digest, and an authentic screenshot of that standalone execution. Do not fabricate these artifacts or imply they exist.

## Next reproducible evidence step

On the exact checked-out revision, run:

```bash
python synthetic_verifier_demo.py > synthetic-demo-output.json
python -m pytest test_synthetic_verifier_demo.py -q
git rev-parse HEAD
sha256sum synthetic-demo-output.json
```

Capture the console session as an actual screenshot, redact any unrelated local identifiers, and publish the JSON, test transcript, exact SHA, and hash only after verifying their provenance.

**Boundaries:** Synthetic fixture and FakeNLIProvider only; no real NLI performance measurement, patient data, clinical validation, or patient-care authorization.
