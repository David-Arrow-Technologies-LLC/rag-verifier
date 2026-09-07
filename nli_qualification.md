# Governed real-NLI qualification

`nli_deberta_qualification.json` binds the NLI model, immutable Hugging Face
revision, observed label order, verifier decision thresholds, synthetic medical
cases, expected outcomes, qualification thresholds, and output record to a
canonical SHA-256 digest.

The cases are repository-authored engineering fixtures, not clinical validation
data and not evidence of fitness for clinical use. Deterministic tests use a fake
provider; the dedicated NLI integration gate loads the real pinned model.

Every successful integration run also emits a digest-bound JSON evidence artifact
containing the exact source revision, manifest digest, model identity, runtime
package versions, aggregate metrics, and per-case expected and observed outcomes.
The gate derives the source revision from a clean, explicitly checked-out PR head,
recomputes aggregate metrics from the retained case evidence, and fails on the same
qualification execution that produced the artifact. GitHub Actions retains that
artifact for 90 days.
