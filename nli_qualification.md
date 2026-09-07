# Governed real-NLI qualification

`nli_deberta_qualification.json` binds the NLI model, immutable Hugging Face
revision, observed label order, verifier decision thresholds, synthetic medical
cases, expected outcomes, qualification thresholds, and output record to a
canonical SHA-256 digest.

The cases are repository-authored engineering fixtures, not clinical validation
data and not evidence of fitness for clinical use. Deterministic tests use a fake
provider; the dedicated NLI integration gate loads the real pinned model.
