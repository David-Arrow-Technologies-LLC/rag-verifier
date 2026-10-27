# Decimal periods in sentence boundaries

The existing RAGVerifier.split_claims sentence heuristic preserves a period between Unicode decimal digits (Python regex `\d`) within the same hypothesis. For example, `Synthetic value is 9.0. [C1]` is one sentence with its citation attached; the final period still ends the sentence. Signed and multi-digit decimal values retain their original text. Other periods, exclamation marks, question marks, trailing citation markers and whitespace normalization retain the existing boundary behavior.

This narrow repair does not implement a numeric parser or general sentence tokenizer. It does not resolve abbreviations, URLs, leading-dot number notation, version strings, or clinical proposition eligibility. Existing conjunction decomposition, citation checks, NLI thresholds, result envelopes and answer aggregation remain unchanged. Producer PASS/trusted_release is not K2A release authorization or clinical validation.

Sixteen deterministic regression cases cover decimal hypotheses, terminal and multiple citations, adjacent sentences, unterminated text, exact FakeNLI hypothesis dispatch, PASS/FAIL/REVIEW, repeated results, and uncited/unknown/unretrieved citation failures. All sixteen fail against the previous splitter. No real model is used.

DA Health's historical source-bound exports and evidence receipts remain pinned to their original producer revision. After this repair merges, a separate DA PR must deliberately bind the new producer revision/blob, rerun existing characterization and issue new evidence. Do not relabel earlier receipts or change DA Diagnostics runtime to consume this patch.
