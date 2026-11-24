# Offline annotation infrastructure

Research-only Python API in annotation.py. No real reviewer records, approvals, frozen manifest or inference are created by this slice.

1. Coordinator runs the existing candidate audit, loads the candidate JSON, and calls prepare(candidate). Save the returned packet and private mapping separately in access-controlled storage. Distribute only the packet; do not share candidate source, mapping, answer key or model outputs. Opaque randomized IDs avoid the label suffix in original case IDs. Public repository access may compromise blinding; document actual exposure.
2. Each reviewer independently submits a JSON object with reviewer_id, packet_digest (canonical SHA-256 via digest(packet)), and annotations containing item_id, label (entailment/neutral/contradiction), rationale. Decisions are derived from the existing label/status contract, not independently measured.
3. agreement(packet, first, second) validates coverage and computes raw agreement, Cohen's kappa (null if undefined), and disagreements.
4. adjudicate(packet, first, second, record) requires adjudicator_id, packet_digest, review_digests in input order, and resolutions with item_id, label, rationale for every disagreement. Preserve all original records and private mapping. Output is a proposal, not approval.
5. A future separately authorized slice must validate the private mapping against the exact candidate, authenticate reviewer credentials and attestations, review criticality and semantic overlap, verify balance after adjudication, and bind approved evidence to a versioned manifest. No function in this slice can authorize freeze or inference.

Limitations: distinct strings do not prove distinct humans; hashes are not signatures; public source may expose labels; independent annotation cannot repair prior non-independent authorship. Tests use fictional identities and labels only. No clinical use.
