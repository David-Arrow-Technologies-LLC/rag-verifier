# Adversarial document and prompt-injection qualification

RAG-V30 establishes a deterministic, versioned security boundary between retrieved
documents and prompt construction. Retrieved text is untrusted data. A document
that contains a recognized instruction-override, role-boundary injection,
secret-exfiltration request, tool command, citation bypass, or authority
impersonation is blocked before it can enter the generated context.

The policy is intentionally fail closed:

- malformed, duplicate-key, digest-mismatched, non-finite, unsupported-version,
  incomplete-category, and unknown-rule manifests are rejected;
- every attack case must produce the exact expected decision and ordered rule set;
- benign clinical documents are included to detect overblocking regressions;
- qualification evidence is canonical and bound to the source revision, implementation SHA-256, and corpus SHA-256;
- the approved corpus digest is additionally pinned by the repository-owned supply-chain policy, preventing corpus-only rehashing from silently qualifying;
- CI retains the generated qualification evidence for 90 days;
- a failed case produces an overall `FAIL` qualification.

The repository policy pin is separate from the manifest's self-digest, but it is
not an independent change-authority trust root: both the corpus and repository
policy remain governed by the repository change process. A cryptographically or
administratively independent corpus approval root must therefore be supplied by
an external governance boundary if that stronger property is required. RAG-V30
does not claim that property.

Run the deterministic qualification with:

```bash
python run_adversarial_qualification.py \
  --manifest adversarial_qualification_manifest.json \
  --output qualification-evidence/rag-v30-adversarial.json \
  --source-revision <40-character-git-sha>
```

This control is defense in depth, not a claim that pattern matching can recognize
every possible prompt injection. New attack forms must be added as immutable
qualification cases before policy expansion. Model-level and end-to-end red-team
testing remain required for deployment qualification, especially in healthcare.

RAG-V30 is an engineering security qualification. It does not establish clinical
validity or authorize autonomous clinical use.
