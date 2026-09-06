# Governed MiniLM healthcare integration qualification

`healthcare_minilm_qualification.json` exercises the complete healthcare embedding qualification path with `sentence-transformers/all-MiniLM-L6-v2` at an immutable Hub commit.

The six corpus entries and queries are repository-authored synthetic engineering fixtures. They contain no patient data and are not an approved clinical corpus, clinical evidence source, or evidence of fitness for clinical use. Passing this benchmark qualifies only the pinned model/runtime integration contract.

Deterministic CI validates manifest integrity, exact model arguments, and the
immutable post-load descriptor without downloading an ML model. A dedicated PR
integration gate loads the exact pinned model, verifies its observed dimension,
and requires the governed qualification record to return `PASS`.

```bash
pytest -m integration test_healthcare_real_model_qualification.py -v
```
