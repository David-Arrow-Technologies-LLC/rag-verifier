# Governed MiniLM healthcare integration qualification

`healthcare_minilm_qualification.json` exercises the complete healthcare embedding qualification path with `sentence-transformers/all-MiniLM-L6-v2` at an immutable Hub commit.

The six corpus entries and queries are repository-authored synthetic engineering fixtures. They contain no patient data and are not an approved clinical corpus, clinical evidence source, or evidence of fitness for clinical use. Passing this benchmark qualifies only the pinned model/runtime integration contract.

Deterministic CI validates manifest integrity, exact model arguments, and observed embedding dimension without downloading an ML model. The test marked `integration` loads the real pinned model and remains excluded from the deterministic unit workflow.

```bash
pytest -m integration test_healthcare_real_model_qualification.py -v
```
