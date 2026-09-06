import pytest

from healthcare_real_model_qualification import build_sentence_transformer_runner, load_qualification_manifest


MANIFEST_PATH = "healthcare_minilm_qualification.json"


class _ObservedModel:
    def __init__(self, dimension):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self):
        return self.dimension


class _DeterministicProvider:
    calls = []

    def __init__(self, model_id, revision):
        type(self).calls.append((model_id, revision))
        self.model = _ObservedModel(384)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return [[1.0] + [0.0] * 383 for _ in texts]

    encode_queries = encode
    encode_documents = encode


def test_repository_manifest_is_valid_and_pins_minilm():
    model = load_qualification_manifest(MANIFEST_PATH).payload()["model"]
    assert model["model_id"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert model["model_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert model["embedding_dimension"] == 384


def test_runner_passes_exact_pinned_identity_to_provider():
    _DeterministicProvider.calls = []
    manifest = load_qualification_manifest(MANIFEST_PATH)
    runner = build_sentence_transformer_runner(manifest, _DeterministicProvider)
    runner.provider_factory(manifest.payload()["model"])
    assert _DeterministicProvider.calls == [("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")]


def test_runner_rejects_observed_dimension_mismatch():
    class WrongDimensionProvider(_DeterministicProvider):
        def __init__(self, model_id, revision):
            self.model = _ObservedModel(768)

    manifest = load_qualification_manifest(MANIFEST_PATH)
    runner = build_sentence_transformer_runner(manifest, WrongDimensionProvider)
    with pytest.raises(ValueError, match="embedding dimension mismatch"):
        runner.provider_factory(manifest.payload()["model"])


def test_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"payload": {}, "payload": {}, "payload_sha256": "x"}')
    with pytest.raises(ValueError, match="duplicate JSON key: payload"):
        load_qualification_manifest(path)


@pytest.mark.integration
def test_pinned_minilm_qualifies_through_governed_manifest():
    manifest = load_qualification_manifest(MANIFEST_PATH)
    result = manifest.qualify(build_sentence_transformer_runner(manifest))
    assert result["manifest_sha256"] == manifest.payload_sha256
    assert result["qualification_record"]["qualification"]["status"] == "PASS"
