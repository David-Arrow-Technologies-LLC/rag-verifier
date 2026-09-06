import json

from embedding_provider import SentenceTransformerEmbeddingProvider
from healthcare_embedding_qualification import HealthcareEmbeddingQualificationRunner, LoadedEmbeddingProvider
from healthcare_qualification_manifest import VersionedHealthcareQualificationManifest
from model_qualification import ModelQualificationPolicy


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_qualification_manifest(path):
    with open(path, encoding="utf-8") as manifest_file:
        document = json.load(manifest_file, object_pairs_hook=_reject_duplicate_keys)
    return VersionedHealthcareQualificationManifest(document)


def build_sentence_transformer_runner(manifest, provider_class=SentenceTransformerEmbeddingProvider):
    if not isinstance(manifest, VersionedHealthcareQualificationManifest):
        raise ValueError("manifest must be VersionedHealthcareQualificationManifest")
    payload = manifest.payload()
    model = payload["model"]
    if model["provider_type"] != "sentence-transformer":
        raise ValueError("manifest provider_type is not sentence-transformer")

    def provider_factory(candidate):
        provider = provider_class(model_id=candidate["model_id"], revision=candidate["model_revision"])
        dimension_reader = getattr(getattr(provider, "model", None), "get_sentence_embedding_dimension", None)
        if not callable(dimension_reader):
            raise ValueError("loaded model does not report embedding dimension")
        observed_dimension = dimension_reader()
        if isinstance(observed_dimension, bool) or not isinstance(observed_dimension, int) or observed_dimension != candidate["embedding_dimension"]:
            raise ValueError("loaded model embedding dimension mismatch")
        return LoadedEmbeddingProvider(
            provider=provider,
            model_id=candidate["model_id"],
            model_revision=candidate["model_revision"],
            provider_type=candidate["provider_type"],
            embedding_dimension=observed_dimension,
        )

    policy = payload["policy"]
    return HealthcareEmbeddingQualificationRunner(
        provider_factory=provider_factory,
        policy=ModelQualificationPolicy(pass_thresholds=policy["pass_thresholds"], review_thresholds=policy["review_thresholds"]),
        benchmark_id=payload["benchmark_id"],
        benchmark_version=payload["benchmark_version"],
        top_k=payload["top_k"],
    )


def qualify_sentence_transformer_manifest(path, provider_class=None):
    manifest = load_qualification_manifest(path)
    kwargs = {} if provider_class is None else {"provider_class": provider_class}
    return manifest.qualify(build_sentence_transformer_runner(manifest, **kwargs))
