import json

from embedding_provider import (
    LoadedModelDescriptor,
    SentenceTransformerEmbeddingProvider,
)
from healthcare_embedding_qualification import (
    HealthcareEmbeddingQualificationRunner,
    LoadedEmbeddingProvider,
)
from healthcare_qualification_manifest import (
    VersionedHealthcareQualificationManifest,
)
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


def _build_sentence_transformer_runner(manifest, provider_class):
    if not isinstance(manifest, VersionedHealthcareQualificationManifest):
        raise ValueError("manifest must be VersionedHealthcareQualificationManifest")
    payload = manifest.payload()
    model = payload["model"]
    if model["provider_type"] != "sentence-transformer":
        raise ValueError("manifest provider_type is not sentence-transformer")

    def provider_factory(candidate):
        provider = provider_class(model_id=candidate["model_id"], revision=candidate["model_revision"])
        descriptor = getattr(provider, "loaded_model_descriptor", None)
        if not isinstance(descriptor, LoadedModelDescriptor):
            raise ValueError("provider did not return a trusted loaded-model descriptor")
        observed_model = {
            "model_id": descriptor.model_id,
            "model_revision": descriptor.model_revision,
            "provider_type": descriptor.provider_type,
            "embedding_dimension": descriptor.embedding_dimension,
        }
        if observed_model != candidate:
            raise ValueError("loaded model identity does not match manifest")
        return LoadedEmbeddingProvider(
            provider=provider,
            **observed_model,
        )

    policy = payload["policy"]
    return HealthcareEmbeddingQualificationRunner(
        provider_factory=provider_factory,
        policy=ModelQualificationPolicy(pass_thresholds=policy["pass_thresholds"], review_thresholds=policy["review_thresholds"]),
        benchmark_id=payload["benchmark_id"],
        benchmark_version=payload["benchmark_version"],
        top_k=payload["top_k"],
    )


def build_sentence_transformer_runner(manifest):
    return _build_sentence_transformer_runner(
        manifest,
        SentenceTransformerEmbeddingProvider,
    )


def qualify_sentence_transformer_manifest(path):
    manifest = load_qualification_manifest(path)
    return manifest.qualify(build_sentence_transformer_runner(manifest))
