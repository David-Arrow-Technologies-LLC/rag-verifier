import hashlib
import json

import pytest

from embedding_provider import FakeEmbeddingProvider
from healthcare_embedding_qualification import (
    HealthcareEmbeddingQualificationRunner,
    LoadedEmbeddingProvider,
)
from healthcare_qualification_manifest import (
    VersionedHealthcareQualificationManifest,
)
from model_qualification import ModelQualificationPolicy


def _payload():
    return {
        "schema_version": 1,
        "benchmark_id": "healthcare-retrieval",
        "benchmark_version": "healthcare-mini-v1",
        "top_k": 1,
        "model": {
            "model_id": "example/healthcare-embedding-model",
            "model_revision": "revision-001",
            "provider_type": "sentence-transformer",
            "embedding_dimension": 2,
        },
        "chunks": {
            "DOC-HTN": "Hypertension is elevated arterial blood pressure.",
        },
        "queries": [
            {
                "query_id": "Q-HTN",
                "text": "What condition involves elevated blood pressure?",
                "relevant_ids": ["DOC-HTN"],
            }
        ],
    }


def _document(payload=None):
    payload = payload or _payload()
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }


def _runner(factory):
    policy = ModelQualificationPolicy(
        pass_thresholds={
            "recall_at_k": 1.0,
            "mrr": 1.0,
            "precision_at_k": 1.0,
        },
        review_thresholds={
            "recall_at_k": 0.5,
            "mrr": 0.5,
            "precision_at_k": 0.5,
        },
    )
    return HealthcareEmbeddingQualificationRunner(
        provider_factory=factory,
        policy=policy,
        benchmark_id="healthcare-retrieval",
        benchmark_version="healthcare-mini-v1",
        top_k=1,
    )


def _loaded_provider():
    payload = _payload()
    provider = FakeEmbeddingProvider()
    provider.set_embedding(
        payload["chunks"]["DOC-HTN"],
        [1.0, 0.0],
    )
    provider.set_embedding(
        payload["queries"][0]["text"],
        [1.0, 0.0],
    )
    return LoadedEmbeddingProvider(
        provider=provider,
        **payload["model"],
    )


def test_manifest_qualifies_and_binds_digest_to_record():
    document = _document()
    manifest = VersionedHealthcareQualificationManifest(document)

    result = manifest.qualify(
        _runner(lambda model: _loaded_provider())
    )

    assert result["manifest_sha256"] == document["payload_sha256"]
    assert result["qualification_record"]["qualification"]["status"] == "PASS"


def test_manifest_rejects_payload_tampering():
    document = _document()
    document["payload"]["benchmark_version"] = "tampered"

    with pytest.raises(ValueError, match="payload digest mismatch"):
        VersionedHealthcareQualificationManifest(document)


def test_manifest_rejects_unknown_relevance_id():
    payload = _payload()
    payload["queries"][0]["relevant_ids"] = ["DOC-MISSING"]

    with pytest.raises(
        ValueError,
        match="relevant_ids must exist in the chunk corpus",
    ):
        VersionedHealthcareQualificationManifest(_document(payload))


def test_runner_mismatch_fails_before_provider_loading():
    calls = []
    runner = _runner(lambda model: calls.append(model))
    runner.top_k = 2
    manifest = VersionedHealthcareQualificationManifest(_document())

    with pytest.raises(
        ValueError,
        match="runner configuration does not match manifest",
    ):
        manifest.qualify(runner)

    assert calls == []


def test_manifest_payload_returns_isolated_copy():
    manifest = VersionedHealthcareQualificationManifest(_document())
    first = manifest.payload()
    first["model"]["model_id"] = "tampered"

    assert manifest.payload()["model"]["model_id"] == (
        "example/healthcare-embedding-model"
    )


@pytest.mark.parametrize("schema_version", [True, 0, 2, "1"])
def test_manifest_rejects_unsupported_schema_version(schema_version):
    payload = _payload()
    payload["schema_version"] = schema_version

    with pytest.raises(
        ValueError,
        match="unsupported manifest schema_version",
    ):
        VersionedHealthcareQualificationManifest(_document(payload))
