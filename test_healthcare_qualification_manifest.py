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
            "model_revision": "1" * 40,
            "provider_type": "sentence-transformer",
            "embedding_dimension": 2,
        },
        "policy": {
            "pass_thresholds": {
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "precision_at_k": 1.0,
            },
            "review_thresholds": {
                "recall_at_k": 0.5,
                "mrr": 0.5,
                "precision_at_k": 0.5,
            },
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
    manifest_policy = _payload()["policy"]
    policy = ModelQualificationPolicy(
        pass_thresholds=manifest_policy["pass_thresholds"],
        review_thresholds=manifest_policy["review_thresholds"],
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


def test_verified_manifest_digest_is_read_only():
    manifest = VersionedHealthcareQualificationManifest(_document())

    with pytest.raises(AttributeError):
        manifest.payload_sha256 = "0" * 64


@pytest.mark.parametrize("schema_version", [True, 0, 2, "1"])
def test_manifest_rejects_unsupported_schema_version(schema_version):
    payload = _payload()
    payload["schema_version"] = schema_version

    with pytest.raises(
        ValueError,
        match="unsupported manifest schema_version",
    ):
        VersionedHealthcareQualificationManifest(_document(payload))


def test_manifest_rejects_policy_mismatch_before_provider_loading():
    calls = []
    runner = _runner(lambda model: calls.append(model))
    runner.record_builder.policy = ModelQualificationPolicy(
        pass_thresholds={
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "precision_at_k": 0.0,
        },
        review_thresholds={
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "precision_at_k": 0.0,
        },
    )
    manifest = VersionedHealthcareQualificationManifest(_document())

    with pytest.raises(ValueError, match="runner policy does not match manifest"):
        manifest.qualify(runner)

    assert calls == []


@pytest.mark.parametrize("invalid_top_k", [True, 1.0])
def test_invalid_runner_top_k_fails_before_provider_loading(invalid_top_k):
    calls = []
    runner = _runner(lambda model: calls.append(model))
    runner.top_k = invalid_top_k
    manifest = VersionedHealthcareQualificationManifest(_document())

    with pytest.raises(ValueError, match="runner configuration is invalid"):
        manifest.qualify(runner)

    assert calls == []


@pytest.mark.parametrize("nested_object", ["model", "query"])
def test_manifest_rejects_unsupported_nested_fields(nested_object):
    payload = _payload()
    if nested_object == "model":
        payload["model"]["model_revison"] = "misspelled"
    else:
        payload["queries"][0]["relevnt_ids"] = ["DOC-HTN"]

    with pytest.raises(ValueError, match="manifest (model|query) fields are invalid"):
        VersionedHealthcareQualificationManifest(_document(payload))


@pytest.mark.parametrize("revision", ["main", "v1.0", "A" * 40, "a" * 39])
def test_sentence_transformer_revision_must_be_immutable_commit(revision):
    payload = _payload()
    payload["model"]["model_revision"] = revision

    with pytest.raises(
        ValueError,
        match="model_revision must be a 40-character lowercase commit SHA",
    ):
        VersionedHealthcareQualificationManifest(_document(payload))


def test_manifest_rejects_substituted_qualification_record():
    class SubstitutedRunner(HealthcareEmbeddingQualificationRunner):
        def qualify(self, model, chunks, queries):
            return {
                "model": dict(model, model_id="different/model"),
                "benchmark": {
                    "benchmark_id": self.benchmark_id,
                    "benchmark_version": self.benchmark_version,
                    "top_k": self.top_k,
                    "query_count": len(queries),
                },
                "metrics": {
                    "recall_at_k": 1.0,
                    "mrr": 1.0,
                    "precision_at_k": 1.0,
                },
                "qualification": {
                    "status": "PASS",
                    "reason": "QUALIFICATION_THRESHOLDS_SATISFIED",
                },
            }

    base = _runner(lambda model: _loaded_provider())
    runner = SubstitutedRunner(
        provider_factory=base.provider_factory,
        policy=base.record_builder.policy,
        benchmark_id=base.benchmark_id,
        benchmark_version=base.benchmark_version,
        top_k=base.top_k,
    )
    manifest = VersionedHealthcareQualificationManifest(_document())

    with pytest.raises(ValueError, match="qualification record model mismatch"):
        manifest.qualify(runner)


def test_manifest_rejects_unsupported_policy_metric():
    payload = _payload()
    payload["policy"]["pass_thresholds"]["unsupported"] = 0.5
    payload["policy"]["review_thresholds"]["unsupported"] = 0.25

    with pytest.raises(ValueError, match="manifest policy metrics are invalid"):
        VersionedHealthcareQualificationManifest(_document(payload))


def test_manifest_rejects_policy_mutation_during_qualification():
    class PolicyMutatingRunner(HealthcareEmbeddingQualificationRunner):
        def qualify(self, model, chunks, queries):
            record = super().qualify(model, chunks, queries)
            self.record_builder.policy.pass_thresholds["mrr"] = 0.0
            return record

    base = _runner(lambda model: _loaded_provider())
    runner = PolicyMutatingRunner(
        provider_factory=base.provider_factory,
        policy=base.record_builder.policy,
        benchmark_id=base.benchmark_id,
        benchmark_version=base.benchmark_version,
        top_k=base.top_k,
    )

    with pytest.raises(
        ValueError,
        match="runner policy changed during qualification",
    ):
        VersionedHealthcareQualificationManifest(_document()).qualify(runner)
