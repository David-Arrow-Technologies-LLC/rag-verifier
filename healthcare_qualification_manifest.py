import hashlib
import hmac
import json
import math
import re

from healthcare_embedding_qualification import (
    HealthcareEmbeddingQualificationRunner,
)
from healthcare_retrieval_benchmark import (
    HealthcareRetrievalBenchmark,
)
from model_qualification import ModelQualificationPolicy


def canonical_payload(payload):
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def payload_sha256(payload):
    return hashlib.sha256(
        canonical_payload(payload).encode("utf-8")
    ).hexdigest()


class VersionedHealthcareQualificationManifest:
    REQUIRED_PAYLOAD_FIELDS = {
        "schema_version",
        "benchmark_id",
        "benchmark_version",
        "top_k",
        "model",
        "policy",
        "chunks",
        "queries",
    }

    def __init__(self, document):
        if not isinstance(document, dict):
            raise ValueError("manifest document must be a dict")
        if set(document) != {"payload", "payload_sha256"}:
            raise ValueError("manifest document fields are invalid")

        payload = document["payload"]
        digest = document["payload_sha256"]
        if not isinstance(payload, dict):
            raise ValueError("manifest payload must be a dict")
        if set(payload) != self.REQUIRED_PAYLOAD_FIELDS:
            raise ValueError("manifest payload fields are invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("payload_sha256 must be a lowercase SHA-256 digest")

        observed_digest = payload_sha256(payload)
        if not hmac.compare_digest(digest, observed_digest):
            raise ValueError("manifest payload digest mismatch")

        if (
            isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or payload["schema_version"] != 1
        ):
            raise ValueError("unsupported manifest schema_version")

        HealthcareEmbeddingQualificationRunner._validate_model(
            payload["model"]
        )
        if set(payload["model"]) != {
            "model_id",
            "model_revision",
            "provider_type",
            "embedding_dimension",
        }:
            raise ValueError("manifest model fields are invalid")
        if (
            payload["model"]["provider_type"] == "sentence-transformer"
            and re.fullmatch(
                r"[0-9a-f]{40}",
                payload["model"]["model_revision"],
            ) is None
        ):
            raise ValueError(
                "sentence-transformer model_revision must be a "
                "40-character lowercase commit SHA"
            )

        policy = payload["policy"]
        if not isinstance(policy, dict) or set(policy) != {
            "pass_thresholds",
            "review_thresholds",
        }:
            raise ValueError("manifest policy fields are invalid")
        ModelQualificationPolicy(
            pass_thresholds=policy["pass_thresholds"],
            review_thresholds=policy["review_thresholds"],
        )
        supported_metrics = {
            "recall_at_k",
            "mrr",
            "precision_at_k",
        }
        if set(policy["pass_thresholds"]) != supported_metrics:
            raise ValueError("manifest policy metrics are invalid")

        HealthcareEmbeddingQualificationRunner._validate_chunks(
            payload["chunks"]
        )
        for query in payload["queries"]:
            if not isinstance(query, dict) or set(query) != {
                "query_id",
                "text",
                "relevant_ids",
            }:
                raise ValueError("manifest query fields are invalid")
        HealthcareRetrievalBenchmark.validate_queries(
            payload["queries"],
            corpus_ids=payload["chunks"].keys(),
        )

        for field in ("benchmark_id", "benchmark_version"):
            value = payload[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")

        top_k = payload["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        self._payload_json = canonical_payload(payload)

    @property
    def payload_sha256(self):
        return hashlib.sha256(
            self._payload_json.encode("utf-8")
        ).hexdigest()

    def payload(self):
        return json.loads(self._payload_json)

    def qualify(self, runner):
        if not isinstance(runner, HealthcareEmbeddingQualificationRunner):
            raise ValueError(
                "runner must be HealthcareEmbeddingQualificationRunner"
            )

        payload = self.payload()
        for field in ("benchmark_id", "benchmark_version"):
            value = getattr(runner, field, None)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("runner configuration is invalid")
        if (
            isinstance(runner.top_k, bool)
            or not isinstance(runner.top_k, int)
            or runner.top_k <= 0
        ):
            raise ValueError("runner configuration is invalid")

        expected_runner_contract = (
            payload["benchmark_id"],
            payload["benchmark_version"],
            payload["top_k"],
        )
        observed_runner_contract = (
            runner.benchmark_id,
            runner.benchmark_version,
            runner.top_k,
        )
        if observed_runner_contract != expected_runner_contract:
            raise ValueError(
                "runner configuration does not match manifest"
            )

        runner_policy = getattr(
            getattr(runner, "record_builder", None),
            "policy",
            None,
        )
        policy = payload["policy"]
        if (
            not isinstance(runner_policy, ModelQualificationPolicy)
            or runner_policy.pass_thresholds != policy["pass_thresholds"]
            or runner_policy.review_thresholds != policy["review_thresholds"]
        ):
            raise ValueError("runner policy does not match manifest")

        manifest_policy = ModelQualificationPolicy(
            pass_thresholds=policy["pass_thresholds"],
            review_thresholds=policy["review_thresholds"],
        )

        record = runner.qualify(
            payload["model"],
            payload["chunks"],
            payload["queries"],
        )
        if (
            runner_policy.pass_thresholds != policy["pass_thresholds"]
            or runner_policy.review_thresholds != policy["review_thresholds"]
        ):
            raise ValueError("runner policy changed during qualification")
        self._validate_record(record, payload, manifest_policy)
        return {
            "manifest_sha256": self.payload_sha256,
            "qualification_record": record,
        }

    @staticmethod
    def _validate_record(record, payload, policy):
        if not isinstance(record, dict) or set(record) != {
            "model",
            "benchmark",
            "metrics",
            "qualification",
        }:
            raise ValueError("qualification record is invalid")

        if record["model"] != payload["model"]:
            raise ValueError("qualification record model mismatch")

        expected_benchmark = {
            "benchmark_id": payload["benchmark_id"],
            "benchmark_version": payload["benchmark_version"],
            "top_k": payload["top_k"],
            "query_count": len(payload["queries"]),
        }
        if record["benchmark"] != expected_benchmark:
            raise ValueError("qualification record benchmark mismatch")

        metrics = record["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != {
            "recall_at_k",
            "mrr",
            "precision_at_k",
        }:
            raise ValueError("qualification record metrics are invalid")
        for value in metrics.values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
                or value > 1.0
            ):
                raise ValueError("qualification record metrics are invalid")

        expected_qualification = policy.decide(metrics)
        if record["qualification"] != expected_qualification:
            raise ValueError("qualification record decision mismatch")
