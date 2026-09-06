import hashlib
import hmac
import json

from healthcare_embedding_qualification import (
    HealthcareEmbeddingQualificationRunner,
)
from healthcare_retrieval_benchmark import (
    HealthcareRetrievalBenchmark,
)


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
        HealthcareEmbeddingQualificationRunner._validate_chunks(
            payload["chunks"]
        )
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
        self._payload_sha256 = digest

    @property
    def payload_sha256(self):
        return self._payload_sha256

    def payload(self):
        return json.loads(self._payload_json)

    def qualify(self, runner):
        if not isinstance(runner, HealthcareEmbeddingQualificationRunner):
            raise ValueError(
                "runner must be HealthcareEmbeddingQualificationRunner"
            )

        payload = self.payload()
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

        record = runner.qualify(
            payload["model"],
            payload["chunks"],
            payload["queries"],
        )
        return {
            "manifest_sha256": self.payload_sha256,
            "qualification_record": record,
        }
