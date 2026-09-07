import hashlib
import hmac
import json
import re

from nli_provider import HuggingFaceNLIProvider, LoadedNLIModelDescriptor
from verifier import RAGVerifier


def canonical_payload(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def payload_sha256(value):
    return hashlib.sha256(canonical_payload(value).encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class VersionedNLIQualificationManifest:
    PAYLOAD_FIELDS = {"schema_version", "benchmark_id", "benchmark_version", "model", "policy", "cases", "qualification_thresholds"}
    MODEL_FIELDS = {"model_id", "model_revision", "provider_type", "labels"}
    POLICY_FIELDS = {"pass_threshold", "fail_threshold"}
    CASE_FIELDS = {"case_id", "premise", "hypothesis", "expected_label", "expected_status"}
    QUALIFICATION_FIELDS = {"minimum_label_accuracy", "minimum_decision_accuracy"}
    LABELS = {"entailment", "contradiction", "neutral"}
    STATUSES = {"PASS", "REVIEW", "FAIL"}

    def __init__(self, document):
        if not isinstance(document, dict) or set(document) != {"payload", "payload_sha256"}:
            raise ValueError("manifest document fields are invalid")
        payload = document["payload"]
        digest = document["payload_sha256"]
        if not isinstance(payload, dict) or set(payload) != self.PAYLOAD_FIELDS:
            raise ValueError("manifest payload fields are invalid")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("payload_sha256 must be a lowercase SHA-256 digest")
        if not hmac.compare_digest(digest, payload_sha256(payload)):
            raise ValueError("manifest payload digest mismatch")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported manifest schema_version")
        for field in ("benchmark_id", "benchmark_version"):
            if not isinstance(payload[field], str) or not payload[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
        self._validate_model(payload["model"])
        self._validate_policy(payload["policy"])
        self._validate_qualification_thresholds(payload["qualification_thresholds"])
        self._validate_cases(payload["cases"])
        self._payload_json = canonical_payload(payload)

    @classmethod
    def _validate_model(cls, model):
        if not isinstance(model, dict) or set(model) != cls.MODEL_FIELDS:
            raise ValueError("manifest model fields are invalid")
        if not isinstance(model["model_id"], str) or not model["model_id"].strip():
            raise ValueError("model_id must be a non-empty string")
        if re.fullmatch(r"[0-9a-f]{40}", model["model_revision"]) is None:
            raise ValueError("model_revision must be a 40-character lowercase commit SHA")
        if model["provider_type"] != "huggingface-sequence-classification":
            raise ValueError("unsupported NLI provider_type")
        if not isinstance(model["labels"], list) or len(model["labels"]) != 3 or set(model["labels"]) != cls.LABELS:
            raise ValueError("manifest model labels are invalid")

    @classmethod
    def _validate_policy(cls, policy):
        if not isinstance(policy, dict) or set(policy) != cls.POLICY_FIELDS:
            raise ValueError("manifest policy fields are invalid")
        for value in policy.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError("manifest policy thresholds are invalid")

    @classmethod
    def _validate_qualification_thresholds(cls, thresholds):
        if not isinstance(thresholds, dict) or set(thresholds) != cls.QUALIFICATION_FIELDS:
            raise ValueError("qualification threshold fields are invalid")
        for value in thresholds.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError("qualification thresholds are invalid")

    @classmethod
    def _validate_cases(cls, cases):
        if not isinstance(cases, list) or not cases:
            raise ValueError("cases must be a non-empty list")
        case_ids = set()
        for case in cases:
            if not isinstance(case, dict) or set(case) != cls.CASE_FIELDS:
                raise ValueError("manifest case fields are invalid")
            for field in ("case_id", "premise", "hypothesis"):
                if not isinstance(case[field], str) or not case[field].strip():
                    raise ValueError(f"{field} must be a non-empty string")
            if case["case_id"] in case_ids:
                raise ValueError("case_id values must be unique")
            case_ids.add(case["case_id"])
            if case["expected_label"] not in cls.LABELS:
                raise ValueError("expected_label is invalid")
            if case["expected_status"] not in cls.STATUSES:
                raise ValueError("expected_status is invalid")

    @property
    def payload_sha256(self):
        return hashlib.sha256(self._payload_json.encode("utf-8")).hexdigest()

    def payload(self):
        return json.loads(self._payload_json)

    def qualify(self, runner):
        if not isinstance(runner, NLIQualificationRunner):
            raise ValueError("runner must be NLIQualificationRunner")
        if runner.manifest_sha256 != self.payload_sha256:
            raise ValueError("runner does not match manifest")
        record = runner.run()
        self._validate_record(record)
        return {"manifest_sha256": self.payload_sha256, "qualification_record": record}

    def _validate_record(self, record):
        payload = self.payload()
        if not isinstance(record, dict) or set(record) != {"model", "benchmark", "policy", "case_results", "metrics", "qualification"}:
            raise ValueError("qualification record is invalid")
        if record["model"] != payload["model"] or record["policy"] != payload["policy"]:
            raise ValueError("qualification record identity mismatch")
        expected_benchmark = {"benchmark_id": payload["benchmark_id"], "benchmark_version": payload["benchmark_version"], "case_count": len(payload["cases"])}
        if record["benchmark"] != expected_benchmark:
            raise ValueError("qualification record benchmark mismatch")
        case_results = record["case_results"]
        if not isinstance(case_results, list) or len(case_results) != len(payload["cases"]):
            raise ValueError("qualification record case results are invalid")
        expected_cases = {case["case_id"]: case for case in payload["cases"]}
        policy_verifier = RAGVerifier(
            chunks={},
            retrieved_ids=set(),
            nli_provider=object(),
            pass_threshold=payload["policy"]["pass_threshold"],
            fail_threshold=payload["policy"]["fail_threshold"],
        )
        seen_case_ids = set()
        label_matches = decision_matches = 0
        for result in case_results:
            fields = {"case_id", "expected_label", "predicted_label", "expected_status", "observed_status", "scores"}
            if not isinstance(result, dict) or set(result) != fields:
                raise ValueError("qualification record case result is invalid")
            case_id = result["case_id"]
            if case_id in seen_case_ids or case_id not in expected_cases:
                raise ValueError("qualification record case identity is invalid")
            seen_case_ids.add(case_id)
            expected_case = expected_cases[case_id]
            if result["expected_label"] != expected_case["expected_label"] or result["expected_status"] != expected_case["expected_status"]:
                raise ValueError("qualification record expected outcome mismatch")
            if result["predicted_label"] not in self.LABELS or result["observed_status"] not in self.STATUSES:
                raise ValueError("qualification record observed outcome is invalid")
            scores = result["scores"]
            if not isinstance(scores, dict) or set(scores) != self.LABELS:
                raise ValueError("qualification record scores are invalid")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0 for value in scores.values()):
                raise ValueError("qualification record scores are invalid")
            validation = policy_verifier.validate_nli_scores(scores)
            if not validation["valid"]:
                raise ValueError("qualification record scores are invalid")
            normalized = validation["scores"]
            predicted_label = max(sorted(normalized), key=normalized.get)
            observed_status = policy_verifier.classify_nli_scores(normalized)["status"]
            if result["predicted_label"] != predicted_label or result["observed_status"] != observed_status:
                raise ValueError("qualification record case decision mismatch")
            label_matches += predicted_label == expected_case["expected_label"]
            decision_matches += observed_status == expected_case["expected_status"]
        metrics = record["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != {"label_accuracy", "decision_accuracy"}:
            raise ValueError("qualification record metrics are invalid")
        for value in metrics.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError("qualification record metrics are invalid")
        count = len(case_results)
        expected_metrics = {
            "label_accuracy": label_matches / count,
            "decision_accuracy": decision_matches / count,
        }
        if metrics != expected_metrics:
            raise ValueError("qualification record metrics do not match case results")
        thresholds = payload["qualification_thresholds"]
        passed = metrics["label_accuracy"] >= thresholds["minimum_label_accuracy"] and metrics["decision_accuracy"] >= thresholds["minimum_decision_accuracy"]
        expected = {"status": "PASS" if passed else "FAIL", "reason": "QUALIFICATION_THRESHOLDS_SATISFIED" if passed else "QUALIFICATION_THRESHOLDS_NOT_SATISFIED"}
        if record["qualification"] != expected:
            raise ValueError("qualification record decision mismatch")


class NLIQualificationRunner:
    def __init__(self, manifest, provider_factory):
        if not isinstance(manifest, VersionedNLIQualificationManifest):
            raise ValueError("manifest must be VersionedNLIQualificationManifest")
        if not callable(provider_factory):
            raise ValueError("provider_factory must be callable")
        self._payload_json = canonical_payload(manifest.payload())
        self.manifest_sha256 = manifest.payload_sha256
        self._provider_factory = provider_factory

    def run(self):
        payload = json.loads(self._payload_json)
        model = payload["model"]
        provider = self._provider_factory(model["model_id"], model["model_revision"])
        descriptor = getattr(provider, "loaded_model_descriptor", None)
        if not isinstance(descriptor, LoadedNLIModelDescriptor):
            raise ValueError("provider did not return a trusted loaded-model descriptor")
        observed_model = {"model_id": descriptor.model_id, "model_revision": descriptor.model_revision, "provider_type": descriptor.provider_type, "labels": list(descriptor.labels)}
        if observed_model != model:
            raise ValueError("loaded NLI model identity does not match manifest")
        policy = payload["policy"]
        verifier = RAGVerifier(chunks={}, retrieved_ids=set(), nli_provider=provider, pass_threshold=policy["pass_threshold"], fail_threshold=policy["fail_threshold"])
        label_matches = decision_matches = 0
        case_results = []
        for case in payload["cases"]:
            scores = provider.predict(case["premise"], case["hypothesis"])
            validation = verifier.validate_nli_scores(scores)
            if not validation["valid"]:
                raise ValueError("NLI provider output contract violation")
            normalized = validation["scores"]
            predicted_label = max(sorted(normalized), key=normalized.get)
            observed_status = verifier.classify_nli_scores(normalized)["status"]
            label_matches += predicted_label == case["expected_label"]
            decision_matches += observed_status == case["expected_status"]
            case_results.append({
                "case_id": case["case_id"],
                "expected_label": case["expected_label"],
                "predicted_label": predicted_label,
                "expected_status": case["expected_status"],
                "observed_status": observed_status,
                "scores": {label: normalized[label] for label in sorted(normalized)},
            })
        count = len(payload["cases"])
        metrics = {"label_accuracy": label_matches / count, "decision_accuracy": decision_matches / count}
        thresholds = payload["qualification_thresholds"]
        passed = metrics["label_accuracy"] >= thresholds["minimum_label_accuracy"] and metrics["decision_accuracy"] >= thresholds["minimum_decision_accuracy"]
        return {"model": dict(model), "benchmark": {"benchmark_id": payload["benchmark_id"], "benchmark_version": payload["benchmark_version"], "case_count": count}, "policy": dict(policy), "case_results": case_results, "metrics": metrics, "qualification": {"status": "PASS" if passed else "FAIL", "reason": "QUALIFICATION_THRESHOLDS_SATISFIED" if passed else "QUALIFICATION_THRESHOLDS_NOT_SATISFIED"}}


def load_nli_qualification_manifest(path):
    with open(path, encoding="utf-8") as manifest_file:
        return VersionedNLIQualificationManifest(json.load(manifest_file, object_pairs_hook=_reject_duplicate_keys))


def build_nli_qualification_runner(manifest):
    return NLIQualificationRunner(manifest, lambda model_id, revision: HuggingFaceNLIProvider(model_id, revision))


def qualify_nli_manifest(path):
    manifest = load_nli_qualification_manifest(path)
    return manifest.qualify(build_nli_qualification_runner(manifest))
