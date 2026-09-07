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
    CASE_FIELDS_V1 = {"case_id", "premise", "hypothesis", "expected_label", "expected_status"}
    CASE_FIELDS_V2 = CASE_FIELDS_V1 | {"category", "critical"}
    QUALIFICATION_FIELDS_V1 = {"minimum_label_accuracy", "minimum_decision_accuracy"}
    QUALIFICATION_FIELDS_V2 = QUALIFICATION_FIELDS_V1 | {"minimum_per_label_accuracy", "require_all_critical_cases"}
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
        if type(payload["schema_version"]) is not int or payload["schema_version"] not in {1, 2}:
            raise ValueError("unsupported manifest schema_version")
        for field in ("benchmark_id", "benchmark_version"):
            if not isinstance(payload[field], str) or not payload[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
        self._validate_model(payload["model"])
        self._validate_policy(payload["policy"])
        self._validate_qualification_thresholds(payload["qualification_thresholds"], payload["schema_version"])
        self._validate_cases(payload["cases"], payload["schema_version"])
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
    def _validate_qualification_thresholds(cls, thresholds, schema_version):
        expected_fields = cls.QUALIFICATION_FIELDS_V1 if schema_version == 1 else cls.QUALIFICATION_FIELDS_V2
        if not isinstance(thresholds, dict) or set(thresholds) != expected_fields:
            raise ValueError("qualification threshold fields are invalid")
        for field in cls.QUALIFICATION_FIELDS_V1:
            value = thresholds[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError("qualification thresholds are invalid")
        if schema_version == 2:
            per_label = thresholds["minimum_per_label_accuracy"]
            if not isinstance(per_label, dict) or set(per_label) != cls.LABELS:
                raise ValueError("per-label qualification thresholds are invalid")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0 for value in per_label.values()):
                raise ValueError("per-label qualification thresholds are invalid")
            if type(thresholds["require_all_critical_cases"]) is not bool:
                raise ValueError("require_all_critical_cases must be boolean")

    @classmethod
    def _validate_cases(cls, cases, schema_version):
        if not isinstance(cases, list) or not cases:
            raise ValueError("cases must be a non-empty list")
        case_ids = set()
        for case in cases:
            expected_fields = cls.CASE_FIELDS_V1 if schema_version == 1 else cls.CASE_FIELDS_V2
            if not isinstance(case, dict) or set(case) != expected_fields:
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
            if schema_version == 2:
                if not isinstance(case["category"], str) or not case["category"].strip():
                    raise ValueError("category must be a non-empty string")
                if type(case["critical"]) is not bool:
                    raise ValueError("critical must be boolean")
        if schema_version == 2:
            if {case["expected_label"] for case in cases} != cls.LABELS:
                raise ValueError("schema v2 cases must cover every NLI label")
            if not any(case["critical"] for case in cases):
                raise ValueError("schema v2 requires at least one critical case")

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
        schema_version = payload["schema_version"]
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
            if schema_version == 2:
                fields |= {"category", "critical"}
            if not isinstance(result, dict) or set(result) != fields:
                raise ValueError("qualification record case result is invalid")
            case_id = result["case_id"]
            if case_id in seen_case_ids or case_id not in expected_cases:
                raise ValueError("qualification record case identity is invalid")
            seen_case_ids.add(case_id)
            expected_case = expected_cases[case_id]
            if result["expected_label"] != expected_case["expected_label"] or result["expected_status"] != expected_case["expected_status"]:
                raise ValueError("qualification record expected outcome mismatch")
            if schema_version == 2 and (result["category"] != expected_case["category"] or result["critical"] != expected_case["critical"]):
                raise ValueError("qualification record case governance mismatch")
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
        metric_fields = {"label_accuracy", "decision_accuracy"}
        if schema_version == 2:
            metric_fields |= {"per_label_accuracy", "critical_case_accuracy"}
        if not isinstance(metrics, dict) or set(metrics) != metric_fields:
            raise ValueError("qualification record metrics are invalid")
        for field in ("label_accuracy", "decision_accuracy"):
            value = metrics[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError("qualification record metrics are invalid")
        count = len(case_results)
        expected_metrics = {
            "label_accuracy": label_matches / count,
            "decision_accuracy": decision_matches / count,
        }
        if schema_version == 2:
            per_label = {}
            for label in self.LABELS:
                label_cases = [case for case in case_results if case["expected_label"] == label]
                per_label[label] = sum(case["predicted_label"] == label for case in label_cases) / len(label_cases)
            critical_cases = [case for case in case_results if case["critical"]]
            if not critical_cases:
                raise ValueError("schema v2 requires critical cases")
            critical_accuracy = sum(
                case["predicted_label"] == case["expected_label"] and case["observed_status"] == case["expected_status"]
                for case in critical_cases
            ) / len(critical_cases)
            expected_metrics.update({"per_label_accuracy": per_label, "critical_case_accuracy": critical_accuracy})
        if metrics != expected_metrics:
            raise ValueError("qualification record metrics do not match case results")
        thresholds = payload["qualification_thresholds"]
        passed = metrics["label_accuracy"] >= thresholds["minimum_label_accuracy"] and metrics["decision_accuracy"] >= thresholds["minimum_decision_accuracy"]
        if schema_version == 2:
            passed = passed and all(metrics["per_label_accuracy"][label] >= minimum for label, minimum in thresholds["minimum_per_label_accuracy"].items())
            if thresholds["require_all_critical_cases"]:
                passed = passed and metrics["critical_case_accuracy"] == 1.0
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
                **({"category": case["category"], "critical": case["critical"]} if payload["schema_version"] == 2 else {}),
            })
        count = len(payload["cases"])
        metrics = {"label_accuracy": label_matches / count, "decision_accuracy": decision_matches / count}
        if payload["schema_version"] == 2:
            metrics["per_label_accuracy"] = {
                label: sum(case["predicted_label"] == label for case in case_results if case["expected_label"] == label)
                / sum(case["expected_label"] == label for case in case_results)
                for label in VersionedNLIQualificationManifest.LABELS
            }
            critical_cases = [case for case in case_results if case["critical"]]
            metrics["critical_case_accuracy"] = sum(
                case["predicted_label"] == case["expected_label"] and case["observed_status"] == case["expected_status"]
                for case in critical_cases
            ) / len(critical_cases)
        thresholds = payload["qualification_thresholds"]
        passed = metrics["label_accuracy"] >= thresholds["minimum_label_accuracy"] and metrics["decision_accuracy"] >= thresholds["minimum_decision_accuracy"]
        if payload["schema_version"] == 2:
            passed = passed and all(metrics["per_label_accuracy"][label] >= minimum for label, minimum in thresholds["minimum_per_label_accuracy"].items())
            if thresholds["require_all_critical_cases"]:
                passed = passed and metrics["critical_case_accuracy"] == 1.0
        return {"model": dict(model), "benchmark": {"benchmark_id": payload["benchmark_id"], "benchmark_version": payload["benchmark_version"], "case_count": count}, "policy": dict(policy), "case_results": case_results, "metrics": metrics, "qualification": {"status": "PASS" if passed else "FAIL", "reason": "QUALIFICATION_THRESHOLDS_SATISFIED" if passed else "QUALIFICATION_THRESHOLDS_NOT_SATISFIED"}}


def load_nli_qualification_manifest(path):
    with open(path, encoding="utf-8") as manifest_file:
        return VersionedNLIQualificationManifest(json.load(manifest_file, object_pairs_hook=_reject_duplicate_keys))


def build_nli_qualification_runner(manifest):
    return NLIQualificationRunner(manifest, lambda model_id, revision: HuggingFaceNLIProvider(model_id, revision))


def qualify_nli_manifest(path):
    manifest = load_nli_qualification_manifest(path)
    return manifest.qualify(build_nli_qualification_runner(manifest))
