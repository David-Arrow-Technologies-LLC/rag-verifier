import hashlib
import json
import re
from pathlib import Path

from supply_chain_policy import file_sha256, reject_duplicate_json_keys


POLICY_VERSION = "rag-v30-adversarial-injection-v1"
DECISION_ALLOW = "ALLOW"
DECISION_BLOCK = "BLOCK"
MANIFEST_FIELDS = {"schema_version", "policy_version", "required_categories", "cases"}
CASE_FIELDS = {"id", "category", "document", "expected_decision", "expected_rule_ids"}
REQUIRED_CATEGORIES = {
    "benign-clinical-content",
    "instruction-override",
    "role-boundary-injection",
    "secret-exfiltration",
    "tool-command",
    "citation-bypass",
    "authority-impersonation",
}
CATEGORY_RULE_IDS = {
    category: category
    for category in REQUIRED_CATEGORIES - {"benign-clinical-content"}
}
RULES = (
    (
        "instruction-override",
        re.compile(
            r"(?:"
            r"\b(?:ignore|disregard|override|forget)\b[^.!?;]{0,80}"
            r"\b(?:previous|prior|system|developer|safety|instructions?|"
            r"everything\s+above|all\s+(?:text|content|instructions?|directions?)\s+above)\b"
            r"|\b(?:follow|obey)\s+(?:these|the\s+following)\s+"
            r"(?:directions|instructions)\s+instead\b"
            r")",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role-boundary-injection",
        re.compile(
            r"(?:<\\|(?:system|developer|assistant|user)\\|>|\\[/?INST\\]|"
            r"^\\s*#{1,6}[ \\t]*"
            r"(?:(?:\\*\\*|__|`)(?:system|developer|assistant|user)\\s*:?"
            r"(?:\\*\\*|__|`)|(?:system|developer|assistant|user))"
            r"[ \\t]*(?::|[-—])?[ \\t]*"
            r"(?=$|"
            r"(?:treat|ignore|disregard|override|forget|follow|obey|replace|"
            r"answer|state|report|reveal|show|print|execute|run|call|invoke|"
            r"respond|use|act\\s+as|you\\s+are|do\\s+not|don\'t|never|must|shall)\\b))",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "secret-exfiltration",
        re.compile(
            r"\b(?:reveal|print|show|expose|return)\b.{0,80}"
            r"\b(?:system prompt|developer message|secret|credential|api[ -]?key|access token)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "tool-command",
        re.compile(
            r"\b(?:call|invoke|execute|run)\b.{0,40}"
            r"\b(?:tool|function|shell|command)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "citation-bypass",
        re.compile(
            r"(?:"
            r"\b(?:do not|don't)\b[^.!?;]{0,40}"
            r"\b(?:include|provide|use|add)\b[^.!?;]{0,20}\bcitations?\b"
            r"|\b(?:omit|remove|fabricate|invent)\b[^.!?;]{0,40}\bcitations?\b"
            r")",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "authority-impersonation",
        re.compile(
            r"\b(?:system|developer|administrator|security policy)\s+"
            r"(?:message|instruction|notice|override)\b",
            re.IGNORECASE,
        ),
    ),
)


_PROTECTIVE_NEGATION = re.compile(
    r"\b(?:do\s+not|don't|never)"
    r"(?:"
    r"\s+(?:ever|again|intentionally|deliberately|knowingly|accidentally)"
    r"|\s+under\s+any\s+circumstances"
    r"|\s+at\s+any\s+time"
    r"){0,2}\s*$",
    re.IGNORECASE,
)
_PROTECTIVELY_NEGATED_RULES = {"instruction-override", "citation-bypass"}


def _is_protectively_negated(document, match_start):
    prefix = re.sub(r"\s+", " ", document[:match_start])
    return _PROTECTIVE_NEGATION.search(prefix) is not None


def _rule_matches(rule_id, pattern, document):
    for match in pattern.finditer(document):
        if (
            rule_id in _PROTECTIVELY_NEGATED_RULES
            and _is_protectively_negated(document, match.start())
        ):
            continue
        return True
    return False


class PromptInjectionDetectedError(ValueError):
    def __init__(self, chunk_id, rule_ids):
        self.chunk_id = chunk_id
        self.rule_ids = tuple(rule_ids)
        super().__init__(
            f"prompt injection detected in evidence chunk {chunk_id}: "
            + ", ".join(self.rule_ids)
        )


def canonical_payload(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def payload_sha256(value):
    return hashlib.sha256(canonical_payload(value).encode()).hexdigest()


def _reject_non_finite_json_constant(value):
    raise ValueError(f"non-finite JSON constant is invalid: {value}")


def inspect_document(document):
    if not isinstance(document, str) or not document.strip():
        raise ValueError("document must be a non-empty string")
    rule_ids = [
        rule_id
        for rule_id, pattern in RULES
        if _rule_matches(rule_id, pattern, document)
    ]
    return {
        "decision": DECISION_BLOCK if rule_ids else DECISION_ALLOW,
        "rule_ids": rule_ids,
    }


def enforce_document_security(chunk_id, document):
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("chunk_id must be a non-empty string")
    result = inspect_document(document)
    if result["decision"] == DECISION_BLOCK:
        raise PromptInjectionDetectedError(chunk_id, result["rule_ids"])
    return document


def load_adversarial_manifest(path):
    document = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json_keys,
        parse_constant=_reject_non_finite_json_constant,
    )
    if not isinstance(document, dict) or set(document) != {"payload", "payload_sha256"}:
        raise ValueError("adversarial manifest envelope is invalid")
    payload = document["payload"]
    digest = document["payload_sha256"]
    if not isinstance(payload, dict) or set(payload) != MANIFEST_FIELDS:
        raise ValueError("adversarial manifest payload is invalid")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("adversarial manifest digest is invalid")
    if payload_sha256(payload) != digest:
        raise ValueError("adversarial manifest payload digest mismatch")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("adversarial manifest schema is invalid")
    if payload["policy_version"] != POLICY_VERSION:
        raise ValueError("unsupported adversarial policy version")
    categories = payload["required_categories"]
    if (
        not isinstance(categories, list)
        or len(categories) != len(REQUIRED_CATEGORIES)
        or set(categories) != REQUIRED_CATEGORIES
    ):
        raise ValueError("adversarial manifest categories are invalid")
    cases = payload["cases"]
    if not isinstance(cases, list) or not cases or any(not isinstance(case, dict) for case in cases):
        raise ValueError("adversarial cases are invalid")
    ids = []
    observed_categories = set()
    known_rule_ids = {rule_id for rule_id, _ in RULES}
    for case in cases:
        if set(case) != CASE_FIELDS:
            raise ValueError("adversarial case schema is invalid")
        case_id = case["id"]
        category = case["category"]
        expected = case["expected_decision"]
        expected_rule_ids = case["expected_rule_ids"]
        if not isinstance(case_id, str) or re.fullmatch(r"ADV-[0-9]{3}", case_id) is None:
            raise ValueError("adversarial case ID is invalid")
        if category not in REQUIRED_CATEGORIES:
            raise ValueError("adversarial case category is invalid")
        inspect_document(case["document"])
        if expected not in {DECISION_ALLOW, DECISION_BLOCK}:
            raise ValueError("adversarial expected decision is invalid")
        required_rule_id = CATEGORY_RULE_IDS.get(category)
        if (
            not isinstance(expected_rule_ids, list)
            or len(expected_rule_ids) != len(set(expected_rule_ids))
            or any(rule_id not in known_rule_ids for rule_id in expected_rule_ids)
            or (
                category == "benign-clinical-content"
                and (expected != DECISION_ALLOW or expected_rule_ids)
            )
            or (
                required_rule_id is not None
                and (
                    expected != DECISION_BLOCK
                    or required_rule_id not in expected_rule_ids
                )
            )
        ):
            raise ValueError("adversarial expected rules are invalid")
        ids.append(case_id)
        observed_categories.add(category)
    if len(ids) != len(set(ids)):
        raise ValueError("adversarial case IDs must be unique")
    if observed_categories != REQUIRED_CATEGORIES:
        raise ValueError("adversarial manifest does not cover every required category")
    return document


def qualify_adversarial_manifest(path, source_revision, implementation_path=__file__):
    if (
        not isinstance(source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
    ):
        raise ValueError("source revision is invalid")
    manifest = load_adversarial_manifest(path)
    results = []
    for case in manifest["payload"]["cases"]:
        observed = inspect_document(case["document"])
        passed = (
            observed["decision"] == case["expected_decision"]
            and observed["rule_ids"] == case["expected_rule_ids"]
        )
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expected_decision": case["expected_decision"],
                "observed_decision": observed["decision"],
                "expected_rule_ids": case["expected_rule_ids"],
                "observed_rule_ids": observed["rule_ids"],
                "passed": passed,
            }
        )
    passed_cases = sum(1 for result in results if result["passed"])
    qualification_status = "PASS" if passed_cases == len(results) else "FAIL"
    payload = {
        "artifact_schema_version": 1,
        "policy_version": POLICY_VERSION,
        "source_revision": source_revision,
        "implementation_sha256": file_sha256(implementation_path),
        "manifest_sha256": manifest["payload_sha256"],
        "metrics": {
            "total_cases": len(results),
            "passed_cases": passed_cases,
            "blocked_attack_cases": sum(
                1
                for result in results
                if result["expected_decision"] == DECISION_BLOCK
                and result["observed_decision"] == DECISION_BLOCK
            ),
            "allowed_benign_cases": sum(
                1
                for result in results
                if result["expected_decision"] == DECISION_ALLOW
                and result["observed_decision"] == DECISION_ALLOW
            ),
        },
        "case_results": results,
        "qualification": {
            "status": qualification_status,
            "reason": (
                "ALL_ADVERSARIAL_CASES_SATISFIED"
                if qualification_status == "PASS"
                else "ONE_OR_MORE_ADVERSARIAL_CASES_FAILED"
            ),
        },
    }
    return {"payload": payload, "payload_sha256": payload_sha256(payload)}
