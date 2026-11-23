"""Static pre-inference checks for a *candidate*, not a frozen qualification manifest."""
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[2] / "nli_deberta_qualification_v2.json"
CANDIDATE = Path(__file__).resolve().parent / "holdout_cases_candidate.json"
CATEGORIES = {
    "comorbidity", "medication-dose", "laboratory-numeric", "negation",
    "temporal", "laterality", "allergy", "demographic", "missing-evidence", "paraphrase",
}
LABEL_TO_STATUS = {"entailment": "PASS", "contradiction": "FAIL", "neutral": "REVIEW"}


def normalize(value):
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def audit(baseline_path=BASELINE, candidate_path=CANDIDATE):
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))["payload"]["cases"]
    candidate_document = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    if candidate_document["payload_sha256"] != "PENDING_FREEZE_AFTER_REVIEW":
        raise ValueError("candidate freeze sentinel missing; frozen manifests need separate approval")
    payload = candidate_document["payload"]
    baseline_payload = json.loads(Path(baseline_path).read_text(encoding="utf-8"))["payload"]
    if payload["schema_version"] != 2 or payload["model"] != baseline_payload["model"]:
        raise ValueError("holdout schema or pinned model diverges from baseline")
    if payload["policy"] != baseline_payload["policy"]:
        raise ValueError("holdout release thresholds diverge from baseline")
    if payload["qualification_thresholds"]["minimum_label_accuracy"] != 0.9 or (
        payload["qualification_thresholds"]["minimum_decision_accuracy"] != 0.9
    ):
        raise ValueError("holdout accuracy targets diverge from protocol")
    cases = payload["cases"]
    if len(cases) != 90:
        raise ValueError("holdout must contain exactly 90 cases")
    if len({case["case_id"] for case in cases}) != 90:
        raise ValueError("duplicate case IDs")
    if Counter(case["expected_label"] for case in cases) != {key: 30 for key in LABEL_TO_STATUS}:
        raise ValueError("unbalanced labels")
    if any(case["expected_status"] != LABEL_TO_STATUS[case["expected_label"]] for case in cases):
        raise ValueError("label/status mismatch")
    if any(sum(case["category"] == cat and case["expected_label"] == label for case in cases) != 3
           for cat in CATEGORIES for label in LABEL_TO_STATUS):
        raise ValueError("category/label coverage mismatch")
    if any(set(case) != {"case_id", "category", "critical", "premise", "hypothesis",
                             "expected_label", "expected_status"} for case in cases):
        raise ValueError("unexpected candidate case schema")
    if not any(case["critical"] for case in cases):
        raise ValueError("no critical cases")
    if any(not isinstance(case["critical"], bool) for case in cases):
        raise ValueError("invalid critical flag")
    baseline_pairs = {(normalize(c["premise"]), normalize(c["hypothesis"])) for c in baseline}
    candidate_pairs = [(normalize(c["premise"]), normalize(c["hypothesis"])) for c in cases]
    if len(set(candidate_pairs)) != 90 or set(candidate_pairs) & baseline_pairs:
        raise ValueError("exact or normalized duplicate pair")
    # Flag unusually similar complete pairs for human adjudication; do not auto-certify independence.
    comparisons = []
    baseline_strings = [" || ".join(pair) for pair in baseline_pairs]
    for case, pair in zip(cases, candidate_pairs):
        candidate_string = " || ".join(pair)
        for baseline_string in baseline_strings:
            similarity = SequenceMatcher(None, candidate_string, baseline_string).ratio()
            if similarity >= 0.92:
                comparisons.append((case["case_id"], round(similarity, 3)))
    if comparisons:
        raise ValueError(f"near-duplicate baseline pairs require adjudication: {comparisons}")
    # Three hypotheses intentionally share each premise: 90 pairs, not 90 independent scenarios.
    unique_premises = len({normalize(c["premise"]) for c in cases})
    return {"unique_premises": unique_premises, "candidate_cases": 90, "per_label": dict(Counter(c["expected_label"] for c in cases)),
            "per_category": dict(Counter(c["category"] for c in cases)),
            "baseline_pairs": len(baseline), "near_duplicate_threshold": 0.92,
            "annotation_status": "single-author candidate; independent review pending",
            "manifest_status": "unfrozen; no model inference authorized"}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
