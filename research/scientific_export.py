"""Offline export of the documented historical synthetic NLI run only."""
import hashlib
import json
from nli_qualification import (VersionedNLIQualificationManifest, canonical_payload,
                               payload_sha256, _reject_duplicate_keys)

SOURCE = "1211dbd6588fb0ae189c98dc6135b3e8c5b14a6b"
EVIDENCE_SHA = "61046c9d607364bde334a33f1bd4de101a66478ecd5a9ef98036cb5f4ab52f9c"
DATASET_PAYLOAD_SHA = "5f2f90d865c067f32ce902fe0fa01cc7508b81a6f3aca35b800619f55c5b3b13"
CONSUMER_REVISION = "877be61693858513eaf38020ca4146b40e184192"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return canonical_payload(value).encode("utf-8")


def parse(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 10_000_000:
        raise ValueError("Invalid artifact bytes")
    def invalid(_):
        raise ValueError("Nonfinite JSON")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
                      parse_constant=invalid)


def export_historical(dataset_bytes, evidence_bytes):
    """Return manifest bytes and artifact bytes; never invoke a model."""
    if type(evidence_bytes) is not bytes or sha(evidence_bytes) != EVIDENCE_SHA:
        raise ValueError("Historical evidence digest mismatch")
    dataset = VersionedNLIQualificationManifest(parse(dataset_bytes))
    if dataset.payload_sha256 != DATASET_PAYLOAD_SHA:
        raise ValueError("Historical dataset digest mismatch")
    evidence = parse(evidence_bytes)
    if set(evidence) != {"payload", "payload_sha256"}:
        raise ValueError("Evidence envelope mismatch")
    p = evidence["payload"]
    if (payload_sha256(p) != evidence["payload_sha256"]
            or type(p["artifact_schema_version"]) is not int
            or p["artifact_schema_version"] != 2
            or p["source_revision"] != SOURCE
            or p["manifest_sha256"] != dataset.payload_sha256):
        raise ValueError("Evidence binding mismatch")
    record = p["qualification_record"]
    dataset._validate_record(record)
    rows = record["case_results"]
    critical = [row for row in rows if row["critical"]]
    nonpass = [row for row in rows if row["expected_status"] != "PASS"]
    metrics = dict(total=len(rows), expected_non_pass=len(nonpass),
                   false_pass=sum(row["observed_status"] == "PASS" for row in nonpass),
                   label_correct=sum(row["predicted_label"] == row["expected_label"] for row in rows),
                   decision_correct=sum(row["observed_status"] == row["expected_status"] for row in rows),
                   critical_total=len(critical),
                   critical_correct=sum(row["predicted_label"] == row["expected_label"]
                                        and row["observed_status"] == row["expected_status"] for row in critical))
    if not nonpass or not critical:
        raise ValueError("Missing required evaluation denominator")
    artifacts = {
        "protocol": encode({"kind": "retrospective_protocol_reference",
                            "study": "DA-RAG-01A", "consumer_revision": CONSUMER_REVISION,
                            "path": "protocols/DA-RAG-01/PROTOCOL.md",
                            "historical_run_precedes_protocol": True}),
        "dataset": dataset_bytes,
        "policy": encode({"policy": dataset.payload()["policy"],
                          "qualification_thresholds": dataset.payload()["qualification_thresholds"]}),
        "case_results": evidence_bytes,
        "metrics": encode(metrics),
        "environment": encode({k: p[k] for k in ("runtime", "dependency_lock", "ci_supply_chain")}),
    }
    refs = [dict(kind=k, version="historical-nli-export.v1", media_type="application/json",
                 serialization="raw-json-utf8", sha256=sha(v), byte_length=len(v))
            for k,v in sorted(artifacts.items())]
    model = dataset.payload()["model"]
    manifest = dict(schema_version="da-cairl.scientific-evidence.v1",
                    classification="synthetic_only", study="DA-RAG-01A",
                    run_id="github-actions-45019205079", producer_repository="David-Arrow-Technologies-LLC/rag-verifier",
                    source_revision=SOURCE, model_id=model["model_id"], model_revision=model["model_revision"],
                    artifacts=refs, metrics=metrics,
                    limitations=["Historical repository-authored benchmark; not independent or confirmatory.",
                                 "Protocol artifact is a retrospective reference, not a frozen protocol.",
                                 "Exact recorded evidence bytes retained; no inference or source authentication performed."],
                    scientific_approval=False, freeze_authorized=False, inference_authorized=False,
                    clinical_use_authorized=False, translation_authorized=False)
    for k in ("protocol", "dataset", "policy"):
        manifest[k+"_sha256"] = sha(artifacts[k])
    return encode(manifest), artifacts
