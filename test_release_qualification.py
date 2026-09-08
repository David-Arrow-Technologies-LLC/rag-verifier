import json

import pytest

from release_qualification import build_release_qualification, load_release_manifest


MANIFEST = "release_qualification_manifest.json"


def _nli_evidence(status="PASS"):
    return {
        "payload": {
            "source_revision": "a" * 40,
            "ci_supply_chain": {"manifest_sha256": "b" * 64},
            "dependency_lock": {"sha256": "c" * 64},
            "runtime": {"python": "3.12.13"},
            "manifest_sha256": "5f2f90d865c067f32ce902fe0fa01cc7508b81a6f3aca35b800619f55c5b3b13",
            "qualification_record": {"qualification": {"status": status}},
        }
    }


def _embedding(status="PASS"):
    return {
        "manifest_sha256": "ead2869961738f0188a076915317b2e53e562fc84ce3dc9860b9122f30d5bd41",
        "qualification_record": {"qualification": {"status": status}},
    }


def test_repository_release_manifest_is_valid_and_digest_bound():
    document = load_release_manifest(MANIFEST)
    assert document["payload"]["release_policy_version"] == "rag-v28-unified-release-v1"


def test_release_bundle_passes_only_when_both_components_pass(monkeypatch):
    monkeypatch.setattr("release_qualification.qualify_sentence_transformer_manifest", lambda _path: _embedding())
    monkeypatch.setattr("release_qualification.build_qualification_evidence", lambda *_args: _nli_evidence())
    evidence = build_release_qualification(MANIFEST)
    assert evidence["payload"]["qualification"]["status"] == "PASS"
    assert set(evidence["payload"]["components"]) == {"embedding-minilm", "nli-deberta-v2"}


def test_release_bundle_fails_closed_on_component_failure(monkeypatch):
    monkeypatch.setattr("release_qualification.qualify_sentence_transformer_manifest", lambda _path: _embedding("FAIL"))
    monkeypatch.setattr("release_qualification.build_qualification_evidence", lambda *_args: _nli_evidence())
    evidence = build_release_qualification(MANIFEST)
    assert evidence["payload"]["qualification"]["status"] == "FAIL"


def test_release_manifest_rejects_payload_tampering(tmp_path):
    document = json.loads(open(MANIFEST, encoding="utf-8").read())
    document["payload"]["release_policy_version"] = "forged"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="payload digest mismatch"):
        load_release_manifest(path)


def test_release_manifest_rejects_component_file_substitution(tmp_path):
    document = json.loads(open(MANIFEST, encoding="utf-8").read())
    document["payload"]["components"][0]["file_sha256"] = "0" * 64
    from nli_qualification import payload_sha256
    document["payload_sha256"] = payload_sha256(document["payload"])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="file digest mismatch"):
        load_release_manifest(path)
