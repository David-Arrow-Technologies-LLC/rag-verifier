import argparse
import json
from pathlib import Path

from prompt_injection_qualification import canonical_payload, qualify_adversarial_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Run deterministic RAG-V30 adversarial document qualification"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    evidence = qualify_adversarial_manifest(
        args.manifest,
        args.source_revision,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_payload(evidence) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": args.output,
                "status": evidence["payload"]["qualification"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if evidence["payload"]["qualification"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
