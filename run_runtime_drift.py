import argparse
import json
from pathlib import Path

from runtime_drift import DECISION_CURRENT, canonical_payload, evaluate_runtime_drift


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG runtime drift and requalification triggers")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--qualified-evidence", required=True)
    parser.add_argument("--observed-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_runtime_drift(args.policy, args.qualified_evidence, args.observed_evidence)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_payload(result) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": args.output, "decision": result["payload"]["decision"]}, sort_keys=True))
    return 0 if result["payload"]["decision"] == DECISION_CURRENT else 1


if __name__ == "__main__":
    raise SystemExit(main())
