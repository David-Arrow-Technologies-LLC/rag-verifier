import argparse
import json

from release_qualification import write_release_qualification


def main():
    parser = argparse.ArgumentParser(description="Run unified governed RAG release qualification")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", default=".")
    args = parser.parse_args()
    evidence = write_release_qualification(args.manifest, args.output, args.repository)
    print(json.dumps({"artifact": args.output, "payload_sha256": evidence["payload_sha256"]}, sort_keys=True))
    return 0 if evidence["payload"]["qualification"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
