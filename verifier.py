import math
import re
from nli_provider import (
    DEFAULT_NLI_MODEL_ID,
    HuggingFaceNLIProvider,
    resolve_nli_model_revision,
)


class RAGVerifier:

    def __init__(
        self,
        chunks,
        retrieved_ids,
        model_id=DEFAULT_NLI_MODEL_ID,
        pass_threshold=0.90,
        fail_threshold=0.90,
        nli_provider=None,
        model_revision=None,
    ):
        self.chunks = chunks
        self.retrieved_ids = set(retrieved_ids)

        self.pass_threshold = float(pass_threshold)
        self.fail_threshold = float(fail_threshold)

        if not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError(
                "pass_threshold must be between 0 and 1"
            )

        if not 0.0 <= self.fail_threshold <= 1.0:
            raise ValueError(
                "fail_threshold must be between 0 and 1"
            )

        self.nli_provider = (
            nli_provider
            if nli_provider is not None
            else HuggingFaceNLIProvider(
                model_id=model_id,
                revision=resolve_nli_model_revision(model_id, model_revision),
            )
        )


    # --------------------------------------------------------
    # CITATION EXTRACTION
    # --------------------------------------------------------

    def extract_citations(self, text):

        return re.findall(
            r"\[([A-Za-z0-9_-]+)\]",
            text
        )


    # --------------------------------------------------------
    # REMOVE CITATION MARKERS
    # --------------------------------------------------------

    def remove_citations(self, text):

        return re.sub(
            r"\s*\[[A-Za-z0-9_-]+\]",
            "",
            text
        ).strip()


    # --------------------------------------------------------
    # SENTENCE SPLITTING
    # --------------------------------------------------------

    def split_claims(self, answer):

        normalized = re.sub(
            r"\s+",
            " ",
            answer
        ).strip()

        if not normalized:
            return []

        pattern = re.compile(
            r".+?(?:[.!?](?:\s*\[[A-Za-z0-9_-]+\])*\s*|$)"
        )

        matches = [
            match.group(0).strip()
            for match in pattern.finditer(normalized)
            if match.group(0).strip()
        ]

        if not matches:
            return [normalized]

        return matches


    # --------------------------------------------------------
    # ATOMIC PROPOSITION DECOMPOSITION
    # --------------------------------------------------------

    def decompose_atomic_claims(self, claim_text):

        citation_ids = self.extract_citations(
            claim_text
        )

        claim = self.remove_citations(
            claim_text
        ).strip()

        if not claim:
            return []

        terminal = ""

        if claim[-1:] in ".!?":
            terminal = claim[-1]
            core = claim[:-1].strip()
        else:
            core = claim

        parts = re.split(
            r"\s+\band\b\s+",
            core,
            flags=re.IGNORECASE
        )

        if len(parts) == 1:

            return [
                {
                    "claim": claim,
                    "citations": citation_ids
                }
            ]

        verb_pattern = re.compile(
            r"\b("
            r"is|are|was|were|has|have|had|"
            r"occurs|occur|causes|cause|caused|"
            r"develops|develop|developed|"
            r"includes|include|included|"
            r"requires|require|required|"
            r"contains|contain|contained|"
            r"shows|show|showed|"
            r"indicates|indicate|indicated"
            r")\b",
            flags=re.IGNORECASE
        )

        clause_like = [
            bool(verb_pattern.search(part))
            for part in parts
        ]

        if not all(clause_like):

            return [
                {
                    "claim": claim,
                    "citations": citation_ids
                }
            ]

        atomic_claims = []

        for index, part in enumerate(parts):

            atomic_text = part.strip()

            if index == len(parts) - 1 and terminal:
                atomic_text += terminal
            else:
                atomic_text += "."

            atomic_claims.append(
                {
                    "claim": atomic_text,
                    "citations": list(citation_ids)
                }
            )

        return atomic_claims


    # --------------------------------------------------------
    # THRESHOLD-BASED NLI POLICY
    # --------------------------------------------------------

    def validate_nli_scores(
        self,
        scores,
        sum_tolerance=1e-3
    ):
        required_labels = {
            "entailment",
            "contradiction",
            "neutral"
        }

        if not isinstance(scores, dict):
            return {
                "valid": False,
                "reason": "NLI_PROVIDER_OUTPUT_NOT_DICT"
            }

        actual_labels = set(scores.keys())

        if actual_labels != required_labels:
            return {
                "valid": False,
                "reason": "NLI_PROVIDER_LABEL_CONTRACT_VIOLATION"
            }

        normalized = {}

        for label in required_labels:
            value = scores[label]

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float)
            ):
                return {
                    "valid": False,
                    "reason": "NLI_PROVIDER_SCORE_NOT_NUMERIC"
                }

            value = float(value)

            if not math.isfinite(value):
                return {
                    "valid": False,
                    "reason": "NLI_PROVIDER_SCORE_NOT_FINITE"
                }

            if not 0.0 <= value <= 1.0:
                return {
                    "valid": False,
                    "reason": "NLI_PROVIDER_SCORE_OUT_OF_RANGE"
                }

            normalized[label] = value

        total = sum(normalized.values())

        if abs(total - 1.0) > sum_tolerance:
            return {
                "valid": False,
                "reason": "NLI_PROVIDER_PROBABILITY_SUM_INVALID"
            }

        return {
            "valid": True,
            "scores": normalized
        }

    def classify_nli_scores(self, scores):

        entailment = scores.get(
            "entailment",
            0.0
        )

        contradiction = scores.get(
            "contradiction",
            0.0
        )

        neutral = scores.get(
            "neutral",
            0.0
        )


        # Fail precedence is deliberate.
        # If contradiction satisfies the hard failure threshold,
        # the verifier fails closed.
        if contradiction >= self.fail_threshold:

            return {
                "status": "FAIL",
                "reason": "CONTRADICTION_THRESHOLD_MET",
                "decision_label": "contradiction"
            }


        if entailment >= self.pass_threshold:

            return {
                "status": "PASS",
                "reason": "ENTAILMENT_THRESHOLD_MET",
                "decision_label": "entailment"
            }


        return {
            "status": "REVIEW",
            "reason": "NLI_CONFIDENCE_BELOW_DECISION_THRESHOLD",
            "decision_label": (
                max(
                    {
                        "entailment": entailment,
                        "contradiction": contradiction,
                        "neutral": neutral
                    },
                    key={
                        "entailment": entailment,
                        "contradiction": contradiction,
                        "neutral": neutral
                    }.get
                )
            )
        }


    # --------------------------------------------------------
    # VERIFY ONE CITATION
    # --------------------------------------------------------

    def verify_citation(
        self,
        claim,
        citation_id
    ):

        if citation_id not in self.chunks:

            return {
                "citation": citation_id,
                "status": "FAIL",
                "reason": "CITATION_NOT_FOUND"
            }


        if citation_id not in self.retrieved_ids:

            return {
                "citation": citation_id,
                "status": "FAIL",
                "reason": "CITATION_NOT_RETRIEVED"
            }


        premise = self.chunks[citation_id]

        scores = self.nli_provider.predict(
            premise,
            claim
        )

        validation = self.validate_nli_scores(
            scores
        )

        if not validation["valid"]:
            return {
                "citation": citation_id,
                "status": "FAIL",
                "reason": validation["reason"],
                "argmax_prediction": None,
                "decision_label": None,
                "probabilities": None
            }

        scores = validation["scores"]

        argmax_prediction = max(
            scores,
            key=scores.get
        )
        policy = self.classify_nli_scores(
            scores
        )


        return {
            "citation": citation_id,
            "status": policy["status"],
            "reason": policy["reason"],
            "argmax_prediction": argmax_prediction,
            "decision_label": policy["decision_label"],
            "thresholds": {
                "pass_threshold": self.pass_threshold,
                "fail_threshold": self.fail_threshold
            },
            "probabilities": {
                key: round(value, 4)
                for key, value in scores.items()
            }
        }


    # --------------------------------------------------------
    # CITATION AGGREGATION
    # --------------------------------------------------------

    def aggregate_citation_status(
        self,
        citation_verifications
    ):

        statuses = [
            item["status"]
            for item in citation_verifications
        ]

        if "FAIL" in statuses:

            return (
                "FAIL",
                "ONE_OR_MORE_CITATIONS_FAILED"
            )

        if "PASS" in statuses:

            return (
                "PASS",
                "SUPPORTED_BY_VALID_CITATION"
            )

        return (
            "REVIEW",
            "NO_CITATION_ESTABLISHED_SUPPORT"
        )


    # --------------------------------------------------------
    # VERIFY ONE ATOMIC CLAIM
    # --------------------------------------------------------

    def verify_atomic_claim(
        self,
        claim,
        citation_ids
    ):

        if not citation_ids:

            return {
                "claim": claim,
                "citations": [],
                "status": "FAIL",
                "reason": "UNCITED_FACTUAL_CLAIM",
                "verification": []
            }

        citation_verifications = [
            self.verify_citation(
                claim,
                citation_id
            )
            for citation_id in citation_ids
        ]

        status, reason = self.aggregate_citation_status(
            citation_verifications
        )

        return {
            "claim": claim,
            "citations": citation_ids,
            "status": status,
            "reason": reason,
            "verification": citation_verifications
        }


    # --------------------------------------------------------
    # ANSWER AGGREGATION
    # --------------------------------------------------------

    def aggregate_answer_status(
        self,
        claim_results
    ):

        if not claim_results:

            return (
                "FAIL",
                "NO_VERIFIABLE_CLAIMS"
            )

        statuses = [
            result["status"]
            for result in claim_results
        ]

        if "FAIL" in statuses:

            return (
                "FAIL",
                "ONE_OR_MORE_CLAIMS_FAILED"
            )

        if "REVIEW" in statuses:

            return (
                "REVIEW",
                "ONE_OR_MORE_CLAIMS_REQUIRE_REVIEW"
            )

        return (
            "PASS",
            "ALL_ATOMIC_CLAIMS_VERIFIED"
        )


    # --------------------------------------------------------
    # VERIFY COMPLETE ANSWER
    # --------------------------------------------------------

    def verify_answer(
        self,
        answer
    ):

        sentence_units = self.split_claims(
            answer
        )

        atomic_units = []

        for sentence in sentence_units:

            atomic_units.extend(
                self.decompose_atomic_claims(
                    sentence
                )
            )

        claim_results = [
            self.verify_atomic_claim(
                unit["claim"],
                unit["citations"]
            )
            for unit in atomic_units
        ]

        total_claims = len(
            claim_results
        )

        cited_claims = sum(
            1
            for result in claim_results
            if result["citations"]
        )

        coverage_ratio = (
            cited_claims / total_claims
            if total_claims
            else 0.0
        )

        status, reason = self.aggregate_answer_status(
            claim_results
        )

        return {
            "answer": answer,
            "status": status,
            "reason": reason,
            "trusted_release": status == "PASS",
            "policy": {
                "pass_threshold": self.pass_threshold,
                "fail_threshold": self.fail_threshold
            },
            "citation_coverage": {
                "cited_claims": cited_claims,
                "total_claims": total_claims,
                "coverage_ratio": round(
                    coverage_ratio,
                    4
                ),
                "coverage_percent": round(
                    coverage_ratio * 100,
                    1
                )
            },
            "claims": claim_results
        }



