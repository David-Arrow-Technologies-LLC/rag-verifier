import math

from prompt_injection_qualification import enforce_document_security


class ContextBuilder:
    def __init__(
        self,
        max_words=500
    ):
        if (
            isinstance(max_words, bool)
            or not isinstance(max_words, int)
            or max_words <= 0
        ):
            raise ValueError(
                "max_words must be a positive integer"
            )

        self.max_words = max_words

    def validate_evidence(
        self,
        evidence
    ):
        if not isinstance(evidence, list):
            raise ValueError(
                "evidence must be a list"
            )

        required_fields = {
            "chunk_id",
            "text",
            "retrieval_score",
            "rerank_score"
        }

        seen_chunk_ids = set()

        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError(
                    "Each evidence item must be a dictionary"
                )

            missing_fields = (
                required_fields - item.keys()
            )

            if missing_fields:
                raise ValueError(
                    "Evidence item is missing required fields"
                )

            chunk_id = item["chunk_id"]
            text = item["text"]

            if (
                not isinstance(chunk_id, str)
                or not chunk_id.strip()
            ):
                raise ValueError(
                    "chunk_id must be a non-empty string"
                )

            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    "Duplicate chunk_id is not allowed"
                )

            seen_chunk_ids.add(chunk_id)

            if (
                not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError(
                    "text must be a non-empty string"
                )

            enforce_document_security(
                chunk_id,
                text
            )

            for score_name in (
                "retrieval_score",
                "rerank_score"
            ):
                score = item[score_name]

                if (
                    isinstance(score, bool)
                    or not isinstance(
                        score,
                        (int, float)
                    )
                ):
                    raise ValueError(
                        f"{score_name} must be numeric"
                    )

                if not math.isfinite(score):
                    raise ValueError(
                        f"{score_name} must be finite"
                    )

        return evidence

    def build(
        self,
        evidence
    ):
        self.validate_evidence(
            evidence
        )

        selected = []
        words_used = 0

        for item in evidence:
            word_count = len(
                item["text"].split()
            )

            if (
                words_used + word_count
                > self.max_words
            ):
                continue

            citation_id = (
                f"C{len(selected) + 1}"
            )

            selected.append(
                {
                    "citation_id": citation_id,
                    "chunk_id": item["chunk_id"],
                    "text": item["text"],
                    "retrieval_score": float(
                        item["retrieval_score"]
                    ),
                    "rerank_score": float(
                        item["rerank_score"]
                    ),
                    "word_count": word_count
                }
            )

            words_used += word_count

        context_text = "\n\n".join(
            (
                f"[{item['citation_id']}] "
                f"{item['text']}"
            )
            for item in selected
        )

        return {
            "context": context_text,
            "evidence": selected,
            "citation_map": {
                item["citation_id"]: item["text"]
                for item in selected
            },
            "word_count": words_used,
            "max_words": self.max_words
        }
