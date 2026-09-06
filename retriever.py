import math


class Retriever:
    def __init__(
        self,
        chunks,
        embedding_provider,
        expected_dimension=None
    ):
        if (
            expected_dimension is not None
            and (
                isinstance(expected_dimension, bool)
                or not isinstance(expected_dimension, int)
                or expected_dimension <= 0
            )
        ):
            raise ValueError(
                "expected_dimension must be a positive integer"
            )

        self.chunks = chunks
        self.embedding_provider = embedding_provider
        self.expected_dimension = expected_dimension

    def validate_embedding_batch(
        self,
        embeddings,
        expected_count
    ):
        if not isinstance(
            embeddings,
            list
        ):
            raise ValueError(
                "Embedding provider output must be a list"
            )

        if len(embeddings) != expected_count:
            raise ValueError(
                "Embedding provider returned an unexpected number of vectors"
            )

        expected_dimension = None

        for vector in embeddings:
            if not isinstance(
                vector,
                list
            ):
                raise ValueError(
                    "Each embedding vector must be a list"
                )

            if len(vector) == 0:
                raise ValueError(
                    "Embedding vectors must not be empty"
                )

            if expected_dimension is None:
                expected_dimension = len(vector)
            elif len(vector) != expected_dimension:
                raise ValueError(
                    "Embedding vectors must have consistent dimensions"
                )

            if (
                self.expected_dimension is not None
                and len(vector) != self.expected_dimension
            ):
                raise ValueError(
                    "Embedding vector dimension does not match "
                    "expected_dimension"
                )

            for value in vector:
                if (
                    isinstance(value, bool)
                    or not isinstance(
                        value,
                        (int, float)
                    )
                ):
                    raise ValueError(
                        "Embedding values must be numeric"
                    )

                if not math.isfinite(value):
                    raise ValueError(
                        "Embedding values must be finite"
                    )

        return embeddings

    def cosine_similarity(
        self,
        vector_a,
        vector_b
    ):
        if len(vector_a) != len(vector_b):
            raise ValueError(
                "Embedding vectors must have the same dimension"
            )

        dot_product = sum(
            a * b
            for a, b in zip(
                vector_a,
                vector_b
            )
        )

        norm_a = math.sqrt(
            sum(
                value * value
                for value in vector_a
            )
        )

        norm_b = math.sqrt(
            sum(
                value * value
                for value in vector_b
            )
        )

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (
            norm_a * norm_b
        )

    def _encode_queries(self, texts):
        method = getattr(
            self.embedding_provider,
            "encode_queries",
            None
        )

        if callable(method):
            return method(texts)

        return self.embedding_provider.encode(
            texts
        )

    def _encode_documents(self, texts):
        method = getattr(
            self.embedding_provider,
            "encode_documents",
            None
        )

        if callable(method):
            return method(texts)

        return self.embedding_provider.encode(
            texts
        )

    def validate_provider_embeddings(self, query_texts):
        query_vectors = self._encode_queries(query_texts)
        self.validate_embedding_batch(
            query_vectors,
            expected_count=len(query_texts)
        )

        chunk_texts = list(self.chunks.values())
        chunk_vectors = self._encode_documents(chunk_texts)
        self.validate_embedding_batch(
            chunk_vectors,
            expected_count=len(chunk_texts)
        )

        return query_vectors, chunk_vectors

    def retrieve(
        self,
        query,
        top_k=3,
        min_score=None
    ):
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        chunk_ids = list(
            self.chunks.keys()
        )

        chunk_texts = [
            self.chunks[chunk_id]
            for chunk_id in chunk_ids
        ]

        query_vectors = (
            self._encode_queries(
                [query]
            )
        )

        self.validate_embedding_batch(
            query_vectors,
            expected_count=1
        )

        query_vector = query_vectors[0]

        chunk_vectors = (
            self._encode_documents(
                chunk_texts
            )
        )

        self.validate_embedding_batch(
            chunk_vectors,
            expected_count=len(
                chunk_texts
            )
        )

        for chunk_vector in chunk_vectors:
            if (
                len(chunk_vector)
                != len(query_vector)
            ):
                raise ValueError(
                    "Query and chunk embeddings must have the same dimension"
                )

        results = []

        for (
            chunk_id,
            chunk_text,
            chunk_vector
        ) in zip(
            chunk_ids,
            chunk_texts,
            chunk_vectors
        ):
            score = self.cosine_similarity(
                query_vector,
                chunk_vector
            )

            if (
                min_score is not None
                and score < min_score
            ):
                continue

            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "score": score
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]
