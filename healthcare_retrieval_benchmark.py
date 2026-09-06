class HealthcareRetrievalBenchmark:
    def __init__(
        self,
        retriever,
        top_k=5
    ):
        retrieve = getattr(
            retriever,
            "retrieve",
            None
        )

        if not callable(retrieve):
            raise ValueError(
                "retriever must provide callable retrieve()"
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer"
            )

        self.retriever = retriever
        self.top_k = top_k

    @staticmethod
    def _validate_query(query):
        if not isinstance(query, dict):
            raise ValueError(
                "each query must be a dict"
            )

        required = {
            "query_id",
            "text",
            "relevant_ids"
        }

        if not required.issubset(query):
            raise ValueError(
                "query must contain query_id, "
                "text, and relevant_ids"
            )

        query_id = query["query_id"]
        text = query["text"]
        relevant_ids = query["relevant_ids"]

        if (
            not isinstance(query_id, str)
            or not query_id.strip()
        ):
            raise ValueError(
                "query_id must be a non-empty string"
            )

        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError(
                "text must be a non-empty string"
            )

        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
        ):
            raise ValueError(
                "relevant_ids must be a non-empty list"
            )

        for relevant_id in relevant_ids:
            if (
                not isinstance(relevant_id, str)
                or not relevant_id.strip()
            ):
                raise ValueError(
                    "relevant_ids must contain "
                    "non-empty strings"
                )

        if len(set(relevant_ids)) != len(relevant_ids):
            raise ValueError(
                "relevant_ids must be unique"
            )

    @classmethod
    def validate_queries(cls, queries, corpus_ids=None):
        if (
            not isinstance(queries, list)
            or not queries
        ):
            raise ValueError(
                "queries must be a non-empty list"
            )

        query_ids = []
        corpus_id_set = (
            None
            if corpus_ids is None
            else set(corpus_ids)
        )

        for query in queries:
            cls._validate_query(query)
            query_ids.append(query["query_id"])

            if (
                corpus_id_set is not None
                and not set(query["relevant_ids"]).issubset(
                    corpus_id_set
                )
            ):
                raise ValueError(
                    "relevant_ids must exist in the chunk corpus"
                )

        if len(set(query_ids)) != len(query_ids):
            raise ValueError(
                "query_ids must be unique"
            )

    @staticmethod
    def _validate_retrieved(retrieved):
        if not isinstance(retrieved, list):
            raise ValueError(
                "retriever output must be a list"
            )

        retrieved_ids = []

        for item in retrieved:
            if (
                not isinstance(item, dict)
                or "chunk_id" not in item
            ):
                raise ValueError(
                    "retrieved item must contain chunk_id"
                )

            chunk_id = item["chunk_id"]

            if (
                not isinstance(chunk_id, str)
                or not chunk_id.strip()
            ):
                raise ValueError(
                    "chunk_id must be a non-empty string"
                )

            retrieved_ids.append(chunk_id)

        if (
            len(set(retrieved_ids))
            != len(retrieved_ids)
        ):
            raise ValueError(
                "retrieved chunk_ids must be unique"
            )

        return retrieved_ids

    def evaluate(self, queries):
        self.validate_queries(queries)

        per_query = []

        for query in queries:
            text = query["text"]
            query_id = query["query_id"]
            relevant_ids = query["relevant_ids"]

            retrieved = self.retriever.retrieve(
                text,
                top_k=self.top_k
            )

            retrieved_ids = (
                self._validate_retrieved(retrieved)
            )

            if len(retrieved_ids) > self.top_k:
                raise ValueError(
                    "retriever returned more than top_k results"
                )

            relevant_set = set(relevant_ids)

            retrieved_relevant = [
                chunk_id
                for chunk_id in retrieved_ids
                if chunk_id in relevant_set
            ]

            recall_at_k = (
                len(retrieved_relevant)
                / len(relevant_set)
            )

            # Precision@K deliberately uses configured K,
            # not the number of results actually returned.
            precision_at_k = (
                len(retrieved_relevant)
                / self.top_k
            )

            reciprocal_rank = 0.0

            for rank, chunk_id in enumerate(
                retrieved_ids,
                start=1
            ):
                if chunk_id in relevant_set:
                    reciprocal_rank = 1.0 / rank
                    break

            per_query.append({
                "query_id": query_id,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": list(relevant_ids),
                "recall_at_k": recall_at_k,
                "reciprocal_rank": reciprocal_rank,
                "precision_at_k": precision_at_k
            })

        query_count = len(per_query)

        return {
            "query_count": query_count,
            "recall_at_k": (
                sum(
                    item["recall_at_k"]
                    for item in per_query
                )
                / query_count
            ),
            "mrr": (
                sum(
                    item["reciprocal_rank"]
                    for item in per_query
                )
                / query_count
            ),
            "precision_at_k": (
                sum(
                    item["precision_at_k"]
                    for item in per_query
                )
                / query_count
            ),
            "queries": per_query
        }
