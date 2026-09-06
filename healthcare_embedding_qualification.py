from dataclasses import dataclass

from healthcare_retrieval_benchmark import HealthcareRetrievalBenchmark
from model_qualification import ModelQualificationRecordBuilder
from retriever import Retriever


@dataclass(frozen=True)
class LoadedEmbeddingProvider:
    provider: object
    model_id: str
    model_revision: str
    provider_type: str
    embedding_dimension: int


class HealthcareEmbeddingQualificationRunner:
    """Run one governed healthcare retrieval qualification at a time."""

    def __init__(
        self,
        provider_factory,
        policy,
        benchmark_id,
        benchmark_version,
        top_k=5,
    ):
        if not callable(provider_factory):
            raise ValueError("provider_factory must be callable")

        if not isinstance(benchmark_id, str) or not benchmark_id.strip():
            raise ValueError("benchmark_id must be a non-empty string")

        if (
            not isinstance(benchmark_version, str)
            or not benchmark_version.strip()
        ):
            raise ValueError("benchmark_version must be a non-empty string")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        self.provider_factory = provider_factory
        self.record_builder = ModelQualificationRecordBuilder(policy)
        self.benchmark_id = benchmark_id
        self.benchmark_version = benchmark_version
        self.top_k = top_k

    @staticmethod
    def _validate_model(model):
        if not isinstance(model, dict):
            raise ValueError("model must be a dict")

        required = {
            "model_id",
            "model_revision",
            "provider_type",
            "embedding_dimension",
        }

        if not required.issubset(model):
            raise ValueError("model metadata is incomplete")

        for field in ("model_id", "model_revision", "provider_type"):
            value = model[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")

        dimension = model["embedding_dimension"]
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ValueError("embedding_dimension must be a positive integer")

    @staticmethod
    def _validate_chunks(chunks):
        if not isinstance(chunks, dict) or not chunks:
            raise ValueError("chunks must be a non-empty dict")

        for chunk_id, text in chunks.items():
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("chunk ids must be non-empty strings")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("chunk text must be a non-empty string")

    @classmethod
    def _validate_loaded_provider(cls, loaded, model):
        if not isinstance(loaded, LoadedEmbeddingProvider):
            raise ValueError(
                "provider_factory must return LoadedEmbeddingProvider"
            )

        observed_model = {
            "model_id": loaded.model_id,
            "model_revision": loaded.model_revision,
            "provider_type": loaded.provider_type,
            "embedding_dimension": loaded.embedding_dimension,
        }
        cls._validate_model(observed_model)

        if observed_model != {
            field: model[field]
            for field in observed_model
        }:
            raise ValueError(
                "loaded provider identity does not match model metadata"
            )

        encode_queries = getattr(
            loaded.provider,
            "encode_queries",
            None,
        )
        encode_documents = getattr(
            loaded.provider,
            "encode_documents",
            None,
        )
        encode = getattr(loaded.provider, "encode", None)

        if not callable(encode_queries) and not callable(encode):
            raise ValueError(
                "embedding provider must provide encode_queries() or encode()"
            )
        if not callable(encode_documents) and not callable(encode):
            raise ValueError(
                "embedding provider must provide encode_documents() or encode()"
            )

    def qualify(self, model, chunks, queries):
        self._validate_model(model)
        self._validate_chunks(chunks)

        HealthcareRetrievalBenchmark.validate_queries(
            queries,
            corpus_ids=chunks.keys(),
        )

        loaded = self.provider_factory(dict(model))
        self._validate_loaded_provider(loaded, model)

        retriever = Retriever(
            chunks,
            loaded.provider,
            expected_dimension=model["embedding_dimension"],
        )
        retriever.validate_provider_embeddings(
            [query["text"] for query in queries]
        )
        benchmark = HealthcareRetrievalBenchmark(
            retriever,
            top_k=self.top_k,
        )
        metrics = benchmark.evaluate(queries)

        benchmark_metadata = {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "top_k": self.top_k,
        }

        return self.record_builder.build(
            model=dict(model),
            benchmark=benchmark_metadata,
            metrics=metrics,
        )
