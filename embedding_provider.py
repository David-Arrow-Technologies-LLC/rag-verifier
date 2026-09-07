from dataclasses import dataclass


@dataclass(frozen=True)
class LoadedModelDescriptor:
    model_id: str
    model_revision: str
    provider_type: str
    embedding_dimension: int


class FakeEmbeddingProvider:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or {}

    def set_embedding(self, text, vector):
        self.embeddings[text] = list(vector)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        return [
            list(self.embeddings[text])
            for text in texts
        ]

    def encode_queries(self, texts):
        return self.encode(texts)

    def encode_documents(self, texts):
        return self.encode(texts)


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_id=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        revision=None
    ):
        from sentence_transformers import (
            SentenceTransformer
        )

        self.model = SentenceTransformer(
            model_id,
            revision=revision
        )
        dimension = self.model.get_sentence_embedding_dimension()
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ValueError(
                "loaded model did not report a valid embedding dimension"
            )
        self._loaded_model_descriptor = LoadedModelDescriptor(
            model_id=model_id,
            model_revision=revision,
            provider_type="sentence-transformer",
            embedding_dimension=dimension,
        )

    @property
    def loaded_model_descriptor(self):
        return self._loaded_model_descriptor

    def encode(self, texts):
        return self._encode_with_method(
            texts,
            "encode"
        )

    def encode_queries(self, texts):
        if (
            type(self).encode
            is not
            SentenceTransformerEmbeddingProvider.encode
        ):
            return self.encode(texts)

        return self._encode_with_method(
            texts,
            "encode_query"
        )

    def encode_documents(self, texts):
        if (
            type(self).encode
            is not
            SentenceTransformerEmbeddingProvider.encode
        ):
            return self.encode(texts)

        return self._encode_with_method(
            texts,
            "encode_document"
        )

    def _encode_with_method(
        self,
        texts,
        method_name
    ):
        if isinstance(texts, str):
            texts = [texts]

        method = getattr(
            self.model,
            method_name,
            None
        )

        if not callable(method):
            method = self.model.encode

        embeddings = method(
            texts,
            normalize_embeddings=True
        )

        return [
            vector.tolist()
            for vector in embeddings
        ]
