from dataclasses import dataclass


DEFAULT_NLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_NLI_MODEL_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"


def resolve_nli_model_revision(model_id, revision):
    if revision is None and model_id == DEFAULT_NLI_MODEL_ID:
        return DEFAULT_NLI_MODEL_REVISION
    return revision


@dataclass(frozen=True)
class LoadedNLIModelDescriptor:
    model_id: str
    model_revision: str
    provider_type: str
    labels: tuple


class HuggingFaceNLIProvider:
    """Real NLI provider with lazy ML imports and load identity."""

    def __init__(
        self,
        model_id=DEFAULT_NLI_MODEL_ID,
        revision=None,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        revision = resolve_nli_model_revision(model_id, revision)
        self.model_id = model_id
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=revision,
        )
        self.model.eval()
        labels = tuple(
            str(self.model.config.id2label[index]).lower()
            for index in range(len(self.model.config.id2label))
        )
        if set(labels) != {"entailment", "contradiction", "neutral"}:
            raise ValueError("loaded NLI model label contract is invalid")
        self._loaded_model_descriptor = LoadedNLIModelDescriptor(
            model_id=model_id,
            model_revision=revision,
            provider_type="huggingface-sequence-classification",
            labels=labels,
        )

    @property
    def loaded_model_descriptor(self):
        return self._loaded_model_descriptor

    def predict(self, premise, hypothesis):
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
        )
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probabilities = self._torch.softmax(logits, dim=-1)[0]
        return {
            self.model.config.id2label[index].lower(): float(probabilities[index])
            for index in range(len(probabilities))
        }


class FakeNLIProvider:
    """Deterministic NLI provider that imports no ML runtime."""

    def __init__(self, default_scores=None):
        self.default_scores = (
            default_scores
            if default_scores is not None
            else {
                "contradiction": 0.01,
                "entailment": 0.98,
                "neutral": 0.01,
            }
        )
        self.responses = {}

    def set_response(self, premise, hypothesis, scores):
        self.responses[(premise, hypothesis)] = dict(scores)

    def predict(self, premise, hypothesis):
        return dict(self.responses.get((premise, hypothesis), self.default_scores))
