import torch
import numpy as np

from typing import Dict
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim


class Embedding:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)

    def transform_query(self, query: str) -> str:
        """ For retrieval, add the prompt for query (not for documents).
        """
        return f'Represent this sentence for searching relevant passages: {query}'

    def pooling(self, outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()

    def encode(self, docs):
        inputs = self.tokenizer(docs, padding=True, return_tensors='pt')
        for k, v in inputs.items():
            inputs[k] = v
        outputs = self.model(**inputs).last_hidden_state
        embeddings = self.pooling(outputs, inputs, 'cls')
        return embeddings

    def get_similarities(self, query, embeddings):
        # similarities = cos_sim(embeddings[0], embeddings[1:])
        similarities = []

        for i, v in enumerate(embeddings):
            similarity = cos_sim(query, v)
            similarities.append((i, similarity))

        return similarities