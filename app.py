import ollama
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim
from typing import Dict
import torch
import numpy as np

from fastapi import FastAPI

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

    def get_similarities(self, embeddings):
        similarities = cos_sim(embeddings[0], embeddings[1:])
        return similarities
    

class LLM:
    def __init__(self, model_id):
        self.model_id = model_id

    def chat(self, messages, temperature=0.5, stop=[], top_p=0.95, top_k=0.5, stream=False):
        return ollama.chat(
            model=self.model_id,
            messages=messages,
            options={
                'top_p': top_p,
                'top_k': top_k,
                'temperature': temperature,
                'stop': stop
            },
            stream=stream
        )

class Bot:
    def __init__(self, model_id):
        self.embedding = Embedding("mixedbread-ai/mxbai-embed-large-v1")
        self.llm = LLM(model_id)
        
    def chat(self, messages, temperature=0.5, stop=[], top_p=0.95, top_k=0.5, stream=False):
        return self.llm.chat(messages, temperature, stop, top_p, top_k, stream)
    
    
bot = Bot("phi3")
print(bot.embedding.encode(["hello", "world"]))
print(bot.embedding.get_similarities(bot.embedding.encode(["hello", "world", "world", "hello"])))

stream = bot.chat([
    { "role": "system", "content": "You are an AI assistant. Called Bot" },
    { "role": "user", "content": "What is your name? and what is 2+2? and can you translate hello my name is Luca in Italian and Sicilian" },
], stream=True)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
